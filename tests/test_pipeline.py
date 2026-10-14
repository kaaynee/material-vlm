import json

import numpy as np
import pytest
from PIL import Image

from material_vlm.cli import build_parser, resolve_config
from material_vlm.pipeline import analyze, collect_images
from material_vlm.schema import extract_json, parse_analysis
from material_vlm.segmentation import scale_boxes

# Boxes are in the frame of a 32x16 model input; test images are 64x32.
ANSWER = {
    "summary": "A weathered brick wall with a steel door.",
    "materials": [
        {
            "name": "clay brick",
            "category": "masonry",
            "boxes": [[0, 0, 16, 16]],
            "confidence": 0.9,
            "condition": "Fair",
            "evidence": "red rectangular units with mortar joints",
            "defects": [
                {"type": "moss", "severity": "Medium", "description": "green growth",
                 "boxes": [[2, 2, 6, 6]]},
                "hairline cracks",
            ],
        },
        {"name": "steel", "confidence": 1.7, "condition": "rusty", "boxes": [16, 0, 32, 16]},
    ],
}


class FakeVLM:
    def __init__(self, answer, input_size=(32, 16)):
        self.answer = answer
        self.input_size = input_size
        self.calls = []

    def generate(self, images, prompt, **kwargs):
        self.calls.append(len(images))
        return self.answer, [self.input_size] * len(images)


class FakeSegmenter:
    """Returns each box filled in as its mask."""

    def segment(self, image, boxes):
        masks = []
        for x1, y1, x2, y2 in boxes:
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[int(y1):int(y2), int(x1):int(x2)] = True
            masks.append((mask, 0.9))
        return masks


@pytest.fixture
def image_dir(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("b.png", "a.jpg"):
        Image.new("RGB", (64, 32), "gray").save(images / name)
    (images / "notes.txt").write_text("not an image")
    return images


def test_extract_json_from_fenced_output():
    text = f"Here is the result:\n```json\n{json.dumps(ANSWER)}\n```"
    assert extract_json(text) == ANSWER


def test_extract_json_without_object():
    with pytest.raises(ValueError):
        extract_json("I cannot see any image.")


def test_parse_analysis_normalizes_fields():
    analysis = parse_analysis(json.dumps(ANSWER))
    brick, steel = analysis.materials
    assert brick.condition == "fair"
    moss, crack = brick.defects
    assert (moss.type, moss.severity) == ("biological_decay", "medium")
    assert (crack.type, crack.description, crack.boxes) == ("crack", "hairline cracks", [])
    assert steel.condition == "unknown"
    assert steel.confidence == 1.0
    assert steel.boxes == [[16.0, 0.0, 32.0, 16.0]]  # single box wrapped in a list


def test_scale_boxes():
    assert scale_boxes([[10, 5, 20, 10]], "input_pixels", (100, 50), (200, 100)) == [
        [20, 10, 40, 20]
    ]
    assert scale_boxes([[500, 0, 1000, 500]], "normalized_1000", (1, 1), (200, 100)) == [
        [100, 0, 200, 50]
    ]
    # swapped corners are reordered, out-of-range values clipped, empty boxes dropped
    assert scale_boxes([[30, 40, -5, 0], [5, 5, 5, 9]], "input_pixels", (20, 20), (20, 20)) == [
        [0, 0, 20, 20]
    ]


def test_collect_images(image_dir):
    assert [p.name for p in collect_images([image_dir])] == ["a.jpg", "b.png"]
    with pytest.raises(FileNotFoundError):
        collect_images([image_dir / "missing.png"])


def test_analyze_per_image_and_joint(image_dir):
    paths = collect_images([image_dir])

    vlm = FakeVLM(json.dumps(ANSWER))
    results = analyze(vlm, paths, "prompt")
    assert vlm.calls == [1, 1]
    brick = results[0]["analysis"]["materials"][0]
    assert brick["name"] == "clay brick"
    assert brick["boxes"] == [[0, 0, 32, 32]]  # scaled from 32x16 input to 64x32 image
    assert results[0]["overlays"] == {}

    vlm = FakeVLM(json.dumps(ANSWER))
    results = analyze(vlm, paths, "prompt", joint=True)
    assert vlm.calls == [2]
    assert len(results[0]["images"]) == 2
    assert results[0]["analysis"]["materials"][0]["boxes"] == []


def test_analyze_writes_masks_and_overlays(image_dir, tmp_path):
    paths = collect_images([image_dir])
    results = analyze(
        FakeVLM(json.dumps(ANSWER)),
        paths[:1],
        "prompt",
        segmenter=FakeSegmenter(),
        artifacts_dir=tmp_path / "run",
    )
    record = results[0]
    brick, steel = record["analysis"]["materials"]
    assert brick["area_fraction"] == 0.5
    assert brick["mask_score"] == 0.9
    mask = np.array(Image.open(brick["mask"]))
    assert mask.shape == (32, 64) and mask[:, :32].all() and not mask[:, 32:].any()

    moss, crack = brick["defects"]
    assert moss["mask"].endswith("defect_00_00_biological_decay.png")
    assert crack["mask"] is None and crack["area_fraction"] == 0.0

    for path in record["overlays"].values():
        assert Image.open(path).size == (64, 32)


def test_analyze_keeps_raw_output_on_parse_failure(image_dir):
    results = analyze(FakeVLM("no json here"), collect_images([image_dir]), "prompt")
    assert results[0]["analysis"] is None
    assert results[0]["parse_error"]
    assert results[0]["raw_output"] == "no json here"


def test_cli_overrides_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("model_id: some/model\njoint: true\nmax_new_tokens: 256\n")
    args = build_parser().parse_args(
        ["--images", "x", "--config", str(config), "--no-joint", "--temperature", "0.5",
         "--no-segment"]
    )
    resolved = resolve_config(args)
    assert resolved["model_id"] == "some/model"
    assert resolved["joint"] is False
    assert resolved["max_new_tokens"] == 256
    assert resolved["temperature"] == 0.5
    assert resolved["segment"] is False
