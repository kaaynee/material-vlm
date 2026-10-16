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


def test_extract_json_skips_thinking():
    text = 'Sure.<think>maybe {"summary": "draft"} is right</think>\n{"summary": "final"}'
    assert extract_json(text) == {"summary": "final"}
    with pytest.raises(ValueError):
        extract_json('<think>ran out of tokens while thinking about {"a": 1}')


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
    model = tmp_path / "model.yaml"
    model.write_text("model_id: other/model\nthinking: true\n")
    args = build_parser().parse_args(
        ["--images", "x", "--config", str(config), "--config", str(model), "--no-joint",
         "--temperature", "0.5", "--no-segment"]
    )
    resolved = resolve_config(args)
    assert resolved["model_id"] == "other/model"
    assert resolved["thinking"] is True
    assert resolved["joint"] is False
    assert resolved["max_new_tokens"] == 256
    assert resolved["temperature"] == 0.5
    assert resolved["segment"] is False


def test_compare_runs(image_dir, tmp_path):
    from material_vlm.compare import main as compare_main

    paths = collect_images([image_dir])
    for name, answer in [("model-a", json.dumps(ANSWER)), ("model-b", "no json")]:
        results = analyze(FakeVLM(answer), paths, "prompt", segmenter=FakeSegmenter(),
                          artifacts_dir=tmp_path / name)
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"model_id": name, "prompt": "prompt", "results": results})
        )
    out = tmp_path / "comparison"
    compare_main([str(tmp_path / "model-a.json"), str(tmp_path / "model-b.json"), "--out", str(out)])

    summary = json.loads((out / "summary.json").read_text())
    assert [(s["run"], s["parsed"]) for s in summary] == [("model-a", 2), ("model-b", 0)]
    assert summary[0]["defects"] == {"biological_decay": 2, "crack": 2}
    grid = Image.open(out / "0000_a.jpg")
    assert grid.width == 2 * 640


# --- quick wins: zoom pass, SAM 3, overview ---------------------------------

from material_vlm.pipeline import overview, segment_regions, tile_grid  # noqa: E402

WALL = {
    "summary": "wall",
    "materials": [
        {"name": "brick", "boxes": [[0, 0, 100, 100]], "condition": "fair",
         "defects": [{"type": "crack", "severity": "low", "boxes": [[10, 10, 20, 50]]}]},
    ],
}


class ZoomVLM(FakeVLM):
    """Answers the main pass with ``answer`` and the n-th zoom crop with ``zooms[n]``.

    Boxes are in pixels of the image the model sees (input size = image size).
    """

    def __init__(self, answer, zooms):
        super().__init__(answer)
        self.zooms = list(zooms)

    def generate(self, images, prompt, system_prompt="", **kwargs):
        self.calls.append(len(images))
        if "close-up" in system_prompt:
            return self.zooms.pop(0), [im.size for im in images]
        return self.answer, [im.size for im in images]


class FakeDefectSegmenter:
    def __init__(self, detections):
        self.detections = detections  # defect type -> list of boxes

    def detect(self, image, defect_type):
        out = []
        for box in self.detections.get(defect_type, []):
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[box[1]:box[3], box[0]:box[2]] = True
            out.append({"mask": mask, "box": [float(v) for v in box], "score": 0.8,
                        "concept": defect_type})
        return out


def test_tile_grid_covers_image_with_overlap():
    tiles = tile_grid((200, 100), 2, 0.2)
    assert len(tiles) == 4
    assert tiles[0][:2] == (0, 0) and tiles[-1][2:] == (200, 100)
    assert tiles[1][0] < tiles[0][2]  # neighbours overlap


def test_zoom_pass_adds_new_defects_and_skips_duplicates(tmp_path):
    path = tmp_path / "wall.png"
    Image.new("RGB", (100, 100)).save(path)
    # 2 x 2 tiles with 15% overlap on 100 px: origins at 0 and 46, size 54.
    # Tile 0 reports the known crack again and a rust spot in the overlap area;
    # tile 3 sees the same rust spot in its own coordinates.
    empty = json.dumps({"defects": []})
    zooms = [
        json.dumps({"defects": [
            {"material": "brick", "type": "crack", "boxes": [[10, 10, 20, 50]]},
            {"material": "Brick", "type": "rust", "severity": "high", "boxes": [[48, 48, 53, 53]]},
        ]}),
        empty,
        "not json",
        json.dumps({"defects": [
            {"material": "brick", "type": "corrosion", "boxes": [[2, 2, 7, 7]]},
            # a box over the whole crop says nothing about location: dropped
            {"material": "brick", "type": "biological_decay", "boxes": [[0, 0, 54, 54]]},
        ]}),
    ]
    vlm = ZoomVLM(json.dumps(WALL), zooms)
    record = analyze(vlm, [path], "prompt", zoom_tiles=2)
    record = record[0]
    assert vlm.calls == [1, 1, 1, 1, 1]
    defects = record["analysis"]["materials"][0]["defects"]
    assert [(d["type"], d["source"]) for d in defects] == [("crack", "vlm"), ("corrosion", "zoom")]
    assert defects[1]["boxes"] == [[48, 48, 53, 53]]
    assert len(record["zoom_raw_outputs"]) == 4


def test_sam3_masks_and_discovery():
    materials = json.loads(json.dumps(WALL))["materials"]
    for d in materials[0]["defects"]:
        d["source"] = "vlm"
    image = Image.new("RGB", (100, 100))
    sam3 = FakeDefectSegmenter({
        "crack": [[12, 12, 18, 48]],           # inside the VLM crack box -> its mask
        "corrosion": [[60, 60, 80, 80]],      # on the brick, missed by the VLM -> discovered
        "biological_decay": [[200, 200, 210, 210]],  # outside every material -> ignored
    })
    masks = segment_regions(materials, image, FakeSegmenter(), sam3, discover=True)

    brick = materials[0]
    crack, rust = brick["defects"]
    assert crack["mask_source"] == "sam3" and crack["area_fraction"] == 6 * 36 / 100**2
    assert brick["mask_source"] == "sam2"
    assert (rust["type"], rust["source"], rust["boxes"]) == ("corrosion", "sam3", [[60, 60, 80, 80]])
    assert masks[("defect", 0, 1)].sum() == 400


def test_overview_aggregates_materials_across_images():
    a = {"analysis": {"materials": [
        {"name": "Brick", "condition": "fair", "defects": [{"type": "crack"}]}]}}
    b = {"analysis": {"materials": [
        {"name": "brick ", "condition": "poor", "defects": [{"type": "crack"}, {"type": "other"}]},
        {"name": "steel", "condition": "good", "defects": []}]}}
    failed = {"analysis": None}
    brick, steel = overview([a, b, failed])
    assert (brick["name"], brick["images"], brick["worst_condition"]) == ("brick", 2, "poor")
    assert brick["defects"] == {"crack": 2, "other": 1}
    assert steel["worst_condition"] == "good"


def test_find_material_prefers_position_over_name():
    from material_vlm.pipeline import _find_material

    wall = {"name": "brick wall", "boxes": [[0, 0, 100, 100]]}
    door = {"name": "steel door", "boxes": [[60, 20, 90, 100]]}
    plate = {"name": "steel plate", "boxes": [[200, 0, 300, 100]]}
    materials = [wall, door, plate]
    assert _find_material(materials, "brick wall", [70, 50, 80, 60]) is wall   # named + contains
    assert _find_material(materials, "concrete", [70, 50, 80, 60]) is door     # smallest container
    assert _find_material(materials, "brick wall", [250, 50, 260, 60]) is plate  # position beats name
    assert _find_material(materials, "steel door", [500, 50, 510, 60]) is door   # name as fallback
    assert _find_material(materials, "glass", [500, 50, 510, 60]) is None
