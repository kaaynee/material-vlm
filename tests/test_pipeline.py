import json

import pytest
from PIL import Image

from material_vlm.cli import build_parser, resolve_config
from material_vlm.pipeline import analyze, collect_images
from material_vlm.schema import extract_json, parse_analysis

ANSWER = {
    "summary": "A weathered brick wall with a steel door.",
    "materials": [
        {
            "name": "clay brick",
            "category": "masonry",
            "location": "wall",
            "confidence": 0.9,
            "condition": "Fair",
            "defects": ["efflorescence"],
            "evidence": "red rectangular units with mortar joints",
        },
        {"name": "steel", "confidence": 1.7, "condition": "rusty"},
    ],
}


class FakeVLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, images, prompt, **kwargs):
        self.calls.append(len(images))
        return self.answer


@pytest.fixture
def image_dir(tmp_path):
    for name in ("b.png", "a.jpg"):
        Image.new("RGB", (64, 32), "gray").save(tmp_path / name)
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


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
    assert steel.condition == "unknown"
    assert steel.confidence == 1.0
    assert steel.defects == []


def test_collect_images(image_dir):
    assert [p.name for p in collect_images([image_dir])] == ["a.jpg", "b.png"]
    with pytest.raises(FileNotFoundError):
        collect_images([image_dir / "missing.png"])


def test_analyze_per_image_and_joint(image_dir):
    paths = collect_images([image_dir])

    vlm = FakeVLM(json.dumps(ANSWER))
    results = analyze(vlm, paths, "prompt")
    assert vlm.calls == [1, 1]
    assert results[0]["analysis"]["materials"][0]["name"] == "clay brick"

    vlm = FakeVLM(json.dumps(ANSWER))
    results = analyze(vlm, paths, "prompt", joint=True)
    assert vlm.calls == [2]
    assert len(results[0]["images"]) == 2


def test_analyze_keeps_raw_output_on_parse_failure(image_dir):
    results = analyze(FakeVLM("no json here"), collect_images([image_dir]), "prompt")
    assert results[0]["analysis"] is None
    assert results[0]["parse_error"]
    assert results[0]["raw_output"] == "no json here"


def test_cli_overrides_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("model_id: some/model\njoint: true\nmax_new_tokens: 256\n")
    args = build_parser().parse_args(
        ["--images", "x", "--config", str(config), "--no-joint", "--temperature", "0.5"]
    )
    resolved = resolve_config(args)
    assert resolved["model_id"] == "some/model"
    assert resolved["joint"] is False
    assert resolved["max_new_tokens"] == 256
    assert resolved["temperature"] == 0.5
