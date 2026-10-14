"""Structured output of the material analysis and parsing of raw model text."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator

CONDITIONS = ("good", "fair", "poor", "unknown")
DEFECT_TYPES = ("crack", "corrosion", "biological_decay", "other")
SEVERITIES = ("low", "medium", "high", "unknown")

# Keywords mapping free-form defect names onto DEFECT_TYPES.
DEFECT_KEYWORDS = {
    "crack": ("crack", "fracture", "fissure"),
    "corrosion": ("corrosion", "corroded", "rust", "oxid"),
    "biological_decay": (
        "biological", "bio", "mold", "mould", "moss", "algae", "lichen",
        "fung", "rot", "decay", "vegetation", "growth",
    ),
}

BOX_INSTRUCTIONS = {
    "input_pixels": "absolute pixel coordinates [x1, y1, x2, y2] of the image",
    "normalized_1000": "coordinates [x1, y1, x2, y2] normalized to 0-1000 for both axes",
}

SYSTEM_PROMPT_TEMPLATE = """You are an expert in material recognition and condition assessment.
Answer with a single JSON object and nothing else, using exactly this structure:

{{
  "summary": "<one or two sentences describing the scene and its overall state>",
  "materials": [
    {{
      "name": "<specific material, e.g. 'reinforced concrete', 'oak wood', 'galvanized steel'>",
      "category": "<broad class, e.g. 'concrete', 'wood', 'metal', 'glass', 'plastic', 'masonry', 'textile'>",
      "boxes": [[x1, y1, x2, y2], ...],
      "confidence": <number between 0 and 1>,
      "condition": "<one of: good, fair, poor, unknown>",
      "evidence": "<visual cues supporting the material and condition assessment>",
      "defects": [
        {{
          "type": "<one of: crack, corrosion, biological_decay, other>",
          "severity": "<one of: low, medium, high>",
          "description": "<what is visible, e.g. 'hairline crack along mortar joint', 'green algae growth'>",
          "boxes": [[x1, y1, x2, y2], ...]
        }}
      ]
    }}
  ]
}}

"boxes" are tight bounding boxes in {box_instruction}: one box per visible region of
the material, and one box per affected area for each defect.
"biological_decay" covers mold, moss, algae, lichen, fungi and rot.
Use an empty list for "defects" if none are visible and an empty list for "materials" if no material can be identified."""


def build_system_prompt(box_coords: str = "input_pixels") -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(box_instruction=BOX_INSTRUCTIONS[box_coords])


def _clean_boxes(value) -> list[list[float]]:
    """Keep well-formed [x1, y1, x2, y2] boxes and silently drop the rest."""
    if not isinstance(value, list):
        return []
    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        value = [value]  # a single box instead of a list of boxes
    boxes = []
    for box in value:
        if isinstance(box, dict):
            box = box.get("bbox_2d") or box.get("box")
        if isinstance(box, list) and len(box) == 4:
            try:
                boxes.append([float(v) for v in box])
            except (TypeError, ValueError):
                pass
    return boxes


def normalize_defect_type(value) -> str:
    value = str(value).strip().lower().replace(" ", "_")
    if value in DEFECT_TYPES:
        return value
    for defect_type, keywords in DEFECT_KEYWORDS.items():
        if any(k in value for k in keywords):
            return defect_type
    return "other"


class Defect(BaseModel):
    type: Literal["crack", "corrosion", "biological_decay", "other"] = "other"
    severity: Literal["low", "medium", "high", "unknown"] = "unknown"
    description: str = ""
    boxes: list[list[float]] = Field(default_factory=list)

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value):
        return normalize_defect_type(value)

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, value):
        value = str(value).strip().lower()
        return value if value in SEVERITIES else "unknown"

    @field_validator("boxes", mode="before")
    @classmethod
    def _boxes(cls, value):
        return _clean_boxes(value)


class Material(BaseModel):
    name: str
    category: str = "unknown"
    boxes: list[list[float]] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    condition: Literal["good", "fair", "poor", "unknown"] = "unknown"
    evidence: str = ""
    defects: list[Defect] = Field(default_factory=list)

    @field_validator("condition", mode="before")
    @classmethod
    def _normalize_condition(cls, value):
        value = str(value).strip().lower()
        return value if value in CONDITIONS else "unknown"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value):
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("boxes", mode="before")
    @classmethod
    def _boxes(cls, value):
        return _clean_boxes(value)

    @field_validator("defects", mode="before")
    @classmethod
    def _defects(cls, value):
        if not isinstance(value, list):
            return []
        # Plain strings ("rust stains") become defects classified by keyword.
        return [{"type": d, "description": d} if isinstance(d, str) else d for d in value]


class MaterialAnalysis(BaseModel):
    summary: str = ""
    materials: list[Material] = Field(default_factory=list)


def extract_json(text: str) -> dict:
    """Return the first JSON object found in the model output.

    Models often wrap the JSON in a markdown fence or add a sentence around it,
    so scan for the first position that decodes to an object.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in model output")


def parse_analysis(text: str) -> MaterialAnalysis:
    return MaterialAnalysis.model_validate(extract_json(text))
