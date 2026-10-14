"""Structured output of the material analysis and parsing of raw model text."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator

CONDITIONS = ("good", "fair", "poor", "unknown")

SYSTEM_PROMPT = """You are an expert in material recognition and condition assessment.
Answer with a single JSON object and nothing else, using exactly this structure:

{
  "summary": "<one or two sentences describing the scene and its overall state>",
  "materials": [
    {
      "name": "<specific material, e.g. 'reinforced concrete', 'oak wood', 'galvanized steel'>",
      "category": "<broad class, e.g. 'concrete', 'wood', 'metal', 'glass', 'plastic', 'masonry', 'textile'>",
      "location": "<where in the image(s) the material appears>",
      "confidence": <number between 0 and 1>,
      "condition": "<one of: good, fair, poor, unknown>",
      "defects": ["<visible defect>", ...],
      "evidence": "<visual cues supporting the material and condition assessment>"
    }
  ]
}

Use an empty list for "defects" if none are visible and an empty list for "materials" if no material can be identified."""


class Material(BaseModel):
    name: str
    category: str = "unknown"
    location: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    condition: Literal["good", "fair", "poor", "unknown"] = "unknown"
    defects: list[str] = Field(default_factory=list)
    evidence: str = ""

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
