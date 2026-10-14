"""Overlay images showing where materials and defects were detected."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Distinct colors for materials, cycled if there are more materials.
MATERIAL_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189), (140, 86, 75),
    (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207), (214, 39, 40),
]
DEFECT_COLORS = {
    "crack": (230, 25, 75),
    "corrosion": (245, 130, 48),
    "biological_decay": (60, 180, 75),
    "other": (255, 225, 25),
}
CONDITION_COLORS = {
    "good": (60, 180, 75),
    "fair": (255, 190, 0),
    "poor": (230, 25, 75),
    "unknown": (160, 160, 160),
}


def _font(image: Image.Image) -> ImageFont.ImageFont:
    size = max(12, round(max(image.size) / 60))
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _blend_mask(image: Image.Image, mask: np.ndarray, color, alpha: float = 0.45) -> Image.Image:
    layer = Image.new("RGB", image.size, color)
    mask_img = Image.fromarray((mask * round(alpha * 255)).astype(np.uint8))
    return Image.composite(layer, image, mask_img)


def _label(draw: ImageDraw.ImageDraw, xy, text: str, color, font) -> None:
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=color)
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def draw_overlay(image: Image.Image, items: list[dict]) -> Image.Image:
    """Draw items with keys ``boxes``, ``mask`` (bool array or None), ``color``,
    ``label`` and optionally ``label_color`` onto a copy of the image."""
    overlay = image.convert("RGB")
    for item in items:
        if item.get("mask") is not None:
            overlay = _blend_mask(overlay, item["mask"], item["color"])
    draw = ImageDraw.Draw(overlay)
    font = _font(overlay)
    width = max(2, round(max(overlay.size) / 400))
    for item in items:
        for x1, y1, x2, y2 in item["boxes"]:
            draw.rectangle((x1, y1, x2, y2), outline=item["color"], width=width)
        if item["boxes"]:
            x1, y1 = item["boxes"][0][:2]
            _label(draw, (x1 + width, y1 + width), item["label"],
                   item.get("label_color", item["color"]), font)
    return overlay


def material_items(materials: list[dict], masks: dict) -> list[dict]:
    """Overlay items for materials: one color per material, label colored by condition."""
    return [
        {
            "boxes": m["boxes"],
            "mask": masks.get(("material", i)),
            "color": MATERIAL_COLORS[i % len(MATERIAL_COLORS)],
            "label": f"{m['name']} | {m['condition']}",
            "label_color": CONDITION_COLORS[m["condition"]],
        }
        for i, m in enumerate(materials)
    ]


def defect_items(materials: list[dict], masks: dict) -> list[dict]:
    """Overlay items for defects, colored by defect type."""
    items = []
    for i, m in enumerate(materials):
        for j, d in enumerate(m["defects"]):
            items.append(
                {
                    "boxes": d["boxes"],
                    "mask": masks.get(("defect", i, j)),
                    "color": DEFECT_COLORS[d["type"]],
                    "label": f"{d['type']} | {d['severity']} ({m['name']})",
                }
            )
    return items
