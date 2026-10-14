"""Image loading and the prompt + images -> material analysis pipeline."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .schema import build_system_prompt, parse_analysis
from .segmentation import scale_boxes
from .visualize import defect_items, draw_overlay, material_items

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(paths: list[str | Path]) -> list[Path]:
    """Expand files and directories into a sorted list of image files."""
    images: list[Path] = []
    for path in map(Path, paths):
        if path.is_dir():
            images.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
            )
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"image path does not exist: {path}")
    if not images:
        raise FileNotFoundError("no images found in the given paths")
    return images


def load_image(path: Path, max_side: int | None = None) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "item"


def localize(
    analysis: dict,
    image_path: Path,
    input_size: tuple[int, int],
    box_coords: str,
    segmenter=None,
    artifacts_dir: Path | None = None,
    name: str | None = None,
) -> dict[str, str]:
    """Convert boxes to original image pixels, segment them and write artifacts.

    Modifies ``analysis`` in place (boxes, and mask fields when segmenting) and
    returns the paths of the written overlay images.
    """
    image = load_image(image_path)
    materials = analysis["materials"]

    # Every region that gets a mask: materials and the defects within them.
    regions = []
    for i, m in enumerate(materials):
        m["boxes"] = scale_boxes(m["boxes"], box_coords, input_size, image.size)
        regions.append((("material", i), m))
        for j, d in enumerate(m["defects"]):
            d["boxes"] = scale_boxes(d["boxes"], box_coords, input_size, image.size)
            regions.append((("defect", i, j), d))

    masks: dict[tuple, np.ndarray] = {}
    if segmenter is not None:
        all_boxes = [box for _, region in regions for box in region["boxes"]]
        results = iter(segmenter.segment(image, all_boxes))
        for key, region in regions:
            if not region["boxes"]:
                region.update(mask=None, mask_score=None, area_fraction=0.0)
                continue
            parts = [next(results) for _ in region["boxes"]]
            mask = np.logical_or.reduce([m for m, _ in parts])
            masks[key] = mask
            region.update(
                mask=None,
                mask_score=round(float(np.mean([s for _, s in parts])), 3),
                area_fraction=round(float(mask.mean()), 4),
            )

    if artifacts_dir is None:
        return {}
    name = name or image_path.stem
    mask_dir = artifacts_dir / "masks" / name
    for key, region in regions:
        if key not in masks:
            continue
        if key[0] == "material":
            filename = f"material_{key[1]:02d}_{_slug(region['name'])}.png"
        else:
            filename = f"defect_{key[1]:02d}_{key[2]:02d}_{region['type']}.png"
        mask_dir.mkdir(parents=True, exist_ok=True)
        path = mask_dir / filename
        Image.fromarray(masks[key].astype(np.uint8) * 255).save(path)
        region["mask"] = str(path)

    overlay_dir = artifacts_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlays = {
        "materials": overlay_dir / f"{name}_materials.jpg",
        "defects": overlay_dir / f"{name}_defects.jpg",
    }
    draw_overlay(image, material_items(materials, masks)).save(overlays["materials"], quality=90)
    draw_overlay(image, defect_items(materials, masks)).save(overlays["defects"], quality=90)
    return {kind: str(path) for kind, path in overlays.items()}


def analyze(
    vlm,
    image_paths: list[Path],
    prompt: str,
    joint: bool = False,
    max_image_side: int | None = 1280,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    thinking: bool = False,
    box_coords: str = "input_pixels",
    segmenter=None,
    artifacts_dir: Path | None = None,
) -> list[dict]:
    """Run the VLM and return one result record per request.

    With ``joint=False`` every image is analyzed on its own; with ``joint=True``
    all images are sent in a single request and one record is returned. Boxes,
    masks and overlays are only produced per image, since boxes in a joint
    request cannot be attributed to a specific image.
    """
    system_prompt = build_system_prompt(box_coords)
    groups = [image_paths] if joint else [[p] for p in image_paths]
    results = []
    for index, group in enumerate(groups):
        images = [load_image(p, max_image_side) for p in group]
        start = time.perf_counter()
        raw, input_sizes = vlm.generate(
            images,
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            thinking=thinking,
        )
        record = {
            "images": [str(p) for p in group],
            "vlm_seconds": round(time.perf_counter() - start, 2),
            "analysis": None,
            "overlays": {},
            "parse_error": None,
            "raw_output": raw,
        }
        try:
            analysis = parse_analysis(raw).model_dump()
        except ValueError as exc:  # includes pydantic.ValidationError
            record["parse_error"] = str(exc)
            results.append(record)
            continue

        if joint:
            for m in analysis["materials"]:
                m["boxes"] = []
                for d in m["defects"]:
                    d["boxes"] = []
        else:
            record["overlays"] = localize(
                analysis,
                group[0],
                input_sizes[0],
                box_coords,
                segmenter=segmenter,
                artifacts_dir=artifacts_dir,
                name=f"{index:04d}_{group[0].stem}",
            )
        record["analysis"] = analysis
        results.append(record)
    return results
