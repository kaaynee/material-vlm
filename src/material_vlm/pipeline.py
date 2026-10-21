"""Image loading and the prompt + images -> material analysis pipeline."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .schema import (
    CONDITIONS,
    VERIFY_DEFECT_TYPES,
    VERIFY_INSTRUCTION,
    build_system_prompt,
    build_verify_prompt,
    build_zoom_prompt,
    parse_analysis,
    parse_verification,
    parse_zoom,
)
from .segmentation import scale_boxes
from .visualize import defect_items, draw_overlay, material_items, outline_region, rejected_items

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

ZOOM_INSTRUCTION = "This is a close-up crop of the image. List the defects visible in it."
# Zoom boxes covering more of the crop than this are not a localization.
ZOOM_MAX_BOX_FRACTION = 0.6
# Verification crops are at least this fraction of the image's longer side, so
# small defects are shown with enough surrounding context.
VERIFY_MIN_CROP_FRACTION = 0.15


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
    return _downscale(image, max_side)


def _downscale(image: Image.Image, max_side: int | None) -> Image.Image:
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "item"


# --- boxes ---------------------------------------------------------------------


def _area(box: list[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _box_overlap(a: list[float], b: list[float]) -> float:
    """Intersection over the smaller box: 1.0 if one box contains the other."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / min(_area(a), _area(b))


def _contains(box: list[float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _box_mask(shape: tuple[int, int], boxes: list[list[float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))] = True
    return mask


def _mask_at(mask: np.ndarray, x: float, y: float) -> bool:
    return 0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1] and bool(mask[int(y), int(x)])


def scale_analysis(analysis: dict, box_coords: str, input_size, image_size) -> None:
    """Convert all VLM boxes in place to pixel coordinates of the original image."""
    for m in analysis["materials"]:
        m["boxes"] = scale_boxes(m["boxes"], box_coords, input_size, image_size)
        for d in m["defects"]:
            d["boxes"] = scale_boxes(d["boxes"], box_coords, input_size, image_size)
            d["source"] = "vlm"


# --- zoom pass -------------------------------------------------------------------


def tile_grid(size: tuple[int, int], tiles: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Split an image into ``tiles`` x ``tiles`` crops that overlap by ``overlap``."""
    width, height = size
    tile_w = width / (tiles - (tiles - 1) * overlap)
    tile_h = height / (tiles - (tiles - 1) * overlap)
    return [
        (
            round(i * tile_w * (1 - overlap)),
            round(j * tile_h * (1 - overlap)),
            min(width, round(i * tile_w * (1 - overlap) + tile_w)),
            min(height, round(j * tile_h * (1 - overlap) + tile_h)),
        )
        for j in range(tiles)
        for i in range(tiles)
    ]


def _find_material(materials: list[dict], name: str, box: list[float]) -> dict | None:
    """The material the box lies on.

    Among materials whose boxes contain the box center, the named one wins,
    otherwise the one with the smallest box (e.g. a door rather than the wall
    around it). Only if no material contains it does the name alone decide.
    """
    name = name.strip().lower()
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    candidates = [
        (m["name"].strip().lower() != name, min(_area(b) for b in m["boxes"] if _contains(b, cx, cy)), k)
        for k, m in enumerate(materials)
        if any(_contains(b, cx, cy) for b in m["boxes"])
    ]
    if candidates:
        return materials[min(candidates)[2]]
    return next((m for m in materials if m["name"].strip().lower() == name), None)


def merge_defect(materials: list[dict], defect: dict, boxes: list[list[float]], source: str) -> bool:
    """Add a defect found outside the main pass unless it duplicates a known one.

    Boxes overlapping an existing defect of the same type are dropped. Returns
    whether anything was added (as the last defect of its material).
    """
    material = _find_material(materials, defect.get("material", ""), boxes[0])
    if material is None:
        return False
    existing = [
        b for m in materials for d in m["defects"] if d["type"] == defect["type"] for b in d["boxes"]
    ]
    new_boxes = []
    for box in boxes:
        if all(_box_overlap(box, b) < 0.5 for b in existing + new_boxes):
            new_boxes.append(box)
    if not new_boxes:
        return False
    material["defects"].append(
        {
            "type": defect["type"],
            "severity": defect.get("severity", "unknown"),
            "description": defect.get("description", ""),
            "boxes": new_boxes,
            "source": source,
        }
    )
    return True


def zoom_pass(vlm, image, analysis, prompt, tiles, overlap, max_side, box_coords, gen) -> list[str]:
    """Look for defects in overlapping high-resolution crops and merge them in."""
    materials = analysis["materials"]
    system_prompt = build_zoom_prompt(box_coords, [m["name"] for m in materials])
    raw_outputs = []
    for x0, y0, x1, y1 in tile_grid(image.size, tiles, overlap):
        crop = _downscale(image.crop((x0, y0, x1, y1)), max_side)
        raw, input_sizes = vlm.generate(
            [crop], f"{prompt}\n\n{ZOOM_INSTRUCTION}", system_prompt=system_prompt, **gen
        )
        raw_outputs.append(raw)
        try:
            defects = parse_zoom(raw).defects
        except ValueError:
            continue
        tile_area = (x1 - x0) * (y1 - y0)
        for d in defects:
            boxes = scale_boxes(d.boxes, box_coords, input_sizes[0], (x1 - x0, y1 - y0))
            boxes = [
                [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0]
                for bx1, by1, bx2, by2 in boxes
                if (bx2 - bx1) * (by2 - by1) <= ZOOM_MAX_BOX_FRACTION * tile_area
            ]
            if boxes:
                merge_defect(materials, d.model_dump(), boxes, source="zoom")
    return raw_outputs


# --- verification ----------------------------------------------------------------


def verification_views(
    image: Image.Image, mask: np.ndarray | None, boxes: list[list[float]], padding: float, max_side: int | None
) -> tuple[Image.Image, Image.Image]:
    """A padded crop around a region, plain and with the region outlined."""
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    else:
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    # Padded per axis, so a long thin region (a crack, a joint) does not pull in
    # unrelated surface that could distract the model.
    min_side = VERIFY_MIN_CROP_FRACTION * max(image.size)
    crop_w = max((x2 - x1) * (1 + 2 * padding), min_side)
    crop_h = max((y2 - y1) * (1 + 2 * padding), min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = max(0, round(cx - crop_w / 2)), max(0, round(cy - crop_h / 2))
    right, bottom = min(image.width, round(cx + crop_w / 2)), min(image.height, round(cy + crop_h / 2))

    crop = image.crop((left, top, right, bottom))
    crop_mask = mask[top:bottom, left:right] if mask is not None else None
    crop_boxes = [[bx1 - left, by1 - top, bx2 - left, by2 - top] for bx1, by1, bx2, by2 in boxes]
    marked = outline_region(crop, crop_mask, crop_boxes)
    return _downscale(crop, max_side), _downscale(marked, max_side)


def verify_defects(vlm, image, materials, masks, padding, max_side, limit, gen) -> list[str]:
    """Ask the VLM what each defect candidate actually shows.

    Every defect gets a ``verification`` entry (None if it was not checked: no
    boxes, over the per-image limit or an unparseable answer). Returns the raw
    model outputs.
    """
    raw_outputs = []
    checked = 0
    for i, m in enumerate(materials):
        system_prompt = build_verify_prompt(m["name"])
        for j, d in enumerate(m["defects"]):
            d["verification"] = None
            if not d["boxes"] or checked >= limit:
                continue
            checked += 1
            views = verification_views(image, masks.get(("defect", i, j)), d["boxes"], padding, max_side)
            raw, _ = vlm.generate(list(views), VERIFY_INSTRUCTION, system_prompt=system_prompt, **gen)
            raw_outputs.append(raw)
            try:
                answer = parse_verification(raw)
            except ValueError:
                continue
            d["verification"] = {**answer.model_dump(), "original_type": d["type"]}
    return raw_outputs


def apply_verification(materials: list[dict], masks: dict) -> tuple[dict, list[dict]]:
    """Drop rejected defects and relabel accepted ones.

    A defect is accepted if the verifier named a defect type (which replaces the
    original type, along with its severity), rejected if it named a look-alike
    and kept unverified otherwise. Returns the masks re-keyed to the remaining
    defects, with rejected ones under ``("rejected", k)``, and the rejected
    defects with their material name.
    """
    new_masks = {key: mask for key, mask in masks.items() if key[0] == "material"}
    rejected = []
    for i, m in enumerate(materials):
        kept = []
        for j, d in enumerate(m["defects"]):
            mask = masks.get(("defect", i, j))
            v = d.get("verification")
            label = v["label"] if v else "unclear"
            if label in VERIFY_DEFECT_TYPES:
                d["verified"] = True
                d["type"] = VERIFY_DEFECT_TYPES[label]
                if v["severity"] != "unknown":
                    d["severity"] = v["severity"]
            elif label == "unclear":
                d["verified"] = None
            else:
                d["verified"] = False
                if mask is not None:
                    new_masks[("rejected", len(rejected))] = mask
                rejected.append({**d, "material": m["name"]})
                continue
            if mask is not None:
                new_masks[("defect", i, len(kept))] = mask
            kept.append(d)
        m["defects"] = kept
    return new_masks, rejected


# --- masks and artifacts -----------------------------------------------------------


def _regions(materials: list[dict]) -> list[tuple[tuple, dict]]:
    regions = []
    for i, m in enumerate(materials):
        regions.append((("material", i), m))
        for j, d in enumerate(m["defects"]):
            regions.append((("defect", i, j), d))
    return regions


def segment_regions(
    materials: list[dict], image: Image.Image, segmenter=None, defect_segmenter=None, discover=False
) -> dict[tuple, np.ndarray]:
    """Compute a mask per material and defect, and annotate each with mask fields.

    Defects use SAM 3 instances of their type that lie mostly inside the VLM's
    boxes; everything else gets box-prompted SAM 2 masks. With ``discover``,
    SAM 3 instances that match no known defect are added as new defects of the
    material they lie on.
    """
    shape = (image.height, image.width)
    masks: dict[tuple, np.ndarray] = {}
    scores: dict[tuple, float] = {}
    sources: dict[tuple, str] = {}

    detections: dict[str, list[dict]] = {}
    used: set[tuple[str, int]] = set()
    if defect_segmenter is not None:
        from .segmentation import DEFECT_CONCEPTS

        types = {d["type"] for m in materials for d in m["defects"]}
        if discover:
            types |= set(DEFECT_CONCEPTS)
        detections = {t: defect_segmenter.detect(image, t) for t in sorted(types)}
        for key, region in _regions(materials):
            if key[0] != "defect" or not region["boxes"]:
                continue
            area = _box_mask(shape, region["boxes"])
            dets = detections.get(region["type"], [])
            matched = [
                k for k, det in enumerate(dets)
                if (det["mask"] & area).sum() >= 0.5 * max(det["mask"].sum(), 1)
            ]
            if matched:
                masks[key] = np.logical_or.reduce([dets[k]["mask"] for k in matched])
                scores[key] = float(np.mean([dets[k]["score"] for k in matched]))
                sources[key] = "sam3"
                used.update((region["type"], k) for k in matched)

    if segmenter is not None:
        pending = [(key, r) for key, r in _regions(materials) if key not in masks and r["boxes"]]
        results = iter(segmenter.segment(image, [b for _, r in pending for b in r["boxes"]]))
        for key, region in pending:
            parts = [next(results) for _ in region["boxes"]]
            masks[key] = np.logical_or.reduce([m for m, _ in parts])
            scores[key] = float(np.mean([s for _, s in parts]))
            sources[key] = "sam2"

    if discover:
        for defect_type, dets in detections.items():
            for k, det in enumerate(dets):
                if (defect_type, k) in used:
                    continue
                cx = (det["box"][0] + det["box"][2]) / 2
                cy = (det["box"][1] + det["box"][3]) / 2
                owner = next(
                    (
                        i for i in range(len(materials))
                        if (("material", i) in masks and _mask_at(masks[("material", i)], cx, cy))
                        or any(_contains(b, cx, cy) for b in materials[i]["boxes"])
                    ),
                    None,
                )
                if owner is None:
                    continue
                defect = {"type": defect_type, "description": f"found by SAM 3 ('{det['concept']}')"}
                if merge_defect([materials[owner]], defect, [det["box"]], source="sam3"):
                    key = ("defect", owner, len(materials[owner]["defects"]) - 1)
                    masks[key], scores[key], sources[key] = det["mask"], det["score"], "sam3"

    if segmenter is not None or defect_segmenter is not None:
        for key, region in _regions(materials):
            region.update(
                mask=None,
                mask_source=sources.get(key),
                mask_score=round(scores[key], 3) if key in scores else None,
                area_fraction=round(float(masks[key].mean()), 4) if key in masks else 0.0,
            )
    return masks


def write_artifacts(
    materials: list[dict], image, masks, artifacts_dir: Path, name: str, rejected: list[dict] | None = None
) -> dict[str, str]:
    """Save mask PNGs and overlay images; returns the overlay paths. With
    ``rejected`` (verification ran), an overlay of the rejected defects is added."""
    mask_dir = artifacts_dir / "masks" / name
    for key, region in _regions(materials):
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
    if rejected is not None:
        overlays["rejected"] = overlay_dir / f"{name}_rejected.jpg"
        draw_overlay(image, rejected_items(rejected, masks)).save(overlays["rejected"], quality=90)
    return {kind: str(path) for kind, path in overlays.items()}


# --- entry points -------------------------------------------------------------------


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
    defect_segmenter=None,
    discover: bool = False,
    zoom_tiles: int = 0,
    zoom_overlap: float = 0.15,
    verify: bool = False,
    verify_padding: float = 0.5,
    verify_max_side: int | None = 1024,
    verify_max_per_image: int = 40,
    artifacts_dir: Path | None = None,
) -> list[dict]:
    """Run the VLM and return one result record per request.

    With ``joint=False`` every image is analyzed on its own; with ``joint=True``
    all images are sent in a single request and one record is returned. Boxes,
    masks, the zoom pass, verification and overlays are only produced per
    image, since boxes in a joint request cannot be attributed to a specific
    image. Verification runs after segmentation, so it checks the candidates of
    the VLM, the zoom pass and SAM 3 alike.
    """
    gen = {"max_new_tokens": max_new_tokens, "temperature": temperature, "thinking": thinking}
    system_prompt = build_system_prompt(box_coords)
    groups = [image_paths] if joint else [[p] for p in image_paths]
    results = []
    for index, group in enumerate(groups):
        images = [load_image(p, max_image_side) for p in group]
        start = time.perf_counter()
        raw, input_sizes = vlm.generate(images, prompt, system_prompt=system_prompt, **gen)
        record = {
            "images": [str(p) for p in group],
            "vlm_seconds": None,
            "analysis": None,
            "overlays": {},
            "parse_error": None,
            "raw_output": raw,
        }
        try:
            analysis = parse_analysis(raw).model_dump()
        except ValueError as exc:  # includes pydantic.ValidationError
            record["parse_error"] = str(exc)
            record["vlm_seconds"] = round(time.perf_counter() - start, 2)
            results.append(record)
            continue

        if joint:
            for m in analysis["materials"]:
                m["boxes"] = []
                for d in m["defects"]:
                    d["boxes"] = []
        else:
            original = load_image(group[0])
            scale_analysis(analysis, box_coords, input_sizes[0], original.size)
            if zoom_tiles > 1 and analysis["materials"]:
                record["zoom_raw_outputs"] = zoom_pass(
                    vlm, original, analysis, prompt, zoom_tiles, zoom_overlap,
                    max_image_side, box_coords, gen,
                )
        record["vlm_seconds"] = round(time.perf_counter() - start, 2)
        if not joint:
            masks = segment_regions(
                analysis["materials"], original, segmenter, defect_segmenter, discover
            )
            rejected = None
            if verify:
                start = time.perf_counter()
                verify_gen = gen if thinking else {**gen, "max_new_tokens": min(max_new_tokens, 512)}
                record["verification_raw_outputs"] = verify_defects(
                    vlm, original, analysis["materials"], masks, verify_padding,
                    verify_max_side, verify_max_per_image, verify_gen,
                )
                masks, rejected = apply_verification(analysis["materials"], masks)
                record["rejected_defects"] = rejected
                record["verify_seconds"] = round(time.perf_counter() - start, 2)
            if artifacts_dir is not None:
                record["overlays"] = write_artifacts(
                    analysis["materials"], original, masks, artifacts_dir,
                    f"{index:04d}_{group[0].stem}", rejected,
                )
        record["analysis"] = analysis
        results.append(record)
    return results


def overview(results: list[dict]) -> list[dict]:
    """Aggregate materials across images (e.g. frames of one video) by name."""
    rank = {"unknown": 0, "good": 1, "fair": 2, "poor": 3}
    by_name: dict[str, dict] = {}
    for record in results:
        if record["analysis"] is None:
            continue
        for m in record["analysis"]["materials"]:
            name = m["name"].strip().lower()
            entry = by_name.setdefault(
                name,
                {"name": name, "images": 0, "conditions": dict.fromkeys(CONDITIONS, 0), "defects": {}},
            )
            entry["images"] += 1
            entry["conditions"][m["condition"]] += 1
            for d in m["defects"]:
                entry["defects"][d["type"]] = entry["defects"].get(d["type"], 0) + 1
    for entry in by_name.values():
        seen = [c for c, n in entry["conditions"].items() if n]
        entry["worst_condition"] = max(seen, key=rank.__getitem__)
    return sorted(by_name.values(), key=lambda e: -e["images"])
