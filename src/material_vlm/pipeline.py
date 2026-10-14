"""Image loading and the prompt + images -> material analysis pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from .schema import SYSTEM_PROMPT, parse_analysis

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


def analyze(
    vlm,
    image_paths: list[Path],
    prompt: str,
    joint: bool = False,
    max_image_side: int | None = 1280,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> list[dict]:
    """Run the VLM and return one result record per request.

    With ``joint=False`` every image is analyzed on its own; with ``joint=True``
    all images are sent in a single request and one record is returned.
    """
    groups = [image_paths] if joint else [[p] for p in image_paths]
    results = []
    for group in groups:
        images = [load_image(p, max_image_side) for p in group]
        raw = vlm.generate(
            images,
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        record = {
            "images": [str(p) for p in group],
            "analysis": None,
            "parse_error": None,
            "raw_output": raw,
        }
        try:
            record["analysis"] = parse_analysis(raw).model_dump()
        except ValueError as exc:  # includes pydantic.ValidationError
            record["parse_error"] = str(exc)
        results.append(record)
    return results
