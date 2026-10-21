"""Compare runs of different models on the same images.

``material-vlm-compare outputs/site/*.json --out outputs/site/comparison`` writes
one side-by-side image per input image (a column per run, materials on top,
defects below, and defects rejected by verification if any run used it) and a
summary table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE_WIDTH = 640


def load_runs(paths: list[str | Path]) -> list[dict]:
    runs = []
    for path in map(Path, paths):
        run = json.loads(path.read_text())
        run["name"] = path.stem
        runs.append(run)
    return runs


def summarize(run: dict) -> dict:
    results = run["results"]
    parsed = [r for r in results if r["analysis"] is not None]
    materials = [m for r in parsed for m in r["analysis"]["materials"]]
    defects = [d for m in materials for d in m["defects"]]
    scores = [x["mask_score"] for x in materials + defects if x.get("mask_score") is not None]
    seconds = [
        r["vlm_seconds"] + r.get("verify_seconds", 0) for r in results if r.get("vlm_seconds") is not None
    ]
    defect_counts = {}
    for d in defects:
        defect_counts[d["type"]] = defect_counts.get(d["type"], 0) + 1
    verified = "verify_seconds" in (parsed[0] if parsed else {})
    rejected = {}
    for r in parsed:
        for d in r.get("rejected_defects", []):
            label = f"{d['type']}->{d['verification']['label']}"
            rejected[label] = rejected.get(label, 0) + 1
    return {
        "run": run["name"],
        "model": run["model_id"],
        "images": len(results),
        "parsed": len(parsed),
        "materials_per_image": round(len(materials) / max(len(parsed), 1), 2),
        "defects": defect_counts,
        "rejected": rejected if verified else None,
        "mean_mask_score": round(sum(scores) / len(scores), 3) if scores else None,
        "seconds_per_image": round(sum(seconds) / len(seconds), 1) if seconds else None,
    }


def _tile(path: str | None, header: str, font, missing: str = "no output (parse error?)") -> Image.Image:
    if path and Path(path).exists():
        image = Image.open(path).convert("RGB")
        image = image.resize((TILE_WIDTH, round(image.height * TILE_WIDTH / image.width)))
    else:
        image = Image.new("RGB", (TILE_WIDTH, TILE_WIDTH * 3 // 4), (40, 40, 40))
        ImageDraw.Draw(image).text((20, 20), missing, fill="white", font=font)
    canvas = Image.new("RGB", (TILE_WIDTH, image.height + 36), "white")
    canvas.paste(image, (0, 36))
    ImageDraw.Draw(canvas).text((8, 8), header, fill="black", font=font)
    return canvas


def side_by_side(runs: list[dict], index: int, font) -> Image.Image:
    kinds = ["materials", "defects"]
    if any("rejected" in run["results"][index]["overlays"] for run in runs):
        kinds.append("rejected")
    rows = []
    for kind in kinds:
        missing = "not verified" if kind == "rejected" else "no output (parse error?)"
        tiles = [
            _tile(run["results"][index]["overlays"].get(kind), f"{run['name']} - {kind}", font, missing)
            for run in runs
        ]
        row = Image.new("RGB", (TILE_WIDTH * len(tiles), max(t.height for t in tiles)), "white")
        for i, tile in enumerate(tiles):
            row.paste(tile, (i * TILE_WIDTH, 0))
        rows.append(row)
    grid = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height
    return grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="material-vlm-compare", description="Compare runs on the same images."
    )
    parser.add_argument("runs", nargs="+", help="output JSON files of material-vlm")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args(argv)

    runs = load_runs(args.runs)
    images = [r["images"] for r in runs[0]["results"]]
    for run in runs[1:]:
        if [r["images"] for r in run["results"]] != images:
            raise ValueError(f"{run['name']} was run on different images than {runs[0]['name']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for index, group in enumerate(images):
        side_by_side(runs, index, font).save(
            out / f"{index:04d}_{Path(group[0]).stem}.jpg", quality=90
        )

    summary = [summarize(run) for run in runs]
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    header = f"{'run':<24} {'parsed':>8} {'mat/img':>8} {'mask':>6} {'s/img':>7}  defects"
    print(header)
    for s in summary:
        print(
            f"{s['run']:<24} {s['parsed']:>3}/{s['images']:<4} {s['materials_per_image']:>8} "
            f"{s['mean_mask_score'] or '-':>6} {s['seconds_per_image'] or '-':>7}  {s['defects']}"
        )
        if s["rejected"] is not None:
            print(f"{'':<24} rejected by verification: {s['rejected']}")
    print(f"Wrote {len(images)} comparison image(s) and summary.json to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
