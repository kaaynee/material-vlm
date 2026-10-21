"""Command line entry point: ``material-vlm --images ... --output ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .pipeline import analyze, collect_images, overview

DEFAULTS = {
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "prompt_file": None,
    "joint": False,
    "max_image_side": 1280,
    "max_new_tokens": 2048,
    "temperature": 0.0,
    "thinking": False,
    "box_coords": "input_pixels",
    "segment": True,
    "segmenter_id": "facebook/sam2.1-hiera-large",
    "zoom_tiles": 0,
    "zoom_overlap": 0.15,
    "defect_segmenter": "none",
    "sam3_id": "facebook/sam3",
    "sam3_threshold": 0.4,
    "sam3_discover": False,
    "verify": False,
    "verify_padding": 0.5,
    "verify_max_side": 1024,
    "verify_max_per_image": 40,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="material-vlm",
        description="Detect materials and assess their condition in images using a VLM.",
    )
    parser.add_argument(
        "--images", nargs="+", required=True, help="image files and/or directories"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output file (default: stdout). Masks and overlays are written to a "
        "directory of the same name without the extension.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML config file; repeat to layer files, later ones override earlier ones",
    )
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", default=None, help="task prompt text")
    prompt.add_argument("--prompt-file", default=None, help="file containing the task prompt")
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--joint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="analyze all images in one request instead of one request per image",
    )
    parser.add_argument("--max-image-side", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="let Qwen3/3.5 reason before answering (slower, needs more max-new-tokens)",
    )
    parser.add_argument(
        "--box-coords",
        choices=["input_pixels", "normalized_1000"],
        default=None,
        help="box format the VLM uses: input_pixels for Qwen2.5-VL, normalized_1000 "
        "for Qwen3-VL",
    )
    parser.add_argument(
        "--segment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="turn detected boxes into masks with SAM 2",
    )
    parser.add_argument("--segmenter-id", default=None)
    parser.add_argument(
        "--zoom-tiles",
        type=int,
        default=None,
        help="also search N x N high-resolution crops for defects (0 = off)",
    )
    parser.add_argument(
        "--defect-segmenter",
        choices=["none", "sam3"],
        default=None,
        help="segment defects with text-prompted SAM 3 instead of boxes",
    )
    parser.add_argument(
        "--sam3-discover",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="add SAM 3 defect detections the VLM missed",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="check every defect candidate on a close-up crop and drop look-alikes",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    """Merge defaults < config files (in order) < command line arguments."""
    config = dict(DEFAULTS)
    for path in args.config:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        config.update(loaded)
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return config


def resolve_prompt(args: argparse.Namespace, config: dict) -> str:
    if args.prompt:
        return args.prompt
    if config["prompt_file"]:
        return Path(config["prompt_file"]).read_text().strip()
    raise ValueError("no prompt given: use --prompt, --prompt-file or a config with prompt_file")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    prompt = resolve_prompt(args, config)
    image_paths = collect_images(args.images)
    artifacts_dir = Path(args.output).with_suffix("") if args.output else None
    segment = config["segment"] and not config["joint"]
    if segment and artifacts_dir is None:
        raise ValueError("segmentation writes mask files: pass --output or use --no-segment")
    print(f"Analyzing {len(image_paths)} image(s) with {config['model_id']}", file=sys.stderr)

    from .model import VLM

    vlm = VLM(config["model_id"])
    segmenter = defect_segmenter = None
    if segment:
        from .segmentation import Segmenter

        segmenter = Segmenter(config["segmenter_id"])
    if config["defect_segmenter"] == "sam3" and not config["joint"]:
        from .segmentation import DefectSegmenter

        defect_segmenter = DefectSegmenter(config["sam3_id"], config["sam3_threshold"])
    results = analyze(
        vlm,
        image_paths,
        prompt,
        joint=config["joint"],
        max_image_side=config["max_image_side"],
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
        thinking=config["thinking"],
        box_coords=config["box_coords"],
        segmenter=segmenter,
        defect_segmenter=defect_segmenter,
        discover=config["sam3_discover"],
        zoom_tiles=config["zoom_tiles"],
        zoom_overlap=config["zoom_overlap"],
        verify=config["verify"],
        verify_padding=config["verify_padding"],
        verify_max_side=config["verify_max_side"],
        verify_max_per_image=config["verify_max_per_image"],
        artifacts_dir=artifacts_dir,
    )

    payload = json.dumps(
        {
            "model_id": config["model_id"],
            "prompt": prompt,
            "config": config,
            "overview": overview(results),
            "results": results,
        },
        indent=2,
        ensure_ascii=False,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n")
        print(f"Wrote {output}", file=sys.stderr)
        if artifacts_dir.exists():
            print(f"Wrote masks and overlays to {artifacts_dir}", file=sys.stderr)
    else:
        print(payload)

    if config["verify"] and not config["joint"]:
        kept = sum(len(m["defects"]) for r in results if r["analysis"] for m in r["analysis"]["materials"])
        rejected = sum(len(r.get("rejected_defects", [])) for r in results)
        print(f"Verification kept {kept} and rejected {rejected} defect(s)", file=sys.stderr)
    failed = sum(r["analysis"] is None for r in results)
    if failed:
        print(f"{failed}/{len(results)} output(s) could not be parsed as JSON", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
