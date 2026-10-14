"""Command line entry point: ``material-vlm --images ... --output ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .pipeline import analyze, collect_images

DEFAULTS = {
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "prompt_file": None,
    "joint": False,
    "max_image_side": 1280,
    "max_new_tokens": 1024,
    "temperature": 0.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="material-vlm",
        description="Detect materials and assess their condition in images using a VLM.",
    )
    parser.add_argument(
        "--images", nargs="+", required=True, help="image files and/or directories"
    )
    parser.add_argument("--output", default=None, help="JSON output file (default: stdout)")
    parser.add_argument("--config", default=None, help="YAML config file")
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
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    """Merge defaults < config file < command line arguments."""
    config = dict(DEFAULTS)
    if args.config:
        loaded = yaml.safe_load(Path(args.config).read_text()) or {}
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
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
    print(f"Analyzing {len(image_paths)} image(s) with {config['model_id']}", file=sys.stderr)

    from .model import VLM

    vlm = VLM(config["model_id"])
    results = analyze(
        vlm,
        image_paths,
        prompt,
        joint=config["joint"],
        max_image_side=config["max_image_side"],
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
    )

    payload = json.dumps(
        {"model_id": config["model_id"], "prompt": prompt, "results": results},
        indent=2,
        ensure_ascii=False,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n")
        print(f"Wrote {output}", file=sys.stderr)
    else:
        print(payload)

    failed = sum(r["analysis"] is None for r in results)
    if failed:
        print(f"{failed}/{len(results)} output(s) could not be parsed as JSON", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
