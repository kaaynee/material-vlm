"""Extract still frames from a video: ``material-vlm-frames video.mp4 --out data/frames``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def sharpness(frame) -> float:
    """Variance of the Laplacian: higher means sharper, low values indicate blur."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    every: float = 1.0,
    max_frames: int | None = None,
    pick_sharpest: bool = True,
) -> list[Path]:
    """Save one frame per ``every`` seconds of video and return the written paths.

    With ``pick_sharpest`` the least blurry frame of each interval is kept
    instead of the first one, which matters for handheld footage.
    """
    import cv2

    video, out_dir = Path(video), Path(out_dir)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(every * fps))
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    best = None  # (score, frame index, frame) of the current interval

    def save(entry):
        _, index, frame = entry
        path = out_dir / f"{video.stem}_{index / fps:08.2f}s.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(path)

    index = 0
    while max_frames is None or len(saved) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0 and best is not None:
            save(best)
            best = None
        score = sharpness(frame) if pick_sharpest else 0.0
        if best is None or score > best[0]:
            best = (score, index, frame)
        index += 1
    if best is not None and (max_frames is None or len(saved) < max_frames):
        save(best)
    capture.release()
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="material-vlm-frames", description="Extract still frames from a video."
    )
    parser.add_argument("video")
    parser.add_argument("--out", required=True, help="output directory for the frames")
    parser.add_argument(
        "--every", type=float, default=1.0, help="seconds between frames (default: 1)"
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--no-sharpest",
        dest="pick_sharpest",
        action="store_false",
        help="keep the first frame of each interval instead of the sharpest",
    )
    args = parser.parse_args(argv)
    saved = extract_frames(
        args.video, args.out, args.every, args.max_frames, args.pick_sharpest
    )
    print(f"Wrote {len(saved)} frame(s) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
