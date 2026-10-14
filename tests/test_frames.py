import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from material_vlm.frames import extract_frames  # noqa: E402


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    rng = np.random.default_rng(0)
    for i in range(35):  # 3.5 s at 10 fps
        # Frame 5 of each second is noise (sharp), the rest are flat (blurry).
        if i % 10 == 5:
            frame = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        else:
            frame = np.full((48, 64, 3), 128, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_one_frame_per_interval_picks_sharpest(video, tmp_path):
    saved = extract_frames(video, tmp_path / "frames", every=1.0)
    assert [p.name for p in saved] == [
        "clip_00000.50s.jpg",
        "clip_00001.50s.jpg",
        "clip_00002.50s.jpg",
        "clip_00003.00s.jpg",
    ]


def test_first_frame_and_max_frames(video, tmp_path):
    saved = extract_frames(video, tmp_path / "frames", every=1.0, pick_sharpest=False, max_frames=2)
    assert [p.name for p in saved] == ["clip_00000.00s.jpg", "clip_00001.00s.jpg"]
