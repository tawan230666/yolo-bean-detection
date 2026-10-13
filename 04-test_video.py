"""Run detection on a video file and save an annotated copy.

Usage:
    python 04-test_video.py [path/to/video.mp4]
"""
import sys
from pathlib import Path

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "best.pt"
DEFAULT_VIDEO = SCRIPT_DIR / "video" / "road3.mp4"


def main():
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not video_path.is_file():
        raise FileNotFoundError(f"ไม่พบวิดีโอ: {video_path}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(video_path),
        conf=0.20,
        iou=0.50,
        agnostic_nms=True,
        save=True,
        stream=True,  # process frame by frame to keep memory usage flat
    )

    frames = 0
    for _ in results:
        frames += 1

    print(f"\nทดสอบวิดีโอเสร็จแล้ว — {frames} เฟรม")


if __name__ == "__main__":
    main()
