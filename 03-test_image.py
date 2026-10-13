"""Detect beans in a single image.

Usage:
    python 03-test_image.py [path/to/image.jpg]

Annotated output is saved under runs/detect/.
"""
import sys
from pathlib import Path

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "best.pt"
DEFAULT_IMAGE = SCRIPT_DIR / "frame" / "images" / "0001.jpg"


def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not image_path.is_file():
        raise FileNotFoundError(f"ไม่พบรูป: {image_path}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=str(image_path), conf=0.20, save=True)

    for box in results[0].boxes:
        name = model.names[int(box.cls)]
        print(f"  {name}: {float(box.conf):.2f}")
    print(f"\nตรวจจับเสร็จแล้ว — พบ {len(results[0].boxes)} ชิ้น")
    print(f"ผลลัพธ์: {results[0].save_dir}")


if __name__ == "__main__":
    main()
