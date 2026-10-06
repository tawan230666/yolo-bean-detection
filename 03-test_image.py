from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATH = SCRIPT_DIR / "models" / "best.pt"

IMAGE_PATH = SCRIPT_DIR / "frame" / "images" / "0001.jpg"

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.20,
    save=True
)

print("\nตรวจจับเสร็จแล้ว")
