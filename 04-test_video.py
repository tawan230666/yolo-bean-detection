from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATH = SCRIPT_DIR / "models" / "best.pt"
VIDEO_PATH = SCRIPT_DIR / "video" / "road3.mp4"

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(VIDEO_PATH),
    conf=0.20,
    iou=0.50,
    agnostic_nms=True,
    save=True,
    stream=True
)

for result in results:
    pass

print("\nทดสอบวิดีโอเสร็จแล้ว")
