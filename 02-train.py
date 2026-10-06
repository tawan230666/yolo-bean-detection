from ultralytics import YOLO
from pathlib import Path
import shutil
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATH = SCRIPT_DIR / "yolo26n.pt"
DATA_PATH = SCRIPT_DIR / "dataset" / "data.yaml"

def main():
    model = YOLO(str(MODEL_PATH))

    results = model.train(
        data=str(DATA_PATH),

        epochs=100,
        imgsz=640,

        optimizer="auto",

        device="mps",

        degrees=7.0,
        shear=5.0,
        perspective=0.001,

        fliplr=0.5,
        flipud=0.5,

        mosaic=0.1,
        mixup=0.1,
        close_mosaic=10,

        patience=30,

        project=str(SCRIPT_DIR / "runs" / "detect"),
        name="bean_train"
    )

    best_model = Path(model.trainer.best).resolve()
    if not best_model.is_file():
        raise FileNotFoundError(f"ไม่พบ best.pt ของรอบที่เพิ่ง Train: {best_model}")

    latest_dir = SCRIPT_DIR / "models"
    latest_dir.mkdir(exist_ok=True)

    latest_model = latest_dir / "best.pt"
    if latest_model.exists():
        backup = latest_dir / ("best_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".pt")
        shutil.copy2(latest_model, backup)
    temporary_model = latest_dir / "best.pending.pt"
    shutil.copy2(best_model, temporary_model)
    temporary_model.replace(latest_model)

    print("\nTrain เสร็จแล้ว")
    print("โมเดลรอบนี้:")
    print(best_model)

    print("\nโมเดลล่าสุด:")
    print(latest_model)


if __name__ == "__main__":
    main()
