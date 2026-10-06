from pathlib import Path

import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "best.pt"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"ไม่พบโมเดล: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(0)

    try:
        if not cap.isOpened():
            raise RuntimeError("เปิดกล้องไม่ได้")

        print("เปิดกล้องแล้ว")
        print("กด q เพื่อออก")

        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("อ่านภาพจากกล้องไม่ได้")

            results = model.predict(
                source=frame,
                conf=0.20,
                iou=0.50,
                agnostic_nms=True,
                verbose=False,
            )

            annotated_frame = results[0].plot()
            cv2.imshow("YOLO Bean Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("ปิดกล้องแล้ว")


if __name__ == "__main__":
    main()
