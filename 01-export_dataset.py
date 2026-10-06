import json
import random
import shutil
import math
from collections import Counter
from datetime import datetime

from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote


SCRIPT_DIR = Path(__file__).resolve().parent

IMAGES_DIR = Path("/Users/tawan/my_project/bean/frame/images")
OUTPUT_DIR = Path("/Users/tawan/my_project/bean/dataset")

TRAIN_SPLIT = 0.8
SEED = 42


CLASS_NAMES = [
    "ถั่วลิสง",
    "ถั่วปากอ้า",
    "ถั่วลันเตาเขียว",
    "ถั่วลันเตาเขียวชุบแป้ง",
    "แป้งสาลี",
]


def find_json_file():
    json_file = SCRIPT_DIR / "project-4.json"

    if not json_file.exists():
        raise FileNotFoundError(
            "ไม่พบ project-4.json ในโฟลเดอร์ bean"
        )

    return json_file


def get_image_filename(image_value):
    parsed = urlparse(image_value)
    query = parse_qs(parsed.query)

    if "d" in query:
        image_path = unquote(query["d"][0])
    else:
        image_path = unquote(parsed.path)

    return Path(image_path).name


def convert_annotation(result):
    value = result.get("value", {})

    labels = value.get("rectanglelabels", [])

    if not labels:
        return None

    label_name = labels[0]

    if label_name not in CLASS_NAMES:
        raise ValueError(f"Class ที่ไม่รู้จัก: {label_name}")

    class_id = CLASS_NAMES.index(label_name)

    x = float(value["x"])
    y = float(value["y"])
    width = float(value["width"])
    height = float(value["height"])

    if value.get("rotation", 0) != 0:
        raise ValueError("พบกรอบหมุน: ต้องตรวจ annotation ก่อนแปลงเป็น YOLO")
    if not all(math.isfinite(v) for v in (x, y, width, height)) or not (
        -0.000001 <= x <= 100 and -0.000001 <= y <= 100
        and width > 0 and height > 0
        and x + width <= 100.000001 and y + height <= 100.000001
    ):
        raise ValueError(f"กรอบอยู่นอกภาพหรือขนาดไม่ถูกต้อง: {value}")

    # Normalize floating-point noise at the image edges, preserving the source JSON.
    x = max(0.0, x)
    y = max(0.0, y)
    width = min(width, 100.0 - x)
    height = min(height, 100.0 - y)

    x_center = (x + width / 2) / 100
    y_center = (y + height / 2) / 100
    yolo_width = width / 100
    yolo_height = height / 100

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{yolo_width:.6f} "
        f"{yolo_height:.6f}"
    )


def load_tasks(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    usable_tasks_by_filename = {}
    duplicate_count = 0

    for task in data:
        image_value = task.get("data", {}).get("image")

        if not image_value:
            continue

        filename = get_image_filename(image_value)

        image_path = IMAGES_DIR / filename

        if not image_path.exists():
            raise FileNotFoundError(f"ไม่พบรูป: {image_path}")

        annotations = task.get("annotations", [])

        valid_annotations = []

        for annotation in annotations:
            if annotation.get("was_cancelled"):
                continue

            if annotation.get("result"):
                valid_annotations.append(annotation)

        if not valid_annotations:
            continue

        annotation = valid_annotations[-1]

        yolo_lines = []

        for result in annotation.get("result", []):
            if result.get("type") != "rectanglelabels":
                continue

            converted = convert_annotation(result)

            if converted:
                yolo_lines.append(converted)

        if not yolo_lines:
            continue

        # Keep the first usable task for each filename before splitting.
        if filename in usable_tasks_by_filename:
            duplicate_count += 1
            continue

        usable_tasks_by_filename[filename] = {
            "image": image_path,
            "labels": yolo_lines,
        }

    print(f"\nJSON ทั้งหมด: {len(data)} รายการ")
    print(f"ภาพซ้ำ: {duplicate_count} รายการ (ที่มี Label ใช้งานได้)")
    print(
        f"ภาพไม่ซ้ำที่นำไปใช้จริง: {len(usable_tasks_by_filename)} ภาพ"
    )

    return list(usable_tasks_by_filename.values())


def report_boxes(tasks, title):
    counts = Counter(
        int(line.split()[0]) for task in tasks for line in task["labels"]
    )
    print(f"\n{title} — bounding boxes:")
    for class_id, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {counts[class_id]}")
    return counts


def create_directories():
    if OUTPUT_DIR.exists():
        backup = SCRIPT_DIR / "dataset_backups" / datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        backup.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.rename(backup)
        print(f"เก็บ dataset เดิมไว้ที่: {backup}")

    directories = [
        OUTPUT_DIR / "images" / "train",
        OUTPUT_DIR / "images" / "val",
        OUTPUT_DIR / "labels" / "train",
        OUTPUT_DIR / "labels" / "val",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_task(task, split):
    image_path = task["image"]

    output_image = (
        OUTPUT_DIR
        / "images"
        / split
        / image_path.name
    )

    output_label = (
        OUTPUT_DIR
        / "labels"
        / split
        / f"{image_path.stem}.txt"
    )

    shutil.copy2(
        image_path,
        output_image
    )

    with open(
        output_label,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "\n".join(task["labels"])
        )


def create_data_yaml():
    yaml_path = OUTPUT_DIR / "data.yaml"

    lines = [
        f"path: {OUTPUT_DIR}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]

    for class_id, class_name in enumerate(CLASS_NAMES):
        lines.append(
            f'  {class_id}: "{class_name}"'
        )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(lines))


def create_classes_txt():
    classes_path = OUTPUT_DIR / "classes.txt"

    with open(
        classes_path,
        "w",
        encoding="utf-8"
    ) as file:
        for class_name in CLASS_NAMES:
            file.write(class_name + "\n")


def main():
    print("=" * 50)
    print("Label Studio JSON -> YOLO Dataset")
    print("=" * 50)

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์รูป: {IMAGES_DIR}")

    json_file = find_json_file()

    print(f"\nJSON:   {json_file.name}")
    print(f"Images: {IMAGES_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    print("\nClasses:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_id}: {class_name}"
        )

    tasks = load_tasks(json_file)

    if len(tasks) < 2:
        raise ValueError("ต้องมีภาพที่มี Label อย่างน้อย 2 ภาพเพื่อแบ่ง Train/Val")

    counts = report_boxes(tasks, "Unique images ทั้งหมด")
    print("ข้อมูลน้อยที่สุด: " + ", ".join(
        name for i, name in enumerate(CLASS_NAMES)
        if counts[i] == min(counts[j] for j in range(len(CLASS_NAMES)))
    ))
    print("ข้อมูลมากที่สุด: " + ", ".join(
        name for i, name in enumerate(CLASS_NAMES)
        if counts[i] == max(counts[j] for j in range(len(CLASS_NAMES)))
    ))

    filenames = {task["image"].name for task in tasks}
    stems = {task["image"].stem for task in tasks}
    if len(filenames) != len(tasks) or len(stems) != len(tasks):
        raise ValueError("ชื่อภาพหรือชื่อไฟล์ Label ซ้ำกัน")
    old_names = {
        path.name for path in (OUTPUT_DIR / "images").glob("*/*")
        if path.is_file()
    }
    missing = old_names - filenames
    if missing:
        raise ValueError(f"JSON ไม่ครอบคลุมภาพใน dataset เดิม: {sorted(missing)}")

    print(
        f"\nภาพที่มี Label: {len(tasks)} ภาพ"
    )

    random.seed(SEED)
    random.shuffle(tasks)

    split_index = int(
        len(tasks) * TRAIN_SPLIT
    )

    if len(tasks) > 1:
        split_index = max(
            1,
            min(
                split_index,
                len(tasks) - 1
            )
        )

    train_tasks = tasks[:split_index]
    val_tasks = tasks[split_index:]

    if {t["image"].name for t in train_tasks} & {t["image"].name for t in val_tasks}:
        raise ValueError("พบภาพเดียวกันทั้ง Train และ Val")
    for split, subset in (("Train", train_tasks), ("Val", val_tasks)):
        split_counts = report_boxes(subset, split)
        missing_classes = [
            name for i, name in enumerate(CLASS_NAMES) if split_counts[i] == 0
        ]
        if missing_classes:
            raise ValueError(f"{split} ไม่มี class: {', '.join(missing_classes)}")

    create_directories()

    for task in train_tasks:
        save_task(
            task,
            "train"
        )

    for task in val_tasks:
        save_task(
            task,
            "val"
        )

    create_data_yaml()
    create_classes_txt()

    print("\n" + "=" * 50)
    print("Dataset Conversion Complete")
    print("=" * 50)

    print(
        f"Train: {len(train_tasks)} images"
    )

    print(
        f"Val:   {len(val_tasks)} images"
    )

    print(
        f"Total: {len(tasks)} images"
    )

    print(
        f"\nDataset: {OUTPUT_DIR}"
    )

    print(
        f"data.yaml: {OUTPUT_DIR / 'data.yaml'}"
    )

    print(
        f"classes.txt: {OUTPUT_DIR / 'classes.txt'}"
    )

    print(
        "\nพร้อมสำหรับ Train YOLO แล้ว"
    )


if __name__ == "__main__":
    main()
