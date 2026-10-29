from pathlib import Path
import argparse


def build_dataset_yaml(root: Path, train_dir: Path, val_dir: Path, test_dir: Path | None) -> Path:
    """Tạo file dataset.yaml nếu chưa có."""
    dataset_dir = root / "dataset"
    yaml_path = dataset_dir / "data.yaml"

    train_path = train_dir.as_posix()
    val_path = val_dir.as_posix()
    test_part = f"\ntest: {test_dir.as_posix()}" if test_dir is not None else ""

    content = (
        f"train: {train_path}\n"
        f"val: {val_path}{test_part}\n\n"
        "nc: 1\n"
        "names: ['license_plate']\n"
    )

    yaml_path.write_text(content, encoding="utf-8")
    print(f"Đã tạo file dataset YAML: {yaml_path}")
    return yaml_path


def detect_split_dirs(root: Path):
    """Tìm thư mục train/val/test trong dataset/images."""
    images_root = root / "dataset" / "images"
    train_dir = images_root / "train"
    val_dir = images_root / "val"
    test_dir = images_root / "test"

    if train_dir.exists() and val_dir.exists():
        return train_dir, val_dir, test_dir if test_dir.exists() else None

    # fallback: nếu chưa chia tập, dùng thư mục gốc images/labels
    if images_root.exists():
        return images_root, images_root, None

    raise FileNotFoundError(
        "Không tìm thấy dataset/images/train và dataset/images/val. "
        "Hãy chạy script chia dữ liệu trước hoặc kiểm tra lại cấu trúc thư mục."
    )


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện YOLO cho bài toán nhận diện biển số.")
    parser.add_argument("--root", type=str, default=r"E:\code\training_ai_nhan_dien_bien_so", help="Đường dẫn thư mục dự án")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Tên mô hình YOLO hoặc đường dẫn đến .pt")
    parser.add_argument("--epochs", type=int, default=50, help="Số epoch huấn luyện")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset_dir = root / "dataset"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset ở: {dataset_dir}")

    train_dir, val_dir, test_dir = detect_split_dirs(root)

    data_yaml = build_dataset_yaml(root, train_dir, val_dir, test_dir)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Thiếu gói ultralytics. Hãy cài đặt bằng: pip install ultralytics")

    print(f"Bắt đầu huấn luyện với mô hình: {args.model}")
    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=2,
        project=str(root / "runs"),
        name="license_plate_train",
        exist_ok=True,
    )

    print("Huấn luyện hoàn tất.")
    print(f"Kết quả lưu ở: {root / 'runs' / 'detect' / 'license_plate_train'}")


if __name__ == "__main__":
    main()