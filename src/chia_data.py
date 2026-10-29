import os
import random
from pathlib import Path
import shutil

random.seed(42)

# ---------------------------
# Cấu hình
# ---------------------------
root = Path(r"E:\code\training_ai_nhan_dien_bien_so")
images_dir = root / "dataset" / "images"
labels_dir = root / "dataset" / "labels"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# ---------------------------
# Chọn các file ảnh
# ---------------------------
image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

all_images = []
for img in images_dir.iterdir():
    if img.suffix.lower() in image_extensions:
        all_images.append(img)

# Sắp xếp để ổn định
all_images = sorted(all_images)
random.shuffle(all_images)

total = len(all_images)
train_count = int(total * train_ratio)
val_count = int(total * val_ratio)

train_files = all_images[:train_count]
val_files = all_images[train_count:train_count + val_count]
test_files = all_images[train_count + val_count:]

# ---------------------------
# Tạo thư mục mới
# ---------------------------
for split in ["train", "val", "test"]:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

# ---------------------------
# Hàm copy label tương ứng
# ---------------------------
def copy_matching_label(image_path, split):
    label_path = labels_dir / (image_path.stem + ".txt")
    if label_path.exists():
        shutil.copy2(image_path, images_dir / split / image_path.name)
        shutil.copy2(label_path, labels_dir / split / label_path.name)
    else:
        print(f"Không tìm thấy label cho ảnh: {image_path.name}")

# ---------------------------
# Copy dữ liệu
# ---------------------------
for f in train_files:
    copy_matching_label(f, "train")

for f in val_files:
    copy_matching_label(f, "val")

for f in test_files:
    copy_matching_label(f, "test")

print(f"Tổng ảnh: {total}")
print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")
print(f"Test: {len(test_files)}")