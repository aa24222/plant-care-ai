"""
prepare_plantvillage.py

Utility script to:
1. Load and inspect the PlantVillage dataset
2. Display sample images (optional)
3. Generate dataset statistics
4. Split the dataset into train/val folders for YOLO training
"""

from pathlib import Path
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Basic Setup
# ---------------------------------------------------------
DATASET_PATH = Path("datasets/plantvillage/train/PlantVillage")
OUTPUT_DIR = Path("datasets/plantvillage_yolo")
TRAIN_RATIO = 0.80


def load_classes(dataset_path: Path):
    """Return a sorted list of class folder names."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    return classes


# ---------------------------------------------------------
# 2. Display sample images (optional)
# ---------------------------------------------------------
def show_sample_images(dataset_path: Path, classes):
    """Show the first image from each of the first 12 classes."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flat

    for i, ax in enumerate(axes):
        if i < len(classes):
            class_folder = dataset_path / classes[i]
            image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.JPG"))

            if image_files:
                img = Image.open(image_files[0])
                ax.imshow(img)
                ax.set_title(classes[i].replace("___", ": ").replace("__", " - "),
                             fontsize=10, pad=10)
                ax.axis("off")
            else:
                ax.text(0.5, 0.5, "No images", ha="center", va="center")
                ax.axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 3. Dataset Statistics
# ---------------------------------------------------------
def count_images_per_class(dataset_path: Path, classes):
    """Print image count per class and return total number of images."""
    print("\n📊 Dataset Statistics")
    print(f"{'Class Name':<35} {'Image Count':>12}")
    print("-" * 50)

    total_images = 0
    for cls in classes:
        class_folder = dataset_path / cls
        image_count = len(list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.JPG")))
        total_images += image_count
        print(f"{cls:<35} {image_count:>12}")

    print("-" * 50)
    print(f"{'TOTAL':<35} {total_images:>12}")
    return total_images


# ---------------------------------------------------------
# 4. Split Dataset into train/val folders
# ---------------------------------------------------------
def split_dataset(dataset_path: Path, classes, output_dir: Path, train_ratio: float):
    """Split images into train/val directories according to ratio."""
    random.seed(42)

    # Create directory structure
    for split in ["train", "val"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    print("\nSplitting dataset...")
    print(f"{'Class Name':<40} {'Train':>8} {'Val':>8}")
    print("-" * 60)

    for class_name in classes:
        class_folder = dataset_path / class_name
        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.JPG"))

        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:]

        # Create class folders
        for split in ["train", "val"]:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

        # Copy images
        for img in train_imgs:
            shutil.copy(img, output_dir / "train" / class_name / img.name)

        for img in val_imgs:
            shutil.copy(img, output_dir / "val" / class_name / img.name)

        print(f"{class_name:<40} {len(train_imgs):>8} {len(val_imgs):>8}")

    # Summary
    total_train = sum(len(list((output_dir / "train" / cls).glob("*"))) for cls in classes)
    total_val = sum(len(list((output_dir / "val" / cls).glob("*"))) for cls in classes)

    print("-" * 60)
    print(f"{'TOTAL':<40} {total_train:>8} {total_val:>8}")
    print("\n✅ Dataset split complete!")


# ---------------------------------------------------------
# 5. Verify the dataset structure
# ---------------------------------------------------------
def verify_split(output_dir: Path):
    """Verify number of classes and images in each split."""
    print("\nVerifying dataset structure...\n")

    for split in ["train", "val"]:
        split_path = output_dir / split
        class_folders = [d for d in split_path.iterdir() if d.is_dir()]
        total_images = sum(len(list(cf.glob("*"))) for cf in class_folders)

        print(f"{split.upper()}:")
        print(f"  Classes: {len(class_folders)}")
        print(f"  Total images: {total_images}")
        print()

    print("✅ Ready to train!")


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    print("✅ Preparing PlantVillage dataset...\n")

    classes = load_classes(DATASET_PATH)

    print(f"Found {len(classes)} classes.")
    print("First 10 classes:", classes[:10])

    count_images_per_class(DATASET_PATH, classes)

    # Optional: uncomment to show images
    # show_sample_images(DATASET_PATH, classes)

    split_dataset(DATASET_PATH, classes, OUTPUT_DIR, TRAIN_RATIO)

    verify_split(OUTPUT_DIR)
