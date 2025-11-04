"""
train_plant_disease_yolo.py

This script:
1. Loads and trains a YOLOv8 classification model on PlantVillage
2. Inspects the trained model
3. Displays training results (loss curves, confusion matrix)
4. Tests the model on a random validation image
"""

import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import torch
from ultralytics import YOLO


# ---------------------------------------------------------
# 1. Device Setup
# ---------------------------------------------------------
def get_device():
    """Return 'cuda' if GPU available, else 'cpu'."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU — training may be slow.")

    return device


# ---------------------------------------------------------
# 2. Train Model
# ---------------------------------------------------------
def train_model(device):
    """Train a YOLOv8 classification model."""
    model = YOLO("yolov8n-cls.pt")   # Nano classification model
    print("Model loaded: YOLOv8 Nano (Classification)")

    results = model.train(
        data="datasets/plantvillage_yolo",
        epochs=20,
        imgsz=224,
        batch=16,
        patience=5,
        device=device,
        project="runs/classify",
        name="plant_disease_v1",
        plots=True,
        verbose=True
    )

    print("Training complete.")
    return results


# ---------------------------------------------------------
# 3. Load Trained Model
# ---------------------------------------------------------
def load_trained_model():
    """Load trained YOLO model and print class info."""
    model_path = "runs/classify/plant_disease_v1/weights/best.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found.")

    print("Model found.")
    print(f"Location: {os.path.abspath(model_path)}")

    model = YOLO(model_path)

    print("\nModel Information:")
    print(f"Classes trained: {len(model.names)}")
    print("Class names:")
    for idx, name in model.names.items():
        print(f"{idx}: {name}")

    return model


# ---------------------------------------------------------
# 4. Show training result plots
# ---------------------------------------------------------
def show_training_results():
    """Show training results (loss curves + confusion matrix)."""
    print("\nDisplaying training results...\n")

    results_path = "runs/classify/plant_disease_v1/results.png"
    confusion_path = "runs/classify/plant_disease_v1/confusion_matrix.png"

    if os.path.exists(results_path):
        img = PILImage.open(results_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Training Performance")
        plt.show()
    else:
        print("results.png not found")

    if os.path.exists(confusion_path):
        img = PILImage.open(confusion_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Confusion Matrix")
        plt.show()
    else:
        print("confusion_matrix.png not found")


# ---------------------------------------------------------
# 5. Test Model on Random Validation Image
# ---------------------------------------------------------
def test_random_image(model):
    """Pick random validation image, predict, and display results."""
    val_path = Path("datasets/plantvillage_yolo/val")

    classes = [d for d in val_path.iterdir() if d.is_dir()]
    if not classes:
        print("No class folders found in validation directory.")
        return

    random_class = random.choice(classes)
    images = list(random_class.glob("*.jpg")) + list(random_class.glob("*.JPG"))

    if not images:
        print(f"No images found in class folder: {random_class.name}")
        return

    test_image = random.choice(images)
    print(f"\nTesting on image: {test_image.name}")
    print(f"True Label: {random_class.name}")

    # Show the image
    img = PILImage.open(test_image)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True Label: {random_class.name}")
    plt.show()

    # Predict
    results = model.predict(str(test_image), verbose=False)
    probs = results[0].probs

    top3 = probs.top5[:3]

    print("\nTop 3 Predictions:")
    for i, idx in enumerate(top3, 1):
        class_name = model.names[idx]
        confidence = probs.data[idx].item() * 100
        print(f"{i}. {class_name:<40} {confidence:6.2f}%")

    predicted = model.names[probs.top1]
    if predicted == random_class.name:
        print("Prediction: Correct")
    else:
        print(f"Prediction: Incorrect (Predicted: {predicted})")


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    device = get_device()

    # Train model (comment out if already trained)
    train_model(device)

    # Load best model
    model = load_trained_model()

    # Show results
    show_training_results()

    # Test on random validation image
    test_random_image(model)
