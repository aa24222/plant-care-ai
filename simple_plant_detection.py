# simple_plant_detection.py
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

print("Loading your trained disease classifier...\n")

# Load your trained disease classifier
disease_classifier = YOLO('runs/classify/plant_disease_v1/weights/best.pt')

print("Model loaded!\n")
def smart_crop(image_path):
    """
    Smart cropping strategy:
    1. Look for green regions (leaves)
    2. Crop to largest green region
    3. Fallback to center crop if no green found
    """
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        return None, None
    
    height, width = img.shape[:2]
    
    print("Stage 1: Looking for leaves (green regions)...")
    
    # Convert to HSV color space (better for color detection)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define green color range (covers light green to dark green)
    # H: 25-85 (green hues)
    # S: 40-255 (saturation - not too gray)
    # V: 40-255 (brightness - not too dark)
    lower_green = (25, 40, 40)
    upper_green = (85, 255, 255)
    
    # Create mask for green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up mask (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours (leaf shapes)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest green region (likely the main leaf or cluster)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Only use if the green region is significant (at least 5% of image)
        min_area = (width * height) * 0.05
        
        if area > min_area:
            # Get bounding box of largest green region
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding around the detection (10% on each side)
            padding_x = int(w * 0.1)
            padding_y = int(h * 0.1)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(width - x, w + 2 * padding_x)
            h = min(height - y, h + 2 * padding_y)
            
            # Crop to detected region
            cropped = img[y:y+h, x:x+w]
            
            print(f"Found leaf! Cropping to green region (area: {area:.0f} pixels).\n")
            return cropped, 'green_detection'
        else:
            print(f"Green region too small (area: {area:.0f} pixels). Using center crop.\n")
    else:
        print("No green regions detected. Using center crop.\n")
    
    # Fallback: center crop
    size = min(height, width)
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    cropped = img[start_y:start_y+size, start_x:start_x+size]
    
    return cropped, 'center_crop'
    

def analyze_plant(image_path):
    """
    Analyze plant disease:
    1. Smart crop to focus on plant
    2. Classify disease with your trained model
    """
    
    print(f"Analyzing: {image_path}\n")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return None
    
    # Smart crop
    cropped_img, crop_method = smart_crop(image_path)
    
    if cropped_img is None:
        return None
    
    # Save cropped image temporarily
    temp_path = 'temp_cropped.jpg'
    cv2.imwrite(temp_path, cropped_img)
    
    # Stage 2: Classify disease
    print("Stage 2: Classifying disease...")
    results = disease_classifier(temp_path, verbose=False)
    probs = results[0].probs
    
    # Get top 3 predictions
    top3 = []
    for idx in probs.top5[:3]:
        top3.append({
            'disease': disease_classifier.names[idx],
            'confidence': float(probs.data[idx])
        })
    
    result = {
        'image': image_path,
        'crop_method': crop_method,
        'prediction': disease_classifier.names[probs.top1],
        'confidence': float(probs.top1conf),
        'top3': top3
    }
    
    # Visualize
    visualize_result(image_path, cropped_img, result)
    
    # Cleanup
    os.remove(temp_path)
    
    return result


def visualize_result(original_path, cropped_img, result):
    """Display original image, cropped region, and prediction"""
    
    # Read original
    original = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original image
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Cropped + prediction
    axes[1].imshow(cropped_rgb)
    disease = result['prediction'].replace('_', ' ')
    confidence = result['confidence'] * 100
    axes[1].set_title(f'Prediction: {disease}\nConfidence: {confidence:.1f}%', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save instead of show
    output_path = 'detection_result.jpg'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python simple_plant_detection.py <image_path>")
        print("\nExample:")
        print("  python simple_plant_detection.py datasets/plantvillage_yolo/val/Tomato_Early_blight/image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Analyze
    result = analyze_plant(image_path)
    
    if result:
        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\nImage: {result['image']}")
        print(f"Crop method: {result['crop_method']}")
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"\nTop 3 Predictions:")
        for i, pred in enumerate(result['top3'], 1):
            print(f"  {i}. {pred['disease']}: {pred['confidence']*100:.1f}%")
        print("\n" + "=" * 60)