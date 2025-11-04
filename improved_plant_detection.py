# improved_plant_detection.py
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from dotenv import load_dotenv
import os
import sys
from collections import Counter

# Load environment variables
load_dotenv()

print("Loading your trained disease classifier...\n")

# Load your trained disease classifier
disease_classifier = YOLO('runs/classify/plant_disease_v1/weights/best.pt')

print("Model loaded!\n")


def preprocess_for_classification(img):
    """
    Enhance image quality before classification
    """
    # Resize to training size
    img_resized = cv2.resize(img, (224, 224))
    
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return denoised


def detect_plant_regions(image_path):
    """
    Detect both healthy (green) and diseased (yellow/brown) leaf regions
    Returns multiple regions for analysis
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        return None, []
    
    height, width = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    print("Stage 1: Detecting plant regions (green, yellow, brown)...")
    
    # Define color ranges for different leaf states
    # Green regions (healthy leaves)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Yellow regions (diseased leaves)
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Brown regions (severely diseased/dead)
    lower_brown = np.array([5, 40, 20])
    upper_brown = np.array([20, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Combine all plant-related colors
    plant_mask = cv2.bitwise_or(green_mask, yellow_mask)
    plant_mask = cv2.bitwise_or(plant_mask, brown_mask)
    
    # Clean up mask (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours (individual leaves/regions)
    contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No plant regions detected.\n")
        return img, []
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Get top regions (up to 5)
    min_area = (width * height) * 0.03  # At least 3% of image
    regions = []
    
    for i, contour in enumerate(contours[:5]):
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding (10%)
        padding_x = int(w * 0.1)
        padding_y = int(h * 0.1)
        
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(width - x, w + 2 * padding_x)
        h = min(height - y, h + 2 * padding_y)
        
        # Crop region
        cropped = img[y:y+h, x:x+w]
        
        if cropped.size == 0:
            continue
        
        regions.append({
            'image': cropped,
            'bbox': (x, y, w, h),
            'area': area,
            'index': i
        })
    
    print(f"Found {len(regions)} significant plant regions.\n")
    
    return img, regions


def classify_region(region_img, region_index):
    """
    Classify a single region with preprocessing
    """
    # Preprocess
    processed = preprocess_for_classification(region_img)
    
    # Save temporarily
    temp_path = f'temp_region_{region_index}.jpg'
    cv2.imwrite(temp_path, processed)
    
    # Classify
    results = disease_classifier(temp_path, verbose=False)
    probs = results[0].probs
    
    # Cleanup
    os.remove(temp_path)
    
    return {
        'disease': disease_classifier.names[probs.top1],
        'confidence': float(probs.top1conf),
        'top3': [
            {
                'disease': disease_classifier.names[idx],
                'confidence': float(probs.data[idx])
            }
            for idx in probs.top5[:3]
        ]
    }


def multi_region_voting(predictions, regions):
    """
    Combine predictions from multiple regions using weighted voting
    Weight = area * confidence
    """
    if not predictions:
        return None
    
    # Weighted voting
    disease_votes = {}
    total_weight = 0
    
    for pred, region in zip(predictions, regions):
        disease = pred['disease']
        weight = region['area'] * pred['confidence']
        
        if disease not in disease_votes:
            disease_votes[disease] = {'weight': 0, 'count': 0, 'confidences': []}
        
        disease_votes[disease]['weight'] += weight
        disease_votes[disease]['count'] += 1
        disease_votes[disease]['confidences'].append(pred['confidence'])
        total_weight += weight
    
    # Get final prediction
    final_disease = max(disease_votes, key=lambda k: disease_votes[k]['weight'])
    final_weight = disease_votes[final_disease]['weight']
    final_confidence = final_weight / total_weight if total_weight > 0 else 0
    
    # Average confidence for this disease across regions
    avg_confidence = np.mean(disease_votes[final_disease]['confidences'])
    
    return {
        'disease': final_disease,
        'confidence': float(avg_confidence),
        'weighted_confidence': float(final_confidence),
        'region_count': disease_votes[final_disease]['count'],
        'all_votes': disease_votes
    }


def analyze_plant(image_path, method='multi_region'):
    """
    Analyze plant disease with multiple accuracy improvements
    
    Methods:
    - 'single': Detect largest region and classify (fast)
    - 'multi_region': Analyze multiple regions and vote (accurate)
    """
    
    print(f"Analyzing: {image_path}")
    print(f"Method: {method}\n")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return None
    
    # Detect plant regions
    original_img, regions = detect_plant_regions(image_path)
    
    if not regions:
        print("No plant regions detected. Trying center crop fallback...\n")
        return fallback_center_crop(image_path)
    
    # Method 1: Single region (fastest)
    if method == 'single':
        print("Stage 2: Classifying largest region...\n")
        largest = regions[0]
        prediction = classify_region(largest['image'], 0)
        
        result = {
            'image': image_path,
            'method': 'single_region',
            'prediction': prediction['disease'],
            'confidence': prediction['confidence'],
            'top3': prediction['top3'],
            'regions_detected': len(regions)
        }
        
        visualize_result(original_img, [largest], [prediction], result)
        return result
    
    # Method 2: Multi-region voting (most accurate)
    elif method == 'multi_region':
        print("Stage 2: Classifying all detected regions...\n")
        
        predictions = []
        for i, region in enumerate(regions):
            pred = classify_region(region['image'], i)
            predictions.append(pred)
            print(f"  Region {i+1}: {pred['disease']} ({pred['confidence']*100:.1f}%)")
        
        print("\nStage 3: Combining predictions with weighted voting...\n")
        
        final = multi_region_voting(predictions, regions)
        
        result = {
            'image': image_path,
            'method': 'multi_region_voting',
            'prediction': final['disease'],
            'confidence': final['confidence'],
            'weighted_confidence': final['weighted_confidence'],
            'regions_analyzed': len(regions),
            'regions_agreeing': final['region_count'],
            'individual_predictions': [p['disease'] for p in predictions],
            'voting_details': final['all_votes']
        }
        
        visualize_result(original_img, regions, predictions, result)
        return result


def fallback_center_crop(image_path):
    """
    Fallback method when no regions detected
    """
    print("Using center crop fallback...\n")
    
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    size = min(h, w)
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    cropped = img[start_y:start_y+size, start_x:start_x+size]
    
    # Preprocess and classify
    processed = preprocess_for_classification(cropped)
    temp_path = 'temp_center.jpg'
    cv2.imwrite(temp_path, processed)
    
    results = disease_classifier(temp_path, verbose=False)
    probs = results[0].probs
    
    os.remove(temp_path)
    
    return {
        'image': image_path,
        'method': 'center_crop_fallback',
        'prediction': disease_classifier.names[probs.top1],
        'confidence': float(probs.top1conf),
        'top3': [
            {
                'disease': disease_classifier.names[idx],
                'confidence': float(probs.data[idx])
            }
            for idx in probs.top5[:3]
        ]
    }


def visualize_result(original_img, regions, predictions, result):
    """
    Visualize detection and classification results
    """
    fig, axes = plt.subplots(1, min(3, len(regions) + 1), figsize=(15, 5))
    
    if len(regions) == 0:
        axes = [axes]
    
    # Original image with bounding boxes
    img_with_boxes = original_img.copy()
    for i, region in enumerate(regions[:5]):
        x, y, w, h = region['bbox']
        color = (0, 255, 0) if i < len(predictions) else (128, 128, 128)
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), color, 3)
        
        if i < len(predictions):
            label = f"#{i+1}: {predictions[i]['disease'][:15]}"
            cv2.putText(img_with_boxes, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    
    if len(axes) > 1:
        axes[0].imshow(img_rgb)
        axes[0].set_title(f'Detected Regions: {len(regions)}', fontsize=10)
        axes[0].axis('off')
        
        # Show top 2 regions
        for i in range(min(2, len(regions))):
            if i + 1 < len(axes):
                region_rgb = cv2.cvtColor(regions[i]['image'], cv2.COLOR_BGR2RGB)
                axes[i+1].imshow(region_rgb)
                
                pred = predictions[i] if i < len(predictions) else None
                if pred:
                    title = f"Region {i+1}\n{pred['disease'][:20]}\n{pred['confidence']*100:.1f}%"
                    axes[i+1].set_title(title, fontsize=9)
                axes[i+1].axis('off')
    else:
        axes[0].imshow(img_rgb)
        axes[0].set_title('Detection Result', fontsize=10)
        axes[0].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = 'improved_detection_result.jpg'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python improved_plant_detection.py <image_path> [method]")
        print("\nMethods:")
        print("  single       - Fast: Analyze only largest region")
        print("  multi_region - Accurate: Analyze multiple regions and vote (default)")
        print("\nExamples:")
        print("  python improved_plant_detection.py tomato.jpg")
        print("  python improved_plant_detection.py tomato.jpg single")
        print("  python improved_plant_detection.py tomato.jpg multi_region")
        sys.exit(1)
    
    image_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'multi_region'
    
    if method not in ['single', 'multi_region']:
        print(f"Error: Unknown method '{method}'")
        print("Use 'single' or 'multi_region'")
        sys.exit(1)
    
    # Analyze
    result = analyze_plant(image_path, method=method)
    
    if result:
        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nImage: {result['image']}")
        print(f"Method: {result['method']}")
        
        if 'regions_detected' in result:
            print(f"Regions detected: {result['regions_detected']}")
        
        if 'regions_analyzed' in result:
            print(f"Regions analyzed: {result['regions_analyzed']}")
            print(f"Regions agreeing: {result['regions_agreeing']}")
        
        print(f"\nFinal Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        
        if 'weighted_confidence' in result:
            print(f"Weighted Confidence: {result['weighted_confidence']*100:.1f}%")
        
        if 'individual_predictions' in result:
            print(f"\nIndividual region predictions:")
            for i, pred in enumerate(result['individual_predictions'], 1):
                print(f"  Region {i}: {pred}")
        
        if 'top3' in result:
            print(f"\nTop 3 Predictions:")
            for i, pred in enumerate(result['top3'], 1):
                print(f"  {i}. {pred['disease']}: {pred['confidence']*100:.1f}%")
        
        print("\n" + "=" * 70)