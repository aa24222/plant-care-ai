# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from roboflow import Roboflow
from ultralytics import YOLO
from PIL import Image as PILImage
import cv2
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration from .env
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

if not ROBOFLOW_API_KEY:
    raise ValueError("Error: ROBOFLOW_API_KEY not found! Create a .env file with your API key.")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
print("Loading models...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
leaf_detector = rf.workspace("joseph-nelson").project("leaves").version(2).model
disease_classifier = YOLO('../runs/classify/plant_disease_v1/weights/best.pt')
print("Models loaded!\n")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': True,
        'detection': 'Roboflow',
        'classification': 'YOLOv8 Custom'
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Save uploaded file
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # Stage 1: Detect leaves
        detections = leaf_detector.predict(filepath, confidence=30).json()
        num_leaves = len(detections['predictions'])
        
        results = []
        
        if num_leaves == 0:
            # Fallback: center crop
            img = cv2.imread(filepath)
            h, w = img.shape[:2]
            size = min(h, w)
            start_x = (w - size) // 2
            start_y = (h - size) // 2
            cropped = img[start_y:start_y+size, start_x:start_x+size]
            
            temp_path = os.path.join(UPLOAD_FOLDER, 'temp_crop.jpg')
            cv2.imwrite(temp_path, cropped)
            
            disease_result = disease_classifier(temp_path, verbose=False)
            probs = disease_result[0].probs
            
            # Get top 3
            top3 = []
            for idx in probs.top5[:3]:
                top3.append({
                    'disease': disease_classifier.names[idx],
                    'confidence': float(probs.data[idx])
                })
            
            results.append({
                'leaf_number': 1,
                'method': 'center_crop',
                'disease': disease_classifier.names[probs.top1],
                'confidence': float(probs.top1conf),
                'top3': top3
            })
            
            os.remove(temp_path)
        else:
            # Stage 2: Classify each detected leaf
            img = cv2.imread(filepath)
            
            for i, detection in enumerate(detections['predictions'], 1):
                x = int(detection['x'])
                y = int(detection['y'])
                width = int(detection['width'])
                height = int(detection['height'])
                
                x1 = max(0, x - width // 2)
                y1 = max(0, y - height // 2)
                x2 = min(img.shape[1], x + width // 2)
                y2 = min(img.shape[0], y + height // 2)
                
                leaf_crop = img[y1:y2, x1:x2]
                
                if leaf_crop.shape[0] < 20 or leaf_crop.shape[1] < 20:
                    continue
                
                temp_path = os.path.join(UPLOAD_FOLDER, f'temp_leaf_{i}.jpg')
                cv2.imwrite(temp_path, leaf_crop)
                
                disease_result = disease_classifier(temp_path, verbose=False)
                probs = disease_result[0].probs
                
                # Get top 3
                top3 = []
                for idx in probs.top5[:3]:
                    top3.append({
                        'disease': disease_classifier.names[idx],
                        'confidence': float(probs.data[idx])
                    })
                
                results.append({
                    'leaf_number': i,
                    'disease': disease_classifier.names[probs.top1],
                    'confidence': float(probs.top1conf),
                    'top3': top3
                })
                
                os.remove(temp_path)
        
        # Cleanup
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'leaves_detected': num_leaves,
            'results': results
        })
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Two-Stage Plant Disease Detection API")
    print("Stage 1: Roboflow leaf detection")
    print("Stage 2: Custom YOLOv8 disease classification")
    print("\nStarting server on http://localhost:5000\n")
    app.run(debug=True, port=5000)