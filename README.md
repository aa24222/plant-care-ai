# 🌱 Plant Disease Detection with YOLOv8

AI-powered computer vision system for detecting plant diseases in tomatoes, peppers, and potatoes.

## Project Overview

- **Model:** YOLOv8 Classification (Nano)
- **Dataset:** PlantVillage - 41,272 images across 16 disease classes
- **Accuracy:** ~92% validation accuracy
- **Tech Stack:** Python, PyTorch, Ultralytics YOLOv8, Flask (planned)

## Detected Diseases

| Plant | Disease Classes | Count |
|-------|----------------|-------|
| **Peppers** | Bacterial Spot, Healthy | 2 |
| **Potatoes** | Early Blight, Late Blight, Healthy | 3 |
| **Tomatoes** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted), Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy | 10 |

**Total Classes:** 15

## Getting Started
> **Note:** This project uses transfer learning. The base model (`yolov8n-cls.pt`) downloads automatically from Ultralytics. Your fine-tuned weights (`best.pt`) are created during training and saved to `runs/classify/plant_disease_v1/weights/`.

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detector.git
cd plant-disease-detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download the PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease):
```bash
# Option 1: Manual download
# Download and extract to: datasets/plantvillage/

# Option 2: Kaggle CLI
pip install kaggle
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d datasets/plantvillage/
```

Expected structure:
```
Garden/
└── datasets/
    └── plantvillage/
        └── train/
            └── PlantVillage/
                ├── Pepper__bell___Bacterial_spot/
                ├── Tomato_Early_blight/
                └── ... (16 total classes)
```

### 4. Train Model
Open and run `train_model.ipynb` in Jupyter:
```bash
jupyter notebook train_model.ipynb
```

## Model Performance

- **Training Accuracy:** 100%
- **Validation Accuracy:** 92%
- **Training Time:** ~50 minutes (NVIDIA RTX 4060)
- **Inference Time:** ~50ms per image

## Project Structure
```
plant-disease-detector/
├── .gitignore
├── README.md
├── requirements.txt
├── explore_data.ipynb       # Dataset exploration & statistics
├── train_model.ipynb        # Model training notebook
└── frontend/
    └── index.html           # Web interface (coming soon)
```

## Technologies

- **Deep Learning:** Ultralytics YOLOv8
- **Framework:** PyTorch
- **Data Processing:** OpenCV, Pillow, Pandas
- **Visualization:** Matplotlib
- **GPU:** CUDA (NVIDIA)

## Model Weights Explained

This project leverages **transfer learning** for efficient training:

### Pretrained Base Model
- **File:** `yolov8n-cls.pt` (~6 MB)
- **Source:** Downloaded automatically from [Ultralytics GitHub](https://github.com/ultralytics/assets/releases)
- **Training:** Pre-trained on ImageNet dataset (1.2M images, 1000 object classes)
- **Purpose:** Provides foundational knowledge of image features (edges, textures, patterns)
- **Location:** Cached in `~/.cache/torch/hub/ultralytics/` after first download

### Fine-tuned Model
- **File:** `best.pt` (~6 MB)
- **Created:** During training when you run `model.train()`
- **Location:** `runs/classify/plant_disease_v1/weights/best.pt`
- **Specialization:** Fine-tuned specifically for 16 plant disease classes
- **Note:** Not included in this repository due to file size. Train it yourself following the [Getting Started](#getting-started) guide.

**Why Transfer Learning?**
- ✅ Faster training (~50 min vs hours from scratch)
- ✅ Better accuracy with less data (92% with 41k images)
- ✅ Leverages existing knowledge from millions of images
  
## Roadmap

- [x] Dataset exploration
- [x] YOLOv8 model training
- [x] Achieve 90%+ validation accuracy
- [ ] Build Flask REST API
- [ ] Connect frontend to backend
- [ ] Add support for whole plant photos (two-stage detection)
- [ ] Deploy to cloud (Render/Railway)
- [ ] Mobile app (future)

## Features (Planned)

- Upload plant photos for instant diagnosis
- Confidence scores for predictions
- Care recommendations based on detected diseases
- Plant health tracking over time

## License

MIT License

## Author

Ayesha Afia [https://github.com/aa24222]

## Acknowledgments

- PlantVillage Dataset
- Ultralytics YOLOv8
- Kaggle Community
