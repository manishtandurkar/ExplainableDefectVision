# 🏭 Explainable Multi-Class Industrial Defect Detection

## 🎯 Project Overview

This project implements an explainable AI system for industrial surface defect detection using Vision Transformers (ViT) with dual-layer explainability combining Grad-CAM and attention visualization.

## ✨ Key Features

- **🤖 Vision Transformer Model**: Fine-tuned ViT for 6-class defect classification
- **🔍 Dual Explainability**: Grad-CAM + Attention map fusion
- **💬 Root-Cause Explanations**: Human-readable defect analysis
- **🌐 Interactive GUI**: Streamlit web application
- **📊 Comprehensive Evaluation**: Detailed performance metrics

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/manishtandurkar/ExplainableDefectVision.git
cd ExplainableDefectVision

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

**Option A: Manual Download**
1. Visit [Kaggle NEU Metal Surface Defects Dataset](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data)
2. Download the dataset (requires Kaggle account)
3. Extract to `data/` folder with the following structure:
```
data/
├── train/
│   ├── Crazing/
│   ├── Inclusion/
│   ├── Patches/
│   ├── Pitted/
│   ├── Rolled/
│   └── Scratches/
├── valid/
│   └── [same structure]
└── test/
    └── [same structure]
```

**Option B: Using Kaggle API**
```bash
# Install Kaggle CLI
pip install kaggle

# Set up Kaggle credentials (get from kaggle.com/account)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d fantacher/neu-metal-surface-defects-data

# Extract and organize into train/valid/test splits
unzip neu-metal-surface-defects-data.zip -d data/
```

### 3. Train Model (Google Colab)
```bash
# 1. Upload dataset to Google Drive
#    Create folder: My Drive/data/
#    Upload the organized dataset (train/valid/test folders with 6 class subfolders each)

# 2. Upload ExplainableDefectDetection_Colab.ipynb to Google Colab

# 3. In Colab, mount Google Drive when prompted
#    The notebook will automatically access data from Drive/data/

# 4. Run all cells to train the model
#    Model checkpoints will be saved to Drive/models/

# 5. Download trained model (best_vit_defect_detector.pt) from Drive/models/
#    Place in local models/ folder for inference
```

### 4. Run Inference & Explainability (Local)
```bash
# Place trained model in models/ folder
jupyter notebook ExplainableDefectDetection_Inference.ipynb
```

### 5. Launch Streamlit App (Optional)
```bash
streamlit run streamlit_app.py
```

## 📊 Dataset

- **NEU Surface Defect Dataset** ([Kaggle Link](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data))
- **Source**: Northeastern University (NEU) Metal Surface Defects Database
- **6 Defect Classes**: Crazing, Inclusion, Patches, Pitted, Rolled, Scratches
- **1,800 images** (300 per class) organized in train/valid/test splits
- **Formats**: BMP, JPG (200×200 pixels grayscale)
- **Application**: Hot-rolled steel strip surface inspection

## 🏗️ Architecture

```
Input Image → ViT Model → Prediction
     ↓
Explainability Layer:
- Grad-CAM Heatmap
- Attention Visualization  
- Fusion & Root-cause Explanation
     ↓
Interactive GUI Display
```

## 📈 Performance

- **Training Accuracy**: 100% (achieved in 10 epochs)
- **Validation Accuracy**: 100% on all 6 classes
- **Training Time**: ~9 minutes on Google Colab (T4 GPU)
- **Explainability**: Dual-layer visual + textual
- **Inference**: Real-time with interactive web interface

## 🎯 Defect Classes

1. **Crazing**: Surface cracks from thermal stress
2. **Inclusion**: Embedded impurities in steel
3. **Patches**: Irregular surface texture
4. **Pitted**: Corrosion pits (Pitted Surface)
5. **Rolled**: Embedded oxide particles (Rolled-in Scale)
6. **Scratches**: Mechanical abrasions

## 🔄 Workflow

### Training Phase (Google Colab)
1. **Upload dataset** to Google Drive in organized structure:
   - Create `My Drive/data/train/`, `My Drive/data/valid/`, `My Drive/data/test/`
   - Each folder should contain 6 subfolders: Crazing, Inclusion, Patches, Pitted, Rolled, Scratches
2. **Open** `ExplainableDefectDetection_Colab.ipynb` in Colab
3. **Mount** Google Drive when prompted (click the folder icon or run mount cell)
4. **Run** all cells to train the Vision Transformer
5. **Download** trained model from `My Drive/models/best_vit_defect_detector.pt`

### Inference Phase (Local)
1. **Place** trained model in local `models/` folder
2. **Open** `ExplainableDefectDetection_Inference.ipynb`
3. **Run** explainability analysis (Grad-CAM, LIME, SHAP)
4. **Generate** attention visualizations and reports

### Deployment (Optional)
- Launch Streamlit app for interactive defect detection
- Real-time predictions with visual explanations

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, Transformers (HuggingFace)
- **Explainability**: Grad-CAM, Attention Rollout
- **Visualization**: Matplotlib, Seaborn, OpenCV
- **Deployment**: Streamlit
- **Data Science**: NumPy, Pandas, Scikit-learn

## 📁 Project Structure

```
ExplainableDefectVision/
├── ExplainableDefectDetection_Colab.ipynb      # Training (Google Colab)
├── ExplainableDefectDetection_Inference.ipynb  # Inference & Explainability (Local)
├── streamlit_app.py                            # Web application
├── requirements.txt                            # Dependencies
├── data/                                       # Dataset directory
│   ├── train/                                  # Training images
│   │   ├── Crazing/
│   │   ├── Inclusion/
│   │   ├── Patches/
│   │   ├── Pitted/
│   │   ├── Rolled/
│   │   └── Scratches/
│   ├── valid/                                  # Validation images
│   └── test/                                   # Test images
├── models/                                     # Trained models
│   └── best_vit_defect_detector.pt            # Best model weights
└── visualizations/                             # Generated plots
```

## 🎨 Sample Output

**Input**: Steel surface image with scratches  
**Predicted**: Scratches (94% confidence)  
**Explanation**: "Linear mechanical abrasions caused by contact with tools or processing machinery"  
**Visualization**: Red-highlighted zones showing scratch patterns with attention overlay

## 🏆 Innovation Highlights

- ✨ **Dual Explainability Fusion**: Novel combination of Grad-CAM and ViT attention
- 💬 **Natural Language Explanations**: Human-readable root-cause analysis  
- 🎯 **Real-world Application**: Industrial quality control focus
- 🖥️ **Interactive Deployment**: User-friendly web interface

## 📚 References

- Vision Transformer (Dosovitskiy et al., 2020)
- Grad-CAM (Selvaraju et al., 2017)
- NEU Surface Defect Database
- Attention Rollout for Vision Transformers

---

**Tagline**: *"Not just seeing defects — understanding them."*  
A project that blends AI accuracy with human interpretability.