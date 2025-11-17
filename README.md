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
pip install -r requirements.txt
```

### 2. Run Jupyter Notebook
```bash
jupyter notebook ExplainableDefectDetection.ipynb
```

### 3. Launch Streamlit App
```bash
streamlit run streamlit_app.py
```

## 📊 Dataset

- **NEU Surface Defect Dataset**
- **6 Defect Classes**: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches
- **1,800 images** (300×300 pixels)

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

## 📈 Expected Performance

- **Accuracy**: >92%
- **Explainability**: Dual-layer visual + textual
- **Real-time**: Interactive web interface

## 🎯 Defect Classes

1. **Crazing**: Surface cracks from thermal stress
2. **Inclusion**: Embedded impurities in steel
3. **Patches**: Irregular surface texture
4. **Pitted Surface**: Corrosion pits
5. **Rolled-in Scale**: Embedded oxide particles
6. **Scratches**: Mechanical abrasions

## 👥 Team Structure (2-Month Project)

- **Week 1**: Dataset preparation & EDA
- **Week 2**: Data preprocessing & augmentation  
- **Week 3-4**: ViT model training
- **Week 5**: Explainability implementation
- **Week 6**: GUI development
- **Week 7**: Testing & optimization
- **Week 8**: Documentation & presentation

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, Transformers (HuggingFace)
- **Explainability**: Grad-CAM, Attention Rollout
- **Visualization**: Matplotlib, Seaborn, OpenCV
- **Deployment**: Streamlit
- **Data Science**: NumPy, Pandas, Scikit-learn

## 📁 Project Structure

```
ExplainableDefectVision/
├── ExplainableDefectDetection.ipynb  # Main notebook
├── streamlit_app.py                  # Web application
├── requirements.txt                  # Dependencies
├── data/                            # Dataset directory
├── models/                          # Trained models
└── visualizations/                  # Generated plots
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