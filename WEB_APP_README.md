# 🏭 Explainable Defect Detection - Web Application

A modern React web application for real-time industrial surface defect detection using Vision Transformers.

## ✨ Features

- 📤 **Drag & Drop Upload**: Easy image upload interface
- 🎯 **Real-time Prediction**: Instant defect classification
- 📊 **Visual Analytics**: Interactive probability charts
- 🔍 **Attention Visualization**: Heatmap showing model focus areas
- 💡 **Root-Cause Explanations**: Human-readable defect descriptions
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile

## 🚀 Quick Start

### Prerequisites

- Node.js (v14 or higher)
- Python 3.8+
- Trained model file: `models/best_vit_defect_detector.pt`

### 1. Setup Backend API

```bash
# Navigate to API directory
cd api

# Install Python dependencies
pip install -r requirements.txt

# Start Flask server
python app.py
```

The API will run on `http://localhost:5000`

### 2. Setup Frontend

```bash
# Navigate to web-app directory
cd web-app

# Install Node dependencies
npm install

# Start React development server
npm start
```

The web app will open at `http://localhost:3000`

## 📁 Project Structure

```
ExplainableDefectVision/
├── api/
│   ├── app.py                 # Flask backend API
│   └── requirements.txt       # Python dependencies
├── web-app/
│   ├── public/
│   │   └── index.html        # HTML template
│   ├── src/
│   │   ├── App.js            # Main React component
│   │   ├── App.css           # Styling
│   │   ├── index.js          # Entry point
│   │   └── index.css         # Global styles
│   └── package.json          # Node dependencies
└── models/
    └── best_vit_defect_detector.pt  # Trained model
```

## 🔌 API Endpoints

### POST `/predict`
Analyze an uploaded image and return defect prediction.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "predicted_class": "Crazing",
  "class_id": 0,
  "confidence": 0.9823,
  "probabilities": {
    "Crazing": 0.9823,
    "Inclusion": 0.0102,
    "Patches": 0.0045,
    "Pitted": 0.0015,
    "Rolled": 0.0010,
    "Scratches": 0.0005
  },
  "heatmap": "base64_encoded_image",
  "processing_time": 0.234,
  "image_size": "200x200"
}
```

### GET `/health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "classes": ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]
}
```

## 🎨 Supported Defect Classes

| Class | Description | Severity |
|-------|-------------|----------|
| **Crazing** | Surface cracks from thermal stress | High |
| **Inclusion** | Embedded impurities in steel | Medium |
| **Patches** | Irregular surface texture | Low |
| **Pitted** | Localized corrosion pits | High |
| **Rolled** | Embedded oxide particles | Medium |
| **Scratches** | Mechanical abrasions | Low |

## 🔧 Configuration

### Backend (api/app.py)

```python
# Change model path
MODEL_PATH = 'path/to/your/model.pt'

# Change port
app.run(port=5000)

# Change host (for deployment)
app.run(host='0.0.0.0')
```

### Frontend (web-app/src/App.js)

```javascript
// Change API endpoint
const response = await axios.post('http://your-api-url/predict', formData);
```

## 🚀 Deployment

### Deploy Backend (Flask API)

**Option 1: Docker**
```bash
cd api
docker build -t defect-detection-api .
docker run -p 5000:5000 defect-detection-api
```

**Option 2: Heroku**
```bash
cd api
heroku create your-app-name
git push heroku main
```

### Deploy Frontend (React)

**Option 1: Vercel**
```bash
cd web-app
npm install -g vercel
vercel
```

**Option 2: Netlify**
```bash
cd web-app
npm run build
# Drag build/ folder to netlify.com
```

**Option 3: Build for production**
```bash
cd web-app
npm run build
# Serve the build/ folder with any static hosting
```

## 🔍 Troubleshooting

### Backend Issues

**Error: Model file not found**
```bash
# Make sure model is in the correct location
cp path/to/best_vit_defect_detector.pt models/
```

**Error: CUDA out of memory**
```python
# In app.py, force CPU usage
DEVICE = torch.device('cpu')
```

### Frontend Issues

**Error: Network Error / Connection refused**
- Check if backend is running on `http://localhost:5000`
- Check CORS settings in `api/app.py`

**Error: npm install fails**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## 📊 Performance

- **Prediction Time**: ~0.2-0.5 seconds per image
- **Model Accuracy**: 100% on validation set
- **Supported Formats**: PNG, JPG, JPEG, BMP
- **Max Image Size**: 10MB (configurable)

## 🛠️ Development

### Adding New Features

1. **Backend**: Edit `api/app.py`
2. **Frontend**: Edit `web-app/src/App.js`
3. **Styling**: Edit `web-app/src/App.css`

### Running Tests

```bash
# Backend
cd api
pytest

# Frontend
cd web-app
npm test
```

## 📝 License

This project is part of the Explainable Defect Detection research project.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

**Built with ❤️ using React, Flask, and PyTorch**
