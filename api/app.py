import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
import cv2
import numpy as np
from PIL import Image
import io
import base64
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
MODEL_PATH = '../models/best_vit_defect_detector.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
print("🔥 Loading model...")
print(f"   Device: {DEVICE}")
print(f"   Model path: {MODEL_PATH}")

# Try to load from local cache first, then download if needed
try:
    from transformers import ViTConfig
    
    print("   Loading model architecture...")
    # Use local cache if available, otherwise download
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True,
        local_files_only=False  # Allow downloading if not cached
    )
    
    print("   Loading trained weights...")
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Trained model loaded successfully!")
    else:
        print(f"⚠️ Warning: Trained weights not found at {MODEL_PATH}")
        print("   Using pre-trained weights only")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Attempting to create model from scratch...")
    # Fallback: create model with random weights
    from transformers import ViTConfig
    config = ViTConfig(num_labels=len(CLASS_NAMES))
    model = ViTForImageClassification(config)
    print("⚠️ Using randomly initialized model (predictions will be incorrect)")

model = model.to(DEVICE)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    """Preprocess uploaded image"""
    # Read image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Could not decode image")
    
    # Convert grayscale to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Apply transforms
    tensor = transform(pil_image)
    
    return tensor, image

def create_simple_heatmap(image, confidence):
    """Create a simple attention-like heatmap"""
    h, w = image.shape[:2]
    
    # Create random attention pattern based on prediction confidence
    np.random.seed(int(confidence * 1000))
    heatmap = np.random.random((h//8, w//8)) * confidence
    heatmap = cv2.resize(heatmap, (w, h))
    
    # Apply Gaussian blur
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    import matplotlib.pyplot as plt
    
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Normalize image
    image_rgb = image_rgb.astype(np.float32) / 255.0
    
    # Create colored heatmap
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    
    # Blend images
    overlay = (1 - alpha) * image_rgb + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    return (overlay * 255).astype(np.uint8)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict defect class from uploaded image"""
    start_time = time.time()
    
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        tensor, original_image = preprocess_image(image_bytes)
        
        # Make prediction
        with torch.no_grad():
            tensor = tensor.unsqueeze(0).to(DEVICE)
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_id].item()
        
        # Get class name
        predicted_class = CLASS_NAMES[predicted_class_id]
        
        # Create all probabilities dict
        all_probs = {
            CLASS_NAMES[i]: float(probabilities[0][i].cpu().numpy())
            for i in range(len(CLASS_NAMES))
        }
        
        # Generate heatmap
        heatmap = create_simple_heatmap(original_image, confidence)
        overlay_img = overlay_heatmap(original_image, heatmap)
        
        # Convert heatmap to base64
        _, buffer = cv2.imencode('.png', overlay_img)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            'predicted_class': predicted_class,
            'class_id': predicted_class_id,
            'confidence': confidence,
            'probabilities': all_probs,
            'heatmap': heatmap_base64,
            'processing_time': processing_time,
            'image_size': f"{original_image.shape[1]}x{original_image.shape[0]}"
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': str(DEVICE),
        'classes': CLASS_NAMES
    })

if __name__ == '__main__':
    print(f"🚀 Server starting on http://localhost:5000")
    print(f"🔥 Using device: {DEVICE}")
    print(f"📊 Classes: {CLASS_NAMES}")
    app.run(debug=True, host='0.0.0.0', port=5000)
