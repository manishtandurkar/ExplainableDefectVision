import os
import sys

# Suppress all warnings and TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Disable TensorFlow backend in transformers
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import warnings
warnings.filterwarnings('ignore')

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

def fix_state_dict_keys(state_dict):
    """Fix state_dict keys by removing extra 'vit.' prefix if present"""
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove extra 'vit.' prefix from keys like 'vit.vit.embeddings...'
        if key.startswith('vit.vit.'):
            new_key = key.replace('vit.vit.', 'vit.', 1)
            new_state_dict[new_key] = value
        # Also handle classifier keys
        elif key.startswith('vit.classifier.'):
            new_key = key.replace('vit.classifier.', 'classifier.', 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

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
        # Fix state_dict keys if needed
        state_dict = fix_state_dict_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
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

def compute_rollout_attention(attentions, discard_ratio=0.1):
    """Compute attention rollout from all layers"""
    result = torch.eye(attentions[0].size(-1))
    
    for attention in attentions:
        # Average attention heads
        attention_heads_fused = attention.mean(dim=1)
        
        # Drop lowest attentions
        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1) * discard_ratio), largest=False)
        flat[0, indices] = 0
        
        # Normalize
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
        
        # Multiply with previous result
        result = torch.matmul(attention_heads_fused[0], result)
    
    # Look at attention to CLS token
    mask = result[0, 1:]
    
    return mask

def get_attention_based_saliency(model, tensor, target_class, original_image):
    """Generate saliency map using model attention and input gradients"""
    
    # Method 1: Input gradient saliency (most direct)
    tensor.requires_grad = True
    
    # Forward pass
    outputs = model(tensor, output_attentions=True)
    score = outputs.logits[0, target_class]
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    # Get input gradients
    saliency = tensor.grad.data.abs()
    
    # Average across batch and channels, keep spatial dimensions
    saliency_map = saliency[0].mean(dim=0).cpu().numpy()
    
    # Method 2: Combine with attention rollout if available
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        try:
            attention_rollout = compute_rollout_attention(outputs.attentions)
            
            # Reshape attention rollout to 2D grid (14x14 for ViT)
            grid_size = int(np.sqrt(attention_rollout.size(0)))
            attention_map = attention_rollout.reshape(grid_size, grid_size).cpu().numpy()
            
            # Resize to match saliency map
            attention_map = cv2.resize(attention_map, (saliency_map.shape[1], saliency_map.shape[0]))
            
            # Combine both methods (weighted average)
            combined_map = 0.6 * saliency_map + 0.4 * attention_map
        except:
            combined_map = saliency_map
    else:
        combined_map = saliency_map
    
    # Normalize
    combined_map = np.maximum(combined_map, 0)
    if combined_map.max() > 0:
        combined_map = combined_map / combined_map.max()
    
    return combined_map

def create_heatmap_from_attention(attention_map, original_image_shape, original_image):
    """Convert attention map to heatmap overlay with defect-aware enhancement"""
    h, w = original_image_shape[:2]
    
    # Resize attention map to original image size
    heatmap = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Enhance defect regions using image variance
    # Calculate local variance to find anomalous regions
    kernel_size = max(5, min(h, w) // 40)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Detect edges and texture anomalies in original image
    if len(original_image.shape) == 2:
        img_for_analysis = original_image
    else:
        img_for_analysis = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if len(original_image.shape) == 3 else original_image
    
    # Edge detection
    edges = cv2.Canny(img_for_analysis, 50, 150)
    edges = edges.astype(np.float32) / 255.0
    
    # Local standard deviation (texture variation)
    mean = cv2.blur(img_for_analysis.astype(np.float32), (kernel_size, kernel_size))
    mean_sq = cv2.blur(img_for_analysis.astype(np.float32)**2, (kernel_size, kernel_size))
    std_dev = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    std_dev = std_dev / (std_dev.max() + 1e-8)
    
    # Combine model attention with image-based features
    enhanced_heatmap = heatmap * 0.5 + edges * 0.2 + std_dev * 0.3
    
    # Smooth while preserving important regions
    enhanced_heatmap = cv2.GaussianBlur(enhanced_heatmap, (kernel_size, kernel_size), 0)
    
    # Apply adaptive thresholding to focus on high-attention areas
    threshold = np.percentile(enhanced_heatmap, 70)
    enhanced_heatmap = np.where(enhanced_heatmap > threshold, enhanced_heatmap, enhanced_heatmap * 0.3)
    
    # Normalize
    if enhanced_heatmap.max() > enhanced_heatmap.min():
        enhanced_heatmap = (enhanced_heatmap - enhanced_heatmap.min()) / (enhanced_heatmap.max() - enhanced_heatmap.min() + 1e-8)
    
    # Enhance contrast
    enhanced_heatmap = np.power(enhanced_heatmap, 0.8)
    
    return enhanced_heatmap

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
        
        # Make prediction (first pass without gradients)
        with torch.no_grad():
            tensor_no_grad = tensor.unsqueeze(0).to(DEVICE)
            outputs = model(tensor_no_grad)
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
        
        # Generate attention-based heatmap with defect-aware saliency
        model.train()  # Need to enable gradients
        tensor_with_grad = tensor.unsqueeze(0).to(DEVICE)
        attention_map = get_attention_based_saliency(model, tensor_with_grad, predicted_class_id, original_image)
        heatmap = create_heatmap_from_attention(attention_map, original_image.shape, original_image)
        overlay_img = overlay_heatmap(original_image, heatmap)
        model.eval()  # Back to eval mode
        
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
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
