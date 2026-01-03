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
try:
    # Optional: pytorch-grad-cam for better heatmaps
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image as cam_preprocess_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _HAS_GRADCAM = True
except Exception:
    _HAS_GRADCAM = False

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

# Create model locally and load your fine-tuned weights (no HF network calls)
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import ViTConfig, ViTForImageClassification

print("   Creating ViT model config locally (no Hugging Face downloads)")
config = ViTConfig(num_labels=len(CLASS_NAMES))
_base_model = ViTForImageClassification(config)

print("   Loading trained weights (local file)")
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    # Fix state_dict keys if needed
    state_dict = fix_state_dict_keys(state_dict)
    _base_model.load_state_dict(state_dict, strict=False)
    print("✅ Trained model loaded successfully from local file")
else:
    print(f"⚠️ Trained weights not found at {MODEL_PATH}. Model will be randomly initialized")

_base_model = _base_model.to(DEVICE)
_base_model.eval()

# Wrapper class for GradCAM compatibility
# HuggingFace models return SequenceClassifierOutput, but pytorch-grad-cam expects raw tensors
class HuggingFaceViTWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model
        # Expose internal structure for target layer access
        self.vit = hf_model.vit
        self.classifier = hf_model.classifier
    
    def forward(self, x):
        outputs = self.hf_model(x)
        # Return raw logits tensor instead of SequenceClassifierOutput
        return outputs.logits

model = HuggingFaceViTWrapper(_base_model)
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
    """Compute attention rollout from all layers.
    
    This computes how much attention flows from the CLS token to each image patch
    through all transformer layers.
    """
    result = torch.eye(attentions[0].size(-1))
    
    for attention in attentions:
        # Average attention heads
        attention_heads_fused = attention.mean(dim=1)
        
        # Add residual connection (identity)
        # This is important for proper attention flow computation
        I = torch.eye(attention_heads_fused.size(-1))
        attention_heads_fused = (attention_heads_fused + I) / 2
        
        # Normalize rows to sum to 1
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
        
        # Multiply with previous result
        result = torch.matmul(attention_heads_fused[0], result)
    
    # Look at attention FROM CLS token (row 0) to all patch tokens (columns 1:)
    # This tells us which patches the CLS token is paying attention to
    mask = result[0, 1:]
    
    return mask

def get_attention_based_saliency(hf_model, tensor, target_class, original_image):
    """Generate saliency map using ONLY attention rollout (best for ViT models).
    
    This produces clean, interpretable heatmaps that show where the model 
    is actually looking, without the noise from gradient-based methods.
    
    Note: This function requires the raw HuggingFace model (_base_model), not the wrapper.
    """
    
    # Forward pass with attention outputs using the HuggingFace model directly
    with torch.no_grad():
        outputs = hf_model(tensor, output_attentions=True)
    
    # Use attention rollout if available
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        attention_rollout = compute_rollout_attention(outputs.attentions)
        
        # Reshape attention rollout to 2D grid (14x14 for ViT-base with 224x224 input)
        grid_size = int(np.sqrt(attention_rollout.size(0)))
        attention_map = attention_rollout.reshape(grid_size, grid_size).cpu().numpy()
        
        # Normalize to [0, 1]
        attention_map = attention_map - attention_map.min()
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        
        # Enhance contrast to make high-attention areas stand out
        # Apply power transformation (gamma < 1 brightens, gamma > 1 darkens low values)
        attention_map = np.power(attention_map, 0.5)  # Square root to boost mid-values
        
        return attention_map
    else:
        # If no attention available, return a uniform map (shouldn't happen with ViT)
        return np.ones((14, 14), dtype=np.float32) * 0.5

def create_heatmap_from_attention(attention_map, original_image_shape, original_image):
    """Convert attention map to heatmap overlay (standard smooth heatmap)"""
    h, w = original_image_shape[:2]
    
    # Resize attention map to original image size
    heatmap = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
        
    # Smooth to make it look like a "real" heatmap
    kernel_size = max(15, min(h, w) // 15)
    if kernel_size % 2 == 0:
        kernel_size += 1
    heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
    
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.6):
    """Overlay a golden-red heatmap on `image` with transparency."""
    # Ensure image RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    # Normalize image to float [0,1]
    image_float = image_rgb.astype(np.float32) / 255.0

    # Resize heatmap to image size (just in case)
    h, w = image_rgb.shape[:2]
    heat_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize heatmap to 0-255
    heat_uint8 = (heat_resized * 255).astype(np.uint8)
    
    # Apply OpenCV AUTUMN colormap (Red to Yellow = Golden Red)
    heat_colored_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_AUTUMN)
    heat_colored = cv2.cvtColor(heat_colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Create an alpha mask based on heatmap intensity
    # Where heatmap is strong, use higher alpha; where weak, transparent
    heatmap_intensity = heat_resized[..., np.newaxis] # make it (H,W,1)
    
    # Blend: original * (1 - alpha*intensity) + heatmap * (alpha*intensity)
    overlay = image_float * (1.0 - alpha * heatmap_intensity) + heat_colored * (alpha * heatmap_intensity)
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
            # model now returns raw logits (via HuggingFaceViTWrapper)
            logits = model(tensor_no_grad)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_id].item()
        
        # Get class name
        predicted_class = CLASS_NAMES[predicted_class_id]
        
        # Create all probabilities dict
        all_probs = {
            CLASS_NAMES[i]: float(probabilities[0][i].cpu().numpy())
            for i in range(len(CLASS_NAMES))
        }
        
        # Generate heatmap. Prefer pytorch-grad-cam if available
        overlay_img = None
        if _HAS_GRADCAM:
            try:
                # Prepare RGB image in [0,1]
                if len(original_image.shape) == 2:
                    rgb_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_img = original_image.copy()
                
                # RESIZE TO 224x224 is CRITICAL for ViT
                rgb_img = cv2.resize(rgb_img, (224, 224))
                
                rgb_float = (rgb_img.astype(np.float32) / 255.0)

                # Build input tensor for grad-cam
                input_tensor_cam = cam_preprocess_image(rgb_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                # Heuristic to find a suitable target layer for ViT
                def find_vit_target_layer(m):
                    # Try targeting the LayerNorm AFTER the last attention block (output of the block)
                    # This often gives better results for ViT
                    try:
                        return m.vit.encoder.layer[-1].layernorm_after
                    except AttributeError:
                        pass
                    # Fallback to layernorm_before
                    try:
                        return m.vit.encoder.layer[-1].layernorm_before
                    except AttributeError:
                        pass
                    # Try last encoder layer directly
                    try:
                        return m.vit.encoder.layer[-1]
                    except AttributeError:
                        pass
                    return None

                target_layer = find_vit_target_layer(model)
                
                # Define reshape transform for ViT
                def vit_reshape_transform(tensor, height=14, width=14):
                    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                                    height, width, tensor.size(2))
                    # Bring the channels to the first dimension, like in CNNs.
                    result = result.transpose(2, 3).transpose(1, 2)
                    return result

                # If we found a target layer
                if target_layer is not None:
                    # Use EigenCAM instead of GradCAM for better ViT compatibility
                    # EigenCAM doesn't rely on gradients and often produces cleaner results for transformers
                    cam = EigenCAM(model=model, target_layers=[target_layer], reshape_transform=vit_reshape_transform)

                    targets = [ClassifierOutputTarget(predicted_class_id)]
                    grayscale_cam = cam(input_tensor=input_tensor_cam.to(DEVICE), targets=targets)
                    grayscale_cam = grayscale_cam[0]

                    # Create overlay using the library's function
                    # rgb_float is already resized to 224x224 and is [0..1]
                    # COLORMAP_JET: Blue->Cyan->Green->Yellow->Red (standard scientific heatmap)
                    # image_weight=0.3 means 30% original image, 70% heatmap (heatmap dominant)
                    overlay_img = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.3)
                else:
                    print("Could not find suitable target layer for GradCAM")
                    overlay_img = None
            except Exception as e:
                print(f"Grad-CAM failed: {e}")
                import traceback
                traceback.print_exc()
                overlay_img = None

        if overlay_img is None:
            # Fallback to attention-based rollout implementation (cleaner for ViT)
            try:
                # No need for model.train() since we use torch.no_grad() in get_attention_based_saliency
                tensor_input = tensor.unsqueeze(0).to(DEVICE)
                attention_map = get_attention_based_saliency(_base_model, tensor_input, predicted_class_id, original_image)
                
                heatmap = create_heatmap_from_attention(attention_map, original_image.shape, original_image)
                # Use simple overlay for fallback
                overlay_img = overlay_heatmap(original_image, heatmap)
            except Exception as ex:
                print(f"Fallback heatmap generation failed: {ex}")
                import traceback
                traceback.print_exc()
                # Return original image if everything fails
                if len(original_image.shape) == 2:
                    overlay_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                else:
                    overlay_img = original_image
        
        # Convert heatmap to base64
        # cv2.imencode expects BGR. 
        # overlay_img from show_cam_on_image(use_rgb=True) is RGB.
        # So we MUST convert RGB -> BGR.
        overlay_img_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', overlay_img_bgr)
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
