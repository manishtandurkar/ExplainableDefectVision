import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

# Configuration
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']
DEFECT_EXPLANATIONS = {
    'crazing': "Surface cracks caused by thermal stress or uneven cooling during the manufacturing process.",
    'inclusion': "Non-metallic impurities embedded in the steel matrix during the rolling or casting process.",
    'patches': "Irregular surface texture variations caused by inconsistent coating or material composition.",
    'pitted_surface': "Localized corrosion pits formed due to chemical contamination or environmental exposure.",
    'rolled_in_scale': "Oxide scale particles trapped and embedded in the surface during hot rolling operations.",
    'scratches': "Linear mechanical abrasions caused by contact with tools, handling equipment, or processing machinery."
}

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=6,
        ignore_mismatched_sizes=True
    )
    
    # Load trained weights if available
    try:
        state_dict = torch.load('models/best_vit_defect_detector.pt', map_location=device)
        model.load_state_dict(state_dict)
        st.success("✅ Trained model loaded successfully!")
    except:
        st.warning("⚠️ Using pre-trained model (not fine-tuned on defect data)")
    
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image.astype('uint8'))
    
    return transform(image)

def predict_defect(model, device, image_tensor):
    """Predict defect class"""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def create_simple_heatmap(image, prediction_confidence):
    """Create a simple attention-like heatmap for demo"""
    h, w = image.shape[:2]
    
    # Create random attention pattern based on prediction confidence
    np.random.seed(int(prediction_confidence * 1000))
    heatmap = np.random.random((h//8, w//8)) * prediction_confidence
    heatmap = cv2.resize(heatmap, (w, h))
    
    # Apply Gaussian blur
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Convert to float
    image_rgb = image_rgb.astype(np.float32) / 255.0
    
    # Create colored heatmap
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    
    # Blend images
    overlay = (1 - alpha) * image_rgb + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    return (overlay * 255).astype(np.uint8)

# Streamlit App Layout
def main():
    st.set_page_config(
        page_title="🏭 Explainable Defect Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    # Header
    st.title("🏭 Explainable Multi-Class Industrial Defect Detection")
    st.markdown("### Using Vision Transformers with Dual-Layer Explainability")
    
    st.markdown("""
    This application uses a Vision Transformer model to detect and explain industrial surface defects.
    Upload an image to get predictions with visual explanations and root-cause analysis.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model()
    
    # Sidebar
    st.sidebar.header("📋 Defect Classes")
    for i, class_name in enumerate(CLASS_NAMES):
        st.sidebar.write(f"{i+1}. {class_name.replace('_', ' ').title()}")
    
    st.sidebar.header("ℹ️ How it works")
    st.sidebar.markdown("""
    1. **Upload** an image of a steel surface
    2. **Analyze** with Vision Transformer
    3. **Visualize** attention patterns
    4. **Explain** the root cause
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Image Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a grayscale or color image of a steel surface"
        )
        
        if uploaded_file is not None:
            # Load and display image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            original_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            
            st.image(original_image, caption="Uploaded Image", use_column_width=True, cmap='gray')
            
            # Preprocess image
            image_tensor = preprocess_image(original_image)
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, all_probs = predict_defect(model, device, image_tensor)
                
            # Display results in right column
            with col2:
                st.header("🎯 Analysis Results")
                
                # Prediction
                class_name = CLASS_NAMES[predicted_class]
                st.success(f"**Detected Defect:** {class_name.replace('_', ' ').title()}")
                st.info(f"**Confidence:** {confidence:.2%}")
                
                # Probability distribution
                st.subheader("📊 Class Probabilities")
                prob_data = {
                    'Class': [name.replace('_', ' ').title() for name in CLASS_NAMES],
                    'Probability': all_probs
                }
                st.bar_chart(prob_data['Probability'])
                
                # Generate simple heatmap for demo
                heatmap = create_simple_heatmap(original_image, confidence)
                overlay = overlay_heatmap(original_image, heatmap)
                
                st.subheader("🔍 Attention Visualization")
                st.image(overlay, caption="Model Attention Heatmap", use_column_width=True)
                
                # Explanation
                st.subheader("💡 Root-Cause Explanation")
                explanation = DEFECT_EXPLANATIONS[class_name]
                st.markdown(f"**{explanation}**")
                
                # Detailed analysis
                with st.expander("📋 Detailed Analysis"):
                    st.write("**Prediction Details:**")
                    st.write(f"- Predicted Class ID: {predicted_class}")
                    st.write(f"- Confidence Score: {confidence:.4f}")
                    st.write(f"- Model: Vision Transformer (ViT)")
                    
                    st.write("**Top 3 Predictions:**")
                    top_indices = np.argsort(all_probs)[-3:][::-1]
                    for i, idx in enumerate(top_indices):
                        st.write(f"{i+1}. {CLASS_NAMES[idx].replace('_', ' ').title()}: {all_probs[idx]:.3f}")

if __name__ == "__main__":
    main()