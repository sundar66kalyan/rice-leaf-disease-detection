import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(95deg, #2c6e3f, #488a31);
        color: white;
        font-weight: bold;
        border-radius: 30px;
        padding: 0.6rem;
    }
    .stButton > button:hover {
        background: linear-gradient(95deg, #1f582f, #3d7828);
        transform: translateY(-2px);
        transition: all 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

# Disease classes
classes = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Smut"]

# Model accuracy dictionary
MODEL_ACCURACY = {
    "MobileNetV2": 0.9167,
    "CNN": 0.80,
    "Ensemble Voting": 0.923
}

# File paths
MOBILENET_PATH = 'rice_mobilenet_model.h5'
CNN_PATH = 'cnn_rice_model.h5'

# Cache models for better performance
@st.cache_resource
def load_mobilenet_model():
    """Load MobileNetV2 model"""
    try:
        if os.path.exists(MOBILENET_PATH):
            model = tf.keras.models.load_model(MOBILENET_PATH)
            return model
        else:
            st.error(f"MobileNetV2 model not found at {MOBILENET_PATH}")
            return None
    except Exception as e:
        st.error(f"Failed to load MobileNetV2: {e}")
        return None

@st.cache_resource
def load_cnn_model():
    """Load CNN model"""
    try:
        if os.path.exists(CNN_PATH):
            model = tf.keras.models.load_model(CNN_PATH)
            return model
        else:
            st.warning(f"CNN model not found at {CNN_PATH}")
            return None
    except Exception as e:
        st.warning(f"Failed to load CNN: {e}")
        return None

# Load models
with st.spinner("Loading AI models... Please wait."):
    mobilenet_model = load_mobilenet_model()
    cnn_model = load_cnn_model()

# Show loaded models status
if mobilenet_model:
    st.success("✅ MobileNetV2 model loaded (91.67% accuracy)")
if cnn_model:
    st.success("✅ CNN model loaded (80% accuracy)")

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for model prediction"""
    img = np.array(image)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_with_mobilenet(image):
    """Predict using MobileNetV2"""
    if mobilenet_model is None:
        return None, None
    processed = preprocess_image(image, (128, 128))
    prediction = mobilenet_model.predict(processed)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return classes[predicted_class], confidence

def predict_with_cnn(image):
    """Predict using CNN model"""
    if cnn_model is None:
        return None, None
    processed = preprocess_image(image, (128, 128))
    prediction = cnn_model.predict(processed)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return classes[predicted_class], confidence

def predict_ensemble(image):
    """Ensemble prediction combining MobileNet and CNN"""
    predictions = []
    confidences = []
    
    # Get MobileNet prediction
    if mobilenet_model:
        pred1, conf1 = predict_with_mobilenet(image)
        if pred1:
            predictions.append(pred1)
            confidences.append(conf1)
    
    # Get CNN prediction
    if cnn_model:
        pred2, conf2 = predict_with_cnn(image)
        if pred2:
            predictions.append(pred2)
            confidences.append(conf2)
    
    # If no models available
    if len(predictions) == 0:
        return "Unable to predict", 0
    
    # Weighted voting
    class_scores = {c: 0 for c in classes}
    for i, pred in enumerate(predictions):
        class_scores[pred] += confidences[i] / 100
    
    final_class = max(class_scores, key=class_scores.get)
    final_confidence = (class_scores[final_class] * 100) / len(predictions)
    
    return final_class, final_confidence

# Main UI
st.title("🌾 Rice Leaf Disease Detection")
st.markdown("*Powered by AI | Created by Kalyana Sundar - AI Engineer*")

# Sidebar for model selection
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    model_options = ["MobileNetV2", "CNN", "Ensemble Voting"]
    selected_model = st.selectbox("Select Model", model_options)
    
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    # Display accuracy table
    acc_data = []
    for model, acc in MODEL_ACCURACY.items():
        acc_data.append({"Model": model, "Accuracy": f"{acc*100:.1f}%"})
    
    st.dataframe(acc_data, use_container_width=True)
    
    st.markdown("---")
    st.caption("🏆 Best Model: MobileNetV2 with 91.67% accuracy")
    st.caption("👨‍💻 Created by Kalyana Sundar - AI Engineer")

# Main content area - 2 columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Rice Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "png", "jpeg"],
        help="Upload a clear image of a rice leaf for disease detection"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🔬 Diagnosis Result")
    
    if uploaded_file is not None:
        if mobilenet_model is None and cnn_model is None:
            st.error("No models loaded. Please check the model files.")
        else:
            with st.spinner("Analyzing with AI models..."):
                # Make prediction based on selected model
                if selected_model == "MobileNetV2":
                    disease, confidence = predict_with_mobilenet(image)
                    if disease is None:
                        st.error("MobileNetV2 model not available")
                        st.stop()
                elif selected_model == "CNN":
                    disease, confidence = predict_with_cnn(image)
                    if disease is None:
                        st.error("CNN model not available")
                        st.stop()
                else:  # Ensemble Voting
                    disease, confidence = predict_ensemble(image)
                
                # Display results with metrics
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f5f7f0, #ffffff); padding: 1.5rem; border-radius: 15px; text-align: center;">
                    <h2 style="color: #2c6e3f;">🌿 {disease}</h2>
                    <div style="font-size: 3rem; font-weight: bold; margin: 1rem 0;">
                        {confidence:.1f}%
                    </div>
                    <div style="background: #e8f0da; border-radius: 10px; height: 10px; margin: 1rem 0;">
                        <div style="background: linear-gradient(90deg, #2c6e3f, #6baf46); width: {confidence}%; height: 10px; border-radius: 10px;"></div>
                    </div>
                    <p style="color: #666;">Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Treatment recommendation
                st.markdown("---")
                st.subheader("💡 Recommended Action")
                recommendations = {
                    "Bacterial Leaf Blight": "Apply copper-based bactericide, remove infected plants, improve drainage",
                    "Brown Spot": "Use fungicide, maintain proper nutrition, avoid water stress",
                    "Leaf Smut": "Remove infected leaves, apply azoxystrobin, practice crop rotation"
                }
                st.info(recommendations.get(disease, "Consult local agricultural expert"))
                
                # Model info
                st.caption(f"Model used: {selected_model} | Accuracy: {MODEL_ACCURACY.get(selected_model, 0.70)*100:.1f}%")
    else:
        st.info("👈 Please upload an image to start diagnosis")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 AI-Powered Rice Disease Detection | MobileNetV2 + CNN Ensemble</p>
    <p>© 2026 Kalyana Sundar - AI Engineer | Rice Leaf Disease Detection System</p>
</div>
""", unsafe_allow_html=True)