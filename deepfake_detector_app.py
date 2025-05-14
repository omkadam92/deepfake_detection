import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os

# Set page config
st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🔍",
    layout="wide"
)

# Global variables
IMAGE_SIZE = (240, 240)
MODEL_PATH = 'deepfake_inference_model.keras'
THRESHOLD = 0.5

# Function to load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Function to preprocess image
def preprocess_image(image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to BGR if RGB (OpenCV uses BGR)
    if img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize to the required input dimensions
    img_resized = cv2.resize(img_array, IMAGE_SIZE)
    
    # Preprocess for EfficientNetB0
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_resized)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    return img_batch

# Function to make prediction
def predict_deepfake(model, img_batch):
    prediction = model.predict(img_batch, verbose=0)
    probability = prediction[0][0]
    is_fake = probability > THRESHOLD
    return is_fake, probability

# Main application
def main():
    # Title and Introduction
    st.title("Deepfake Image Detector")
    st.markdown("""
    This application uses a deep learning model based on EfficientNetB0 to detect whether an image is real or a deepfake.
    Upload an image and the model will analyze it for you.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    This app uses a fine-tuned EfficientNetB0 model to detect manipulated images (deepfakes).
    The model was trained on a dataset of real and fake images and achieved high accuracy in distinguishing between them.
    """)
    
    # Load model
    model = load_model()
    
    # File uploader - centered and prominent
    st.markdown("### Upload an image to test")
    st.markdown("Select a JPG, JPEG, or PNG image to analyze")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    # Process uploaded image
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Your Image", use_column_width=True)
        
        # Process the image
        img_batch = preprocess_image(image)
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            is_fake, probability = predict_deepfake(model, img_batch)
        
        with col2:
            st.subheader("Prediction Result")
            
            # Display prediction
            result_placeholder = st.empty()
            
            if is_fake:
                result_placeholder.error(f"FAKE IMAGE DETECTED (Confidence: {probability:.2f})")
            else:
                result_placeholder.success(f"REAL IMAGE (Confidence: {1-probability:.2f})")
            
            # Display confidence meter
            st.write("Deepfake Probability:")
            st.progress(float(probability))
            
            # Display confidence values
            col_real, col_fake = st.columns(2)
            with col_real:
                st.metric(label="Real", value=f"{(1-probability)*100:.1f}%")
            with col_fake:
                st.metric(label="Fake", value=f"{probability*100:.1f}%")
            
    else:
        # Show instructions when no image is uploaded
        st.info("👆 Please upload an image to test the deepfake detector")
        
    
    # Information about the model
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown("""
    - **Architecture**: EfficientNetB0
    - **Training**: Fine-tuned on a dataset of real and fake images
    """)

if __name__ == "__main__":
    main() 