import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Configure GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        background-color: #f0f8ff;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🖼️ CIFAR-10 Image Classifier</h1>', unsafe_allow_html=True)

# CIFAR-10 class labels (fixed the labels from notebook.ipynb)
LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache(allow_output_mutation=True)
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('image_classification.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure 'image_classification.h5' file is in the same directory as this app.")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Resize to 32x32 (CIFAR-10 input size)
    img_resized = cv2.resize(img_array, (32, 32))

    # Ensure 3 channels (RGB)
    if len(img_resized.shape) == 2:  # Grayscale
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[2] == 4:  # RGBA
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)

    # Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch

def predict_image(model, image):
    """Make prediction on the image"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence, predictions[0]

# Load model
model = load_model()

# Main app interface
if model is not None:
    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📁 Upload Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to classify. Works best with images containing objects from CIFAR-10 classes."
        )

        # Sample images section
        st.subheader("📷 Sample Options")
        sample_option = st.selectbox(
            "Select a sample image:",
            ["None", "Upload your own image above"]
        )

        # Display CIFAR-10 classes info
        st.subheader("🏷️ CIFAR-10 Classes")
        st.write("This model can classify images into these 10 categories:")

        # Display classes in a simple list (no nested columns)
        for i, label in enumerate(LABELS):
            st.write(f"• {label.capitalize()}")

    with col2:
        st.header("🔮 Prediction Results")

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)

            # Show original image
            st.subheader("Original Image")
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

            # Show processed image (32x32)
            processed_img = preprocess_image(image)
            st.subheader("Processed Image (32x32)")
            st.image(processed_img[0], caption="Resized for model input", width=150)

            # Make prediction
            try:
                predicted_class, confidence, all_predictions = predict_image(model, image)
                predicted_label = LABELS[predicted_class]

                # Display main prediction
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<h2>Predicted Class: {predicted_label.upper()}</h2>'
                    f'<h3>Confidence: {confidence:.2%}</h3>'
                    f'</div>', 
                    unsafe_allow_html=True
                )

                # Display all class probabilities
                st.subheader("📊 All Class Probabilities")

                # Create a DataFrame for better display
                import pandas as pd
                prob_df = pd.DataFrame({
                    'Class': LABELS,
                    'Probability': all_predictions,
                    'Percentage': [f"{prob:.1%}" for prob in all_predictions]
                }).sort_values('Probability', ascending=False)

                # Display as bars
                for idx, row in prob_df.iterrows():
                    st.write(f"**{row['Class'].capitalize()}**: {row['Percentage']}")
                    st.progress(float(row['Probability']))

                # Create a bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(LABELS, all_predictions)
                ax.set_xlabel('Classes')
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Probabilities for All Classes')
                ax.set_ylim(0, 1)

                # Highlight the predicted class
                bars[predicted_class].set_color('red')

                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            st.info("👆 Please upload an image to see the prediction results!")

            # Show model information
            st.subheader("🤖 Model Information")
            st.write("**Model Architecture**: CNN with Batch Normalization")
            st.write("**Dataset**: CIFAR-10")
            st.write("**Input Size**: 32x32x3 (RGB)")
            st.write("**Output Classes**: 10")
            st.write("**Framework**: TensorFlow/Keras")

else:
    st.error("❌ Could not load the model. Please check if 'image_classification.h5' exists.")
    st.info("Make sure you have:")
    st.write("1. Trained the model using the notebook")
    st.write("2. Saved the model as 'image_classification.h5'")
    st.write("3. Placed the model file in the same directory as this app")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit and TensorFlow"
    "</div>", 
    unsafe_allow_html=True
)
