import os
import json
import base64
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="AgroScan AI - Plant Disease Classifier 🌱",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Set background image
bg_image_path = os.path.join(os.path.dirname(__file__), "adrian-regeci-IBoi9f7BdJI-unsplash.jpg")
with open(bg_image_path, "rb") as file:
    encoded = base64.b64encode(file.read()).decode()

st.markdown(
    f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: #ffffff;
        }}
        .stApp {{
            background-color: rgba(0, 0, 0, 0.6);
        }}
        .title {{
            font-size: 42px;
            color: #a8e6cf;
            text-align: center;
            font-weight: bold;
            margin-top: 20px;
        }}
        .subtitle {{
            font-size: 20px;
            color: #ffffff;
            text-align: center;
            margin-bottom: 30px;
        }}
        .stButton > button {{
            background-color: #43a047;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
        }}
        .stButton > button:hover {{
            background-color: #388e3c;
        }}
        .block-text {{
            color: #ffffff;
            font-size: 18px;
        }}
        .result {{
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
        }}
        a {{
            color: #ffffff !important;  /* Force white color */
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and branding
st.markdown('<div class="title">🌿 AgroScan AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Plant Disease Detection & Analysis</div>', unsafe_allow_html=True)

# Load model and class indices
working_dir = os.path.dirname(__file__)
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

model = tf.keras.models.load_model(model_path)

with open(class_indices_path) as f:
    class_indices = json.load(f)

# Ensure keys are integers
inv_class_indices = {int(k): v for k, v in class_indices.items()}

# Preprocessing function
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0
    return img_array

# Prediction function
def predict_image_class(model, image_path, inv_class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_index = int(np.argmax(predictions, axis=1)[0])
    predicted_class = inv_class_indices[predicted_index]
    confidence = float(np.max(predictions))
    return predicted_class, confidence

# Upload section
st.markdown('<p style="color:#8BC34A; font-size:18px; font-weight:bold;">📤 Upload a plant leaf image</p>', unsafe_allow_html=True)
uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    col1, col2 = st.columns([1, 1.5], gap="medium")

    with col1:
        st.markdown('<p class="block-text">🖼 Preview:</p>', unsafe_allow_html=True)
        st.image(image.resize((224, 224)), caption="Uploaded Leaf", use_container_width=True)

    with col2:
        st.markdown('<p class="block-text">🧪 Diagnosis:</p>', unsafe_allow_html=True)
        if st.button("🔍 Classify"):
            with st.spinner("🔬 Analyzing leaf condition..."):
                prediction, confidence = predict_image_class(model, uploaded_image, inv_class_indices)

            # Google search link
            search_query = prediction.replace("_", " ").replace(" ", "+") + "+plant+disease"
            google_url = f"https://www.google.com/search?q={search_query}"

            st.markdown(
                f'<p class="result">🌱 <strong>Detected Condition:</strong> '
                f'<a href="{google_url}" target="_blank">{prediction}</a></p>',
                unsafe_allow_html=True
            )
            st.markdown(f'<p class="result">📊 <strong>Accuracy:</strong> {confidence:.2%}</p>', unsafe_allow_html=True)

else:
    st.markdown('<p style="text-align:center; color:#ffffff; margin-top:30px;">👈 Upload a high-quality image of a plant leaf to begin diagnosis.</p>', unsafe_allow_html=True)
