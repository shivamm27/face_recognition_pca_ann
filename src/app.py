
# app.py
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf

# -------------------------
# Load PCA transformer + ANN model
# -------------------------
@st.cache_resource
def load_models():
    pca = joblib.load("results/pca_transform.pkl")   # adjust path if needed
    model = tf.keras.models.load_model("results/face_ann_model.h5")
    label_encoder = joblib.load("results/label_encoder.pkl")  # if you used one
    return pca, model, label_encoder

pca, ann_model, label_encoder = load_models()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Face Recognition PCA+ANN", page_icon="👤")
st.title("👤 Face Recognition (PCA + ANN)")
st.write("Upload an image to predict the person using PCA + ANN model.")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert image to array
    image = Image.open(uploaded_file).convert("L").resize((100, 100))  # grayscale, 100x100
    img_array = np.array(image).flatten().reshape(1, -1)

    # Apply PCA
    features = pca.transform(img_array)

    # Predict with ANN
    prediction = ann_model.predict(features)
    pred_class = np.argmax(prediction, axis=1)[0]

    # Decode label
    if label_encoder:
        pred_name = label_encoder.inverse_transform([pred_class])[0]
    else:
        pred_name = f"Person {pred_class}"

    st.image(image, caption=f"Prediction: {pred_name}", use_column_width=True)
    st.success(f"✅ Predicted Person: **{pred_name}**")

