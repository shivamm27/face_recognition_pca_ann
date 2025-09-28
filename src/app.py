
# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import os

# -------------------------
# Load ANN model
# -------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("results/face_ann_model.h5")
    return model

ann_model = load_model()

# -------------------------
# Load training data again to fit PCA + LabelEncoder
# -------------------------
@st.cache_resource
def prepare_pca_and_labels():
    # ⚠️ You MUST have your training dataset available in the repo
    # Example: data/train/<person_name>/*.jpg
    import cv2
    import glob

    data_path = "data/train"
    X, y = [], []

    # Iterate over each person folder
    for person_folder in os.listdir(data_path):
        person_path = os.path.join(data_path, person_folder)
        if not os.path.isdir(person_path):
            continue
        for img_path in glob.glob(os.path.join(person_path, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (100, 100))
            X.append(img.flatten())
            y.append(person_folder)

    X = np.array(X)
    y = np.array(y)

    # PCA (keep 50 components like your project)
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)

    # Label Encoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return pca, le

pca, label_encoder = prepare_pca_and_labels()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Face Recognition PCA+ANN", page_icon="👤")
st.title("👤 Face Recognition (PCA + ANN)")
st.write("Upload an image to predict the person using the trained PCA + ANN model.")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess uploaded image
    image = Image.open(uploaded_file).convert("L").resize((100, 100))
    img_array = np.array(image).flatten()

