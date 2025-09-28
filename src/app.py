# app.py
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf

# -------------------------
# Load PCA + ANN + Label Encoder
# -------------------------
@st.cache_resource
def load_models():
    try:
        pca = joblib.load("results/pca_transform.pkl")   # PCA transformer
        model = tf.keras.models.load_model("results/face_ann_model.h5")  # ANN model
        label_encoder = joblib.load("results/label_encoder.pkl")  # Label encoder
        return pca, model, label_encoder
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None, None, None

pca, ann_model, label_encoder = load_models()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Face Recognition PCA+ANN", page_icon="👤")
st.title("👤 Face Recognition (PCA + ANN)")
st.write("Upload an image to predict the person using the trained PCA + ANN model.")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and pca and ann_model:
    # Convert uploaded image
    image = Image.open(uploaded_file).convert("L").resize((100, 100))  # grayscale 100x100
    img_array = np.array(image).flatten().reshape(1, -1)

    # Apply PCA
    try:
        features = pca.transform(img_array)
    except Exception as e:
        st.error(f"⚠️ PCA transform failed: {e}")
        st.stop()

    # ANN Prediction
    try:
        prediction = ann_model.predict(features)
        pred_class = np.argmax(prediction, axis=1)[0]
        pred_name = label_encoder.inverse_transform([pred_class])[0]
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.stop()


# save_models.py
import joblib
import tensorflow as tf

# Assuming you already have these objects after training:
# pca (the fitted PCA transformer)
# label_encoder (the fitted LabelEncoder)
# ann_model (the trained ANN model)

# Save PCA transformer
joblib.dump(pca, "results/pca_transform.pkl")
print("✅ Saved PCA transformer at results/pca_transform.pkl")

# Save Label Encoder
joblib.dump(label_encoder, "results/label_encoder.pkl")
print("✅ Saved Label Encoder at results/label_encoder.pkl")

# Save ANN model
ann_model.save("results/face_ann_model.h5")
print("✅ Saved ANN model at results/face_ann_model.h5")


    # Display result
    st.image(image, caption=f"Prediction: {pred_name}", use_column_width=True)
    st.success(f"✅ Predicted Person: {pred_name}")

elif uploaded_file is not None:
    st.warning("⚠️ Models not loaded. Please check your 'results/' folder for PCA, ANN, and label encoder files.")

