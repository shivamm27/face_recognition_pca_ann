# app.py (minimal)
import streamlit as st
from PIL import Image
import numpy as np
# load your model / encodings here
# e.g. using joblib / tensorflow / face_recognition

st.set_page_config("Face Recognition Demo")
st.title("Face recognition demo")

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_arr = np.array(image)
    # run your detection/recognition pipeline here, draw boxes / labels
    st.image(image, caption="Input image", use_column_width=True)
