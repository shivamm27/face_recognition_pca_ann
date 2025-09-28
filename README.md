# Face Recognition (PCA + ANN)

This project implements a face recognition system using **Principal Component Analysis (PCA)** for dimensionality reduction and an **Artificial Neural Network (ANN)** for classification.

## 🚀 Features

* Preprocessing: Resize, grayscale, flatten images (100x100)
* PCA: Extract eigenfaces, reduce to ~50 features
* ANN: Dense + Softmax layers, trained on PCA features
* Prediction: Upload a new face → PCA → ANN → Predicted Person

## 📂 Project Structure

```
face_recognition_pca_ann/
│── data/           # training/test images
│── results/        # mean_face, eigenfaces, saved models
│   ├── face_ann_model.h5
│   ├── pca_transform.pkl
│   ├── label_encoder.pkl
│── src/            # helper scripts (preprocess, pca, ann, predict)
│── app.py          # Streamlit app (entrypoint)
│── requirements.txt
│── README.md
```

## ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deploy

1. Push this repo to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Create a new app → Select this repo → Choose `app.py`.
4. Share your link: `https://your-username-face-recognition.streamlit.app`
