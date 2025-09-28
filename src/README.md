# 👤 Face Recognition using PCA + ANN

This project implements a **Face Recognition system** using **Principal Component Analysis (PCA)** for feature extraction (Eigenfaces) and an **Artificial Neural Network (ANN)** for classification.

## 🚀 Features

* Preprocessing: Resize → Grayscale → Flatten images (100×100 pixels)
* PCA: Extract eigenfaces, reduce from 10,000 → ~50 features
* ANN: Dense + Softmax classifier trained on PCA features
* Model saved in `.h5` format
* Streamlit web app for uploading and predicting faces

## 📂 Project Structure

```
face_recognition_pca_ann/
│── data/              # Training/test images (not uploaded to repo if large)
│── results/           # Saved outputs & models
│   ├── face_ann_model.h5
│   ├── pca_transform.pkl
│   ├── label_encoder.pkl
│── src/               # Helper scripts (preprocess, pca, ann, predict)
│── app.py             # Streamlit app (entrypoint)
│── requirements.txt   # Dependencies
│── README.md          # Project description
│── .gitignore         # Ignored files
```

## ▶️ Run Locally

1. Clone the repo:

   ```bash
   git clone https://github.com/<your-username>/face-recognition-pca-ann.git
   cd face-recognition-pca-ann
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Create a new app → Select this repo → Choose `app.py` as entrypoint.
4. Share your link:

   ```
   https://<your-username>-face-recognition.streamlit.app
   ```

## 📊 Results

* Accuracy: ~50% (due to small dataset & ANN limitations)
* Saved mean face and eigenfaces
* Example prediction: `face_3.jpg → Predicted: Disha`

## 🔮 Future Improvements

* Use CNNs or transfer learning (e.g., VGGFace)
* Data augmentation
* Larger datasets
* Real-time webcam-based face recognition
