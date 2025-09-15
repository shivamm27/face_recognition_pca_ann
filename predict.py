# src/predict.py (updated)
import cv2, numpy as np, joblib

def predict_face(img_path, mean_face, eigenfaces, model, persons, img_size=(100,100), scaler_path="results/scaler.save"):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.resize(img, img_size).flatten().reshape(-1,1)
    A_test = img - mean_face
    proj_test = np.dot(eigenfaces.T, A_test).ravel().reshape(1,-1)  # (1,k)

    # load and apply scaler if exists
    try:
        scaler = joblib.load(scaler_path)
        proj_test = scaler.transform(proj_test)
    except Exception:
        pass

    probs = model.predict(proj_test, verbose=0).ravel()
    idx = int(np.argmax(probs))
    return persons[idx], probs
