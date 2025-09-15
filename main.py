from src.preprocess import load_images_from_folder
from src.pca import compute_pca
from src.visualize import show_image, show_eigenfaces
from src.ann import train_ann
import os
import numpy as np

# ----------------- PATHS -----------------
# Using your current structure (train/ and train/test/)
data_path = "train"        # Training dataset
test_path = "test"   # Testing dataset

# ----------------- LOAD DATASET -----------------
X, y, persons = load_images_from_folder(data_path, img_size=(100, 100))

print("\n✅ Preprocessing Complete!")
print("Face_Db shape:", X.shape)
print("Labels shape:", y.shape)
print("Persons:", persons)

# ----------------- PCA -----------------
k = 30  # number of components
mean_face, eigenfaces, projected = compute_pca(X, k)

print("\n✅ PCA Complete!")
print("Mean face shape:", mean_face.shape)
print("Eigenfaces shape:", eigenfaces.shape)
print("Projected data shape:", projected.shape)

# ----------------- VISUALIZATION -----------------
os.makedirs("results", exist_ok=True)
show_image(mean_face, title="Mean Face", save_path="results/mean_face.png")
show_eigenfaces(eigenfaces, num=10, save_path="results/eigenfaces.png")

# ----------------- TRAIN ANN -----------------
print("\n🚀 Training ANN Classifier...")
# Train ANN
model, history = train_ann(projected, y, num_classes=len(persons), epochs=30)

# Save the model
model.save("results/face_ann_model.h5")
print("💾 ANN Model saved to results/face_ann_model.h5")

# ----------------- TESTING -----------------
if os.path.exists(test_path):
    from keras.models import load_model
    import cv2

    print("\n🔍 Running predictions on test images...")
    loaded_model = load_model("results/face_ann_model.h5")

    for img_name in os.listdir(test_path):
        img_path = os.path.join(test_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (100, 100)).flatten().reshape(-1, 1)
        img_centered = img - mean_face

        # Project to PCA space
        img_proj = eigenfaces.T @ img_centered

        # Predict class
        pred = loaded_model.predict(img_proj.T)
        class_idx = np.argmax(pred)
        print(f"🖼️ Test Image: {img_name} → Predicted: {persons[class_idx]}")
