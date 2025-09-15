import os
import cv2
import numpy as np

def load_images_from_folder(data_path, img_size=(100, 100)):
    X = []
    y = []
    persons = []

    # Collect class (person) folders
    persons = sorted([p for p in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, p))])

    label = 0
    for person in persons:
        person_folder = os.path.join(data_path, person)

        for filename in os.listdir(person_folder):
            # Only process valid image extensions
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"⚠️ Skipping unreadable file: {img_path}")
                    continue

                img = cv2.resize(img, img_size).flatten()
                X.append(img)
                y.append(label)

        label += 1

    if len(X) == 0:
        raise ValueError("❌ No images loaded! Check dataset structure and file types.")

    X = np.array(X).T  # shape: (features, samples)
    y = np.array(y)
    return X, y, persons
 