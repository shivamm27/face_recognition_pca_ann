import matplotlib.pyplot as plt
import numpy as np
import os

def show_image(vector, shape=(100,100), title="Image", save_path=None):
    os.makedirs("results", exist_ok=True)
    plt.imshow(vector.reshape(shape), cmap="gray")
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")
        plt.close()
    else:
        plt.show()

def show_eigenfaces(eigenfaces, num=10, shape=(100,100), save_path=None):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(12,6))
    for i in range(num):
        plt.subplot(2, num//2, i+1)
        plt.imshow(eigenfaces[:,i].reshape(shape), cmap="gray")
        plt.title(f"Eigenface {i+1}")
        plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")
        plt.close()
    else:
        plt.show()
