import numpy as np

def compute_pca(X, k):
    mean_face = np.mean(X, axis=1).reshape(-1, 1)  # (mn,1)
    A = X - mean_face  # mean-centered
    C = np.dot(A.T, A)  # surrogate covariance
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, idx[:k]]
    eigenfaces = np.dot(A, eigenvectors)
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)
    projected = np.dot(eigenfaces.T, A)
    return mean_face, eigenfaces, projected
