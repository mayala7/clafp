import torch
import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np
from sklearn.impute import SimpleImputer

def apply_pca_to_numpy_array(array, n_components=768):
    # Impute NaN values
    imputer = SimpleImputer(strategy='mean')
    array_imputed = imputer.fit_transform(array)

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(array_imputed)

    return reduced_data

def apply_pca_all_tensors(tf_tensors, n_components=768):
    pca_results = []
    for tf_tensor in tf_tensors:
        numpy_array = tf_tensor.numpy()  # Convert TensorFlow tensor to NumPy
        reduced_data = apply_pca_to_numpy_array(numpy_array, n_components=n_components)
        torch_tensor = torch.from_numpy(reduced_data).float()  # Convert NumPy array to PyTorch tensor
        pca_results.append(torch_tensor)
    return pca_results


