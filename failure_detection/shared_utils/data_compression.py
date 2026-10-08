from sklearn.decomposition import PCA
import numpy as np
import torch


def perform_pca_tensor(tensor: torch.Tensor, n_components, return_pca=False):
    """
    Perform PCA on a 2D tensor and return the transformed tensor.

    Parameters:
    - tensor: 2D torch.Tensor of shape (n_samples, n_features)
    - n_components: Number of principal components to keep

    Returns:
    - transformed_tensor: 2D torch.Tensor of shape (n_samples, n_components)
    """
    assert tensor.ndim == 2, "Input tensor must be 2D"

    tensor_type = tensor.dtype
    device = tensor.device

    tensor = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

    pca = PCA(n_components=n_components)
    transformed_tensor = pca.fit_transform(tensor)

    transformed_tensor = torch.tensor(transformed_tensor, dtype=tensor_type).to(device)
    if return_pca:
        return transformed_tensor, pca
    return transformed_tensor


def project_vector_onto_pca_space(vector: torch.Tensor, pca: PCA):
    """
    Project a vector onto the PCA space defined by the PCA object.

    Parameters:
    - vector: 1D torch.Tensor of shape (n_features,)
    - pca: PCA object fitted on the data

    Returns:
    - projected_vector: 1D torch.Tensor of shape (n_components,)
    """
    assert vector.ndim == 1, "Input vector must be 1D"

    vector_type = vector.dtype
    device = vector.device

    vector = vector.cpu().numpy() if vector.is_cuda else vector.numpy()

    projected_vector = pca.transform(vector.reshape(1, -1))

    projected_vector = torch.tensor(projected_vector, dtype=vector_type).to(device)

    return projected_vector.squeeze(0)
