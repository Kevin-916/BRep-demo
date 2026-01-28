import torch
import inspect
# import omegaconf


def inverse_sigmoid(x):
    """
    Inverse of the sigmoid function.
    Args:
        x (Tensor): Input tensor.
    Returns:
        Tensor: Inverse sigmoid of the input tensor.
    """
    return torch.log(x / (1 - x))
