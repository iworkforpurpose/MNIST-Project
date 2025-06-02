import numpy as np

class ReLU:
    """
    Rectified Linear Unit activation function
    f(x) = max(0, x)
    """
    @staticmethod
    def forward(x):
        """
        Forward pass of ReLU
        Args: 
            x (np.ndarray): Input array
        Returns:
            np.ndarray: Output array
        """
        return np.maximum(0, x)
    def backward(x):
        """
        Derivative of ReLU
        Args: 
            x (np.ndarray): Input array
        Returns:
            np.ndarray: Derivative of ReLU
        """
        return np.where(x > 0, 1, 0)
    
class Sigmoid:
    """
    Sigmoid activation function
    f(x) = 1 / (1 + exp(-x))
    """
    @staticmethod
    def forward(x):
        """
        Forward pass of Sigmoid
        Args: 
            x (np.ndarray): Input array
        Returns:
            np.ndarray: Output array
        """
        return 1 / (1 + np.exp(-x))
    def backward(x):
        """
        Derivative of Sigmoid
        Args: 
            x (np.ndarray): Input array
        Returns:
            np.ndarray: Derivative of Sigmoid
        """
        return x * (1 - x)

class Softmax:
    """
    Softmax activation function
    f(x) = exp(x) / sum(exp(x))
    """
    @staticmethod
    def forward(x):
        """
        Forward pass of Softmax
        Args: 
            x (np.ndarray): Input array
        Returns:
            np.ndarray: Output array
        """
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)
    def backward(x):
        """
        Derivative of Softmax
        Args: 
            x (np.ndarray): Input array
        Returns:
            np.ndarray: Derivative of Softmax
        """
        return x * (1 - x)    
    
class Tanh:
    """
    Hyperbolic Tangent activation function
    f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    @staticmethod
    def forward(x):
        """
        Forward pass of Tanh
        Args: 
            x (np.ndarray): Input array
        Returns:
            np.ndarray: Output array
        """
        return np.tanh(x)
    def backward(x):
        """
        Derivative of Tanh
        Args: 
            x (np.ndarray): Input array
        Returns:
            np.ndarray: Derivative of Tanh
        """
        return 1 - x ** 2