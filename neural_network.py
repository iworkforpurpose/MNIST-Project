import numpy as np
import sys
sys.path.append('../')
from utils.activations import ReLU, Softmax
from utils.metrics import Loss

class Layer:
    def __init__(self, input_size, output_size, activation):
        # He initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros(1 / output_size)
        self.activation = activation
        
        # Caches values for backpropagation
        self.inputs = None
        self.output = None
        self.activation_output = None
        
        # Parameter Gradients
        self.weight_gradient = None
        self.bias_gradient = None
    
    def forward(self, x):
        """
        Forward pass through the layer
        Args:
            x (np.ndarray): Input array
        Returns:
            np.ndarray: layer Output
        """
        self.inputs = x
        self.activatiion_input = np.dot(x, self.weights) + self.bias
        self.output = self.activation.forward(self.activatiion_input)
        return self.output
    
    def backward(self, gradient):
        """
        Backward pass through the layer
        Args:
            gradient (np.ndarray): Gradient of the next layer
        Returns:
            np.ndarray: Gradient of previous layer
        """
        # Gradient through activation function
        activation_gradient = gradient * self.activation.backward(self.activation_input)
        
        # Calculate gradients for parameters
        self.weight_gradient = np.dot(self.inputs.T, activation_gradient)
        self.bias_gradient = np.sum(activation_gradient, axis=0, keepdims=True)
        
        # Calculate gradient for next layer
        return np.dot(activation_gradient, self.weights.T)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network

        Args:
            input_size (int): Number of input features
            hidden_size (int): number of neurons in the hidden layer
            output_size (int): Number of output classes
            learning_rate (float): Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        
        #initialize the layers
        self.hidden_layer = Layer(input_size, hidden_size, ReLU)
        self.output_layer = Layer(hidden_size, output_size, Softmax)
        
        #Initialize the loss function
        self.loss = Loss.MSE
        
        #training history
        self.loss_history = []
        self.accuracy_history = []
        
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x (np.ndarray): Input array
        
        Returns:
            np.ndarray: Output of the network
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        hidden_output = self.hidden_layer.forward(x)    
        return self.output_layer.forward(hidden_output)
    
    def backward(self, x, y):
        """
        Backward pass through the network
        Args:
            x (np.ndarray): Input array
            y (np.ndarray): True labels
        """
        # Get predictions
        predictions = self.forward(x)
        
        # Calculate initial fradient from loss funcion
        gradient = self.loss_function.backward(predictions, y)
        
        # Backpropagate through the output layer
        gradient = self.output_layer.backward(gradient)
        self.hidden_layer.backward(gradient)
        
        # Update parameters
        self.update_parameters()
        
    def update_parameters(self):
        """Update network parameters using gradient descent"""
        # Update output layer
        self.output_layer.weights -= self.learning_rate * self.output_layer.weight_gradient
        self.output_layer.bias -= self.learning_rate * self.output_layer.bias_gradient
        
        #update hidden layer
        self.hidden_layer.weights -= self.learning_rate * self.hidden_layer.weight_gradient
        self.hidden_layer.bias -= self.learning_rate * self.hidden_layer.bias_gradient
        
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """
        Train the neural network
        
        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation data
            y_val (np.ndarray): Validation labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
        """
        