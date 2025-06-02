## MNIST Classification Project
## Project Overview

## Project Structure

## Features
- Neural Network Implementation from scratch using numpy
- Custom implementation of:
    - Forward propagation
    - Bacpropagation
    - Gradient descent optimizer
    - Activation functions (ReLu, Softmax)
- Data Preprocessing and normalization
- Training and evaluation pipeline
- Performance metrics calculation

## Dependencies
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- scikit-learn (for data splitting and metrics)

## Installation
1. Clone the repository
2. Create and activate a virtual enviroment (optimal but recommended)
3. Install dependencies

## Usage
1. Prepare the data
2. Train the model
3. Evaluate the model

## Neural Network Architecture
- Input layer: 784 neurons (28x28 pixels)
- Hidden layer: 128 neurons with ReLu activation
- Output layer: 10 neurons with Softmax activation
- Loss function: Cross-entropy loss
- Optimizer: Stochastic Gradient Descent

## Implementation Details
- The neural network is implemented using only NumPy for matrix operations
- Weights are initialized using the initialization
- Minin-batch gradient descent in used for optimization 
- Learning rate decay is implemented for better convergence

## Results 
- Training accuracy: [To be updated]
- Validation accuracy: [To be updated]
- Test Accuracy: [To be updated]

## Future Improvements
- Add momentum to gradient descent
- Implement dropout for regularization 
- Add batch normalization 
- Experiment with different architectures

## License

## Acknowledgements
- MNIST dataset.
