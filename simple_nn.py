import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv('mnist_train.csv', header=None)
X = df.iloc[:, 1:].values / 255.0  # Normalize pixel values
y = to_categorical(df.iloc[:, 0].values)  # One-hot encode labels

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create model with different activation functions
def create_model(activation):
    model = Sequential([
        Dense(128, activation=activation, input_shape=(784,)),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# List of activation functions to compare
activations = ['relu', 'sigmoid', 'tanh']
histories = []

# Train models with different activations
for activation in activations:
    print(f"\nTraining model with {activation} activation...")
    model = create_model(activation)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20,
                        batch_size=32,
                        verbose=0)
    histories.append(history)
    print(f"{activation} - Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"{activation} - Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

# Visualization
plt.figure(figsize=(14, 6))

# Training Loss
plt.subplot(1, 2, 1)
for i, activation in enumerate(activations):
    plt.plot(histories[i].history['loss'], label=f'{activation}')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Validation Loss
plt.subplot(1, 2, 2)
for i, activation in enumerate(activations):
    plt.plot(histories[i].history['val_loss'], label=f'{activation}')
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()