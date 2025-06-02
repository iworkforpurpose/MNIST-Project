import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MNISTLoader:
    def __init__(self, filepath, val_size=0.2):
        self.filepath = filepath
        self.val_size = val_size
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
    
    def load_data(self):
        """ 
        Load MNIST data from CSV and preprocess it
        """
        print('Loading MNIST data from CSV...')
        
        #Reload the CSV file
        df = pd.read_csv(self.filepath)
        print(df.head())
        
        #Separate features and labels
        X = df.drop('label', axis=1).values
        y = df['label'].values
        
        #Normalize pixel values to range [0,1]
        X = X.astype('float32') / 255.0 
        
        #Reshape images to 28x28
        X = X.reshape(-1, 28, 28)
        
        #Convert labels to one-hot encoding
        y = self._one_hot_encode(y)
        
        #Split data into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=self.val_size, random_state=42)
        
        print(f"data loaded and split:")
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Validation samples: {self.X_val.shape[0]}")
        
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def _one_hot_encode(self, y):
        n_classes = len(np.unique(y))
        one_hot = np.zeros((len(y), n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    def get_batch(self, batch_size, train=True):
        """
        Generate a random batch of data

        Args:
            batch_size (int): Size of the batch
            train (bool): Whether to get batch from training or validation set

        Returns:
            tuple: Batch of images and corresponding labels
        """
        if train:
            X, y = self.X_train, self.y_train
        else:
            X, y = self.X_val, self.y_val
            
        indices = np.random.randint(0, X.shape[0], batch_size)
        return X[indices], y[indices]
    
    def preprocess_single_image(self, image):
        """
        Preprocess a single image for prediction

        Args:
            img (np.array): Image to preprocess

        Returns:
            np.ndarray: Preprocessed image
        """
        # Ensure correct shape
        if image.shape != (28, 28):
            raise ValueError("Image must be (28, 28)")
        
        #Normalize
        else:
            image = image.astype('float32') / 255.0
            
        return image 

#Example usage
if __name__ == "__main__":
    # Initialize the loader
    loader = MNISTLoader("/Users/vighneshnama/Documents/SDT AIML/3rd Year/6th sem/Deep neural networks/MNIST Project/mnist_train.csv")
    # Load and split the data
    X_train, X_val, y_train, y_val = loader.load_data()
        
    # Get a random batch
    batch_x, batch_y = loader.get_batch(batch_size=32)
        
    #print shapes
    print("\nShapes:")
    print(f"Training data: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Batch data: {batch_x.shape}")
    print(f"Batch labels: {batch_y.shape}")