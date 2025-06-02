import numpy

class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Calculate the accuracy of the model
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        Returns:
            float: Accuracy
        """
        pred_labels = numpy.argmax(y_pred, axis=1)
        true_labels = numpy.argmax(y_true, axis=1)
        return numpy.mean(pred_labels == true_labels)
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, num_classes=10):
        """
        Calculate the confusion matrix
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        Returns:
            np.ndarray: Confusion matrix
        """
        pred_labels = numpy.argmax(y_pred, axis=1)
        true_labels = numpy.argmax(y_true, axis=1)
        
        confusion_matrix = numpy.zeros((num_classes, num_classes))
        for t, p in zip(true_labels, pred_labels):
            confusion_matrix[t, p] += 1
        return confusion_matrix
    
class Loss:
    
    class MSE:
        """
        mean squared error
        """
        @staticmethod
        def forward(y_true, y_pred):
            """
            Calculate the mean squared error
            Args:
                y_true (np.ndarray): True labels
                y_pred (np.ndarray): Predicted labels
            Returns:
                float: Mean squared error
            """
            return numpy.mean(numpy.square(y_true - y_pred))
        
        @staticmethod
        def backward(y_true, y_pred):
            """
            Derivative of mean squared error
            Args:
                y_true (np.ndarray): True labels
                y_pred (np.ndarray): Predicted labels
            Returns:
                np.ndarray: Derivative of mean squared error
            """
            return 2 * (y_pred - y_true) / y_pred.shape[0]
        
    
    