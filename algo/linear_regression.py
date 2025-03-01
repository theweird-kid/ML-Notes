import numpy as np

class Linear_Regression:

  # Initializing Hyperparameters: Learning Rate, No. of Iterations
  def __init__(self, learning_rate, no_of_iterations):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations

  def fit(self, X, Y):
    # Number of training examples & number of features
    self.m, self.n = X.shape    # No. of rows & columns

    # Initialize the weight and bias
    self.w = np.zeros(self.n)
    self.b = 0
    self.X = X
    self.Y = Y

    # Implementing Gradient Descent
    for i in range(self.no_of_iterations):
      self.update_weights()

  def update_weights(self,):

    Y_prediction = self.predict(self.X)

    # Calculate Gradients
    dw = (-2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m    # weights
    db = (-2 * np.sum(self.Y - Y_prediction)) / self.m            # bias

    # Update the weights
    self.w = self.w - self.learning_rate * dw                     # weights
    self.b = self.b - self.learning_rate * db                     # bias

  def predict(self, X):
    return X.dot(self.w) + self.b
