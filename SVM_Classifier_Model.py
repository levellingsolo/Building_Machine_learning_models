import numpy as np
class SVM_classifier():
  #initializing the hyperparameters
  def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations
    self.lambda_parameter = lambda_parameter

  def fit(self, X, Y):
    self.X = X if isinstance(X, np.ndarray) else X.values
    self.Y = Y if isinstance(Y, np.ndarray) else Y.values
    #  if len(X.shape) == 1:
    #   X = X.reshape(-1, 1)

    # self.X = np.asarray(X)
    # self.Y = np.asarray(Y)
   

    self.m, self.n = X.shape
    #initiating the weight and bias
    self.w = np.zeros(self.n)
    self.b = 0

    #implementing gradient descent
    for i in range(self.no_of_iterations):
      self.update_weights()

  def update_weights(self):
       
    #label encoding     to convert the outcome 0 and 1   into -1 and 1    for SVM
    Y_label = np.where(self.Y == 0 , -1, 1)
    dw = np.zeros(self.n)
    db = 0

    # gradients (dw, db)
    for index, x_i in enumerate(self.X):
      # Label Encoding Example for x_i
      # print(self.w)
      condition = Y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
      if condition:
        dw = 2 * self.lambda_parameter * self.w
        db = 0
      else:
        
        dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, Y_label[index])
        db = Y_label[index]

      #updating the weights
      self.w = self.w - self.learning_rate * dw
      self.b = self.b - self.learning_rate * db

  #predice the label for a given input value
  def predict(self, X):
    output = np.dot(X, self.w) + self.b
    predicted_label = np.sign(output)
    return predicted_label