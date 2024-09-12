import numpy as np
from scipy.special import expit

class Logistic_Regression():
  def __init__(self, learning_rate, no_of_iterations):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations
  
  def fit(self, X, Y):
    
    if len(X.shape) == 1:
      X = X.reshape(-1, 1)

    self.X = X
    self.Y = Y
    self.m, self.n = X.shape

    #initiating the weight and bias
    self.w = np.zeros(self.n)
    self.b = 0

    #implementing gradient descent
    for i in range(self.no_of_iterations):
      self.update_weights()

  def update_weights(self):
    # Y_prediction = self.predict(self.X)    we do not do predict here because predict function will give classification and not the value as given in regression model
    #sigmoid function
    z = self.X.dot(self.w) + self.b
    Y_prediction = expit(z)   #expit  giveing all zeroes in X_test 
    # z = self.X.dot(self.w) + self.b
    # z = np.clip(z, -500, 500)  # Clip values to prevent overflow
    # Y_prediction = 1 / (1 + np.exp(-z))

    #calculate gradients, (derivatives)
    dw = ((self.X.T).dot(Y_prediction - self.Y))/self.m
    db = (np.sum(Y_prediction - self.Y))/self.m

    #updating the weights
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db

  #sigmoid equation and decision boundary
  def predict(self, X):
    if len(X.shape) == 1:
      X = X.reshape(-1, 1)
    z = X.dot(self.w) + self.b
    Y_pred = expit(z)
    # z = X.dot(self.w) + self.b
    # z = np.clip(z, -500, 500)  # Clip values to prevent overflow
    # Y_prediction = 1 / (1 + np.exp(-z))
    return np.where(Y_pred > 0.5, 1, 0)