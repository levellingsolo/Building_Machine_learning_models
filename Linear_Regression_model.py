import numpy as np

class Linear_Regression():
#initiating the parameters
  def __init__(self, learning_rate, no_of_iterations):   #these two are hyperparameters and w and b are model parameters
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations


  def fit(self, X, Y):         # these are model parameters     X is years exp and Y is salary
    # number of training examples(no of datapoints that we are using in dataset) and no of features(only 1      years of exp here)
    #m is the no of dataponts   n is the feature set
    if len(X.shape) == 1:
      X = X.reshape(-1, 1)   # this will convert (30, )  into (30, 1)  so we will have n values too 

    self.m, self.n = X.shape  # (30, 1)  no of rows and columns
    # m --> number of data ponts in the dataset ( number of rows)
    # n --> number of input features in the dataset (number of columns)

    #initiating the weight and bias
    self.w = np.zeros(self.n)       # we initiating weights as nupmy array and bias as zero because w can have more than one value if there is more than one feature column but bias will have only one value everytime
    self.b = 0
    self.X = X
    self.Y = Y
  
    #implementing gradient descent
    for i in range(self.no_of_iterations):
      self.update_weights()

  def update_weights(self):
    Y_prediction = self.predict(self.X)

    #calculate gradients
    dw = -(2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m

    db = -(2 * (np.sum(self.Y - Y_prediction))) / self.m

    #updating the weights
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db 


  def predict(self, X):
    return X.dot(self.w) + self.b