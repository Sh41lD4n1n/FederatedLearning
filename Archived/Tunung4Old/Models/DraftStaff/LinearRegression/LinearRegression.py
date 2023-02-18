import torch
import numpy as np

class Linear_Regression:
  def __init__(self,lr = 1e-4,epoch = 1000):
    self.w = None
    self.lr = lr
    self.epoch = epoch

  def fit(self,X,y):
    ones = torch.full((X.shape[0],1), 1)
    X = torch.concat((ones,X),1)

    n_samples = X.shape[0]
    n_features = X.shape[1]

    self.w = torch.rand(n_features,1, requires_grad=True,dtype = torch.float64)

    for i in range(self.epoch):
      loss = self._loss(y,self._model(X)).sum()
      loss.backward()
      
      with torch.no_grad():
        self.w -= self.w.grad * self.lr
        self.w.grad.zero_()


  def predict(self,X):
    ones = torch.full((X.shape[0],1), 1)
    X = torch.concat((ones,X),1)
    
    return self._model(X)

  def _loss(self,y_true,y):
    return ((y_true - y)**2).sum()
    
  def _model(self,X):
    z = (X@self.w)
    return z
    

class Linear_Regression_numpy:
  def __init__(self,lr = 1e-4,epoch = 1000) -> None:
    self.w = None
    self.lr = lr
    self.epoch = epoch

  def fit(self,X,y):
    X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
    n_samples = X.shape[0]
    n_features = X.shape[1]

    self.w = np.random.rand(n_features,1)

    for i in range(self.epoch):
      y_pred = self._model(X)
      #loss = self._loss(y,y_pred).sum()

      self.w = self.w - self.lr*self._loss_derivative(X,y,y_pred)

  def predict(self,X):
    X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
    return self._model(X)
  
  def _loss(self,y_true,y):
    return ((y_true - y)**2).sum()

  def _loss_derivative(self,X,y_true,y):
    return 2*(X.T@X@self.w) - 2*(X.T@y_true)

  def _model(self,X):
    z = (X@self.w)
    return z