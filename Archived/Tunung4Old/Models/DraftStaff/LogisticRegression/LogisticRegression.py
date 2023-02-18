import numpy as np
import torch
from Models.ModelClass import Model
from Models.StatisticClass import Statistic

class Logistic_Regression(Model):
  def __init__(self,n_features,stat_collector,lr = 1e-4,epoch = 1000):
    self.w = torch.rand(n_features+1,1, requires_grad=True,dtype = torch.float64)
    self.lr = lr
    self.epoch = epoch
    
    self.stat_collector = stat_collector


  def train(self, X: np.ndarray, y: np.ndarray):
    X,y = torch.tensor(X), torch.tensor(y)
    ones = torch.full((X.shape[0],1), 1)
    X = torch.concat((ones,X),1)

    n_samples = X.shape[0]
    n_features = X.shape[1]


    for i in range(self.epoch):
      loss = self._loss(y,self._model(X)).sum()/n_samples
      
      loss_to_store = loss.detach().numpy()
      loss_to_store = float(loss_to_store)

      loss.backward()
      with torch.no_grad():
        self.w -= self.w.grad * self.lr
        self.w.grad.zero_()

      self.log_results(current_loss=loss_to_store,
                      current_weights=self.get_weights())


  def test(self, X: np.ndarray) -> np.ndarray:
    X = torch.tensor(X)
    ones = torch.full((X.shape[0],1), 1)
    X = torch.concat((ones,X),1)
    return self._model(X).detach().numpy()

  def get_weights(self) -> np.ndarray:
    if self.w==None:
        raise Exception("weights not inicialised")
    return self.w.detach().numpy()

  def set_weights(self, w: np.ndarray):
    with torch.no_grad():
      w = torch.tensor(w, requires_grad=True)
      self.w = w

  def _loss(self,y_true,y):
    #return ( (-y_true*torch.log(y) - (1 - y_true)*torch.log(1-y))  + torch.abs(self.w).sum()*0.1) / y.shape[0]
    return (-y_true*torch.log(y) - (1 - y_true)*torch.log(1-y))#  + torch.abs(self.w).sum()*0.1) / y.shape[0]
  """
  def _loss(self,y_true,y):
  #  return (-y_true*torch.log(y) - (1 - y_true)*torch.log(1-y)).sum() + torch.abs(self.w).sum()*0.1
  """
  def _model(self,X):
    z = (X@self.w)
    return 1/(1 + torch.exp(-z))


  def log_results(self,current_loss, current_weights):
        self.stat_collector.append(loss=current_loss,weights=current_weights)


