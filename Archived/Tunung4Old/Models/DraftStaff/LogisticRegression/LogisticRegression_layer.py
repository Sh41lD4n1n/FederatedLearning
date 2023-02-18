import torch
import numpy as np

from Models.ModelClass import Model
from Models.StatisticClass import Statistic

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
     
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs
    
    def _loss(y_true,y):
        return (-y_true*torch.log(y) - (1 - y_true)*torch.log(1-y)).sum() # + torch.abs(self.w).sum()*0.1

class CustomSGD(torch.optim.Optimizer):
    def __init__(self,params,lr= 1e-3):
        defaults = dict(lr=lr)
        super(CustomSGD, self).__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        raise RuntimeError('sparce gradient')

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                with torch.no_grad():
                    param.add_(d_p, alpha=-group['lr'])
                
                #state = self.state[param]
                #state['step'] += 1

class Logistic_Regression_train(Model):
    def __init__ (self,n_features,stat_collector,lr=1e-3,epoch=1):
        self.model = LogisticRegression(n_features,1)
        self.optimizer = CustomSGD(self.model.parameters(), lr=lr)
        self.epoch = epoch
        
        self.stat_collector = stat_collector
        
    def set_weights(self, w: np.ndarray):
        w = torch.tensor(w)
        self.model.linear = torch.nn.Parameter(w)
    
    def get_weights(self) -> np.ndarray:
        
        weights = []
        weights_list = self.model.parameters()
        for params in next(weights_list):
            for w in params:
                weights.append(w.detach().item())
        intersept = next(weights_list).detach().item()
        weights.insert(0,intersept)
        return np.array(weights)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        X_train,y_train = torch.tensor(X_train).to(torch.float32), torch.tensor(y_train).to(torch.float32)
        self.model.train()
        n_samples = X_train.shape[0]

        for i in range(self.epoch):
            loss_val = 0
            for x,y in zip(X_train,y_train):
                self.optimizer.zero_grad()
                y_pred = self.model(x)

                loss_val += self._loss(y,y_pred)
            loss_val = loss_val#/n_samples
            
            loss_to_save = loss_val.detach().numpy()
            loss_to_save = float(loss_to_save)

            loss_val.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.log_results(current_loss=loss_to_save,
                           current_weights=self.get_weights())

    def test(self, X_test: np.ndarray) -> np.ndarray:
        X_test = torch.tensor(X_test)
        X_test = X_test.to(torch.float32)
        self.model.eval()
        y_pred = []

        for x in X_test:
            y_pred.append(self.model(x).item()) 
        return np.array(y_pred)

    def _loss(self,y,y_pred):
        return LogisticRegression._loss(y,y_pred)
    
    def log_results(self,current_loss, current_weights):
        self.stat_collector.append(loss=current_loss,weights=current_weights)
