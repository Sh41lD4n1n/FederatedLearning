import torch.optim as optim
from torch.optim import Optimizer

"""
Optimizer_SGD - собственная имплементация 
SGD оптимизатора
(пробовал использовать и Optimizer_SGD и встроенную версию SGD)
"""


class Optimizer_SGD(Optimizer):
    
    def __init__(self, params, lr=1e-2):
        super(Optimizer_SGD, self).__init__(params, defaults={'lr': lr})
        #self.momentum = momentum
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict()#dict(mom=torch.zeros_like(p.data))
    
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict()#dict(mom=torch.zeros_like(p.data))
                #mom = self.state[p]['mom']
                #mom = self.momentum * mom - group['lr'] * p.grad.data
                
                #self.w = self.w - M_scaller@(self.lr*(precond_value@der))
                p.data -= group['lr']*p.grad.data
