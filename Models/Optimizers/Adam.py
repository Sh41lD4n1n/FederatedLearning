import torch.optim as optim
from torch.optim import Optimizer
import warnings
import torch


"""
Optimizer_SGD - собственная имплементация 
SGD оптимизатора
(пробовал использовать и Optimizer_SGD и встроенную версию SGD)
"""


class Adam(Optimizer):
    
    def __init__(self, params,beta2, lr=1e-2):
        super(Adam, self).__init__(params, defaults={'lr': lr})
        self.beta2 = beta2
        self.current_iter = 0
        #self.gradients = []

        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(grads=[])
    
    def count_sum(self,grad_list):
        assert self.current_iter>-1, "invalide current_iteration"
        assert self.current_iter==len(grad_list), "Internal iteration counter is not same as external one"
        D_k_sum = 0
        for i in range(1,self.current_iter+1):
            current_grad = grad_list[i-1]
            D_k_sum += (self.beta2**(self.current_iter-i))*(current_grad
                *current_grad)

        return D_k_sum#torch.sparse.spdiags(D_k_sum,torch.tensor([0]))

    def count_param_d(self,D_k_sum):
        D_k = (1 - self.beta2)*D_k_sum/(1 - self.beta2**self.current_iter)
                
        D_k_inv = D_k**(-0.5)
        return D_k_inv,D_k


    def step(self):

        self.current_iter += 1
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    warnings.warn("New layer state was added")
                    self.state[p] = dict(grads=[])#dict(mom=torch.zeros_like(p.data))

                self.state[p]["grads"].append(p.grad.data.detach().reshape(-1))
                
                layer_shape = p.data.shape

                D_k_sum = self.count_sum(grad_list = self.state[p]["grads"])
                
                D_k_inv,_ = self.count_param_d(D_k_sum = D_k_sum)

                D_k_inv = D_k_inv.cpu()
                D_k_inv = torch.sparse.spdiags(D_k_inv,torch.tensor([0]),(D_k_inv.shape[0],D_k_inv.shape[0]))
                
                current_papareters = p.data.reshape(-1)
                current_papareters = current_papareters.cpu()

                cur_grad = p.grad.data.reshape(-1,1)
                cur_grad = cur_grad.cpu()
                cur_grad = cur_grad.to_sparse()
                

                update_val = torch.sparse.mm(D_k_inv,cur_grad)
                update_val = update_val.to_dense()
                update_val = update_val.reshape(-1)

                current_papareters = current_papareters - group['lr']*update_val
                current_papareters = current_papareters.to('cuda' if torch.cuda.is_available() else 'cpu')
                
                p.data = current_papareters.reshape(layer_shape)

                

                
