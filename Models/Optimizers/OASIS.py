import torch.optim as optim
from torch.optim import Optimizer
import torch

"""
Optimizer_SGD - собственная имплементация 
SGD оптимизатора
(пробовал использовать и Optimizer_SGD и встроенную версию SGD)
"""


class OASIS(Optimizer):
    
    def __init__(self, params,beta2,alpha, lr=1e-2):
        super(OASIS, self).__init__(params, defaults={'lr': lr})
        #self.momentum = momentum
        self.alpha = alpha
        self.beta2 = beta2
        self.current_iteration = 0
        self.loss_fn = -10
        self.current_targets = 0

        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(D_prev = 0)#dict(mom=torch.zeros_like(p.data))
    
    def set_loss(self,loss,targets):
        self.loss_fn = loss
        self.current_targets = targets

    # def check_current_iter(self,current_iter = -1):
    #     assert current_iter-1 == self.current_iteration, "Internal iteration counter is not same as external one"
    
    def count_v(self,second_derivative):
        z = torch.full_like(second_derivative, 0.5)
        z = torch.bernoulli(z)
        z[z == 0] = -1

        v = z*torch.matmul(second_derivative,z)
        return v

    def count_next_D_k(self,prev_D_k,v):
        #assert cur_iter >0,'Invalide iteration value'
        #assert cur_iter == self.current_iteration, "Internal iteration counter is not same as external one"

        D_k = self.beta2*prev_D_k + (1 - self.beta2)*v
        return D_k

    def count_alpha_cut(self,D_k):
        D_k = torch.maximum(D_k,torch.tensor(self.alpha))

        return D_k

    # def count_second_derivative(self,grad,param):
    #     derivative_array = []
    #     print(grad.requires_grad,param.requires_grad)
    #     for g in grad:
    #         print(g,param)
            
    #         ddx = torch.autograd.grad(g,param,retain_graph=True)[0].reshape(-1)
    #         print(ddx)  
    #         derivative_array.append(ddx)

    #     derivative_array = torch.stack(derivative_array)
    #     return derivative_array

    def count_second_derivative(self,param):
        return torch.autograd.functional.hessian(self.loss_fn,(param,self.current_targets))[0][0]\
                    .reshape(torch.numel(param))

    def step(self):
        assert self.loss_fn!=-10, "loss isnot defined"
        self.current_iteration += 1

        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict(D_prev = 0)#dict(mom=torch.zeros_like(p.data))

                layer_shape = p.data.shape
                current_papareters = p.data.reshape(-1)

                cur_grad = p.grad.data.reshape(-1,1)
                cur_grad = cur_grad.cpu()
                cur_grad = cur_grad.to_sparse()
                
                second_derivative = self.count_second_derivative(param= torch.tensor(p.clone(),dtype=torch.float32))


                v = self.count_v(second_derivative = second_derivative)
                D_k = self.count_next_D_k(prev_D_k = self.state[p]['D_prev'],v = v)#,cur_iter=current_iter)
                D_k = self.count_alpha_cut(D_k = D_k)

                self.state[p]['D_prev'] = D_k
                #D_k_inv = torch.inverse(D_k**(0.5))
                D_k_inv = D_k**(-1)
                D_k_inv = D_k_inv.cpu()
                D_k_inv = torch.sparse.spdiags(D_k_inv,torch.tensor([0]),(D_k_inv.shape[0],D_k_inv.shape[0]))
                
                


                
                p.data = current_papareters - group['lr']*torch.sparse.mm(D_k_inv,cur_grad)
                p.data = p.data.reshape(layer_shape)
                self.loss_fn = -10


