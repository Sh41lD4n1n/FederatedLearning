import torch.optim as optim
from torch.optim import Optimizer
import torch

from torch.autograd.functional import hessian
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

        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(D_prev = 0)#dict(mom=torch.zeros_like(p.data))
    
    def count_v(self,second_derivative):
        z = torch.full_like(second_derivative, 0.5)
        z = torch.bernoulli(z)
        z[z == 0] = -1

        v = torch.diag(z*torch.matmul(second_derivative,z))
        return v

    def count_next_D_k(self,prev_D_k,v,cur_iter):
        assert cur_iter >0,'Invalide iteration value'
        assert cur_iter == self.current_iteration, "Internal iteration counter is not same as external one"

        D_k = self.beta2*prev_D_k + (1 - self.beta2)*v
        return D_k

    def count_alpha_cut(self,D_k):
        diag_vals = torch.diagonal(D_k, 0)
        diag_vals = torch.maximum(diag_vals,torch.tensor(self.alpha))

        D_k = torch.diagonal_scatter(D_k,diag_vals)
        return D_k
        

    def step(self,current_iter,loss_funct):
        
        assert cur_iter == self.current_iteration, "Internal iteration counter is not same as external one"
        if current_iter != self.current_iteration:
            raise RuntimeError
        self.current_iteration += 1

        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict(D_prev = 0)#dict(mom=torch.zeros_like(p.data))

                layer_shape = p.data.shape
                current_papareters = p.data.reshape(-1)

                second_derivative = hessian(loss_funct, current_papareters)

                print(second_derivative.shape)
                
                v = self.count_v(second_derivative = second_derivative)
                D_k = self.count_next_D_k(prev_D_k = self.state[p]['D_prev'],v = v)
                D_k = self.count_alpha_cut(D_k = D_k)
                D_k_inv = torch.inverse(D_k**(0.5))


                
                p.data = current_papareters - group['lr']*torch.matmul(p.grad.data,D_k_inv)
                p.data.reshape(layer_shape)



# class OASIS_preconditioner(Precond):
#     def __init__(self,beta2,alpha):
#         self.beta2 = beta2
#         self.alpha = alpha
#         self.D_k = 0
#         self.k = 1
#         #np.random.seed(seed=42)

#     def reset(self):
#         self.k = 1
#         self.D_k = 0

#     def get_next_value(self,derivative,second_derivative,iteration):
#         iteration += 1
        
#         if iteration != self.k:
#             raise RuntimeError
#         self.k += 1

#         z = np.random.binomial(1, 0.5, size = second_derivative.shape[0])
#         z[z==0]=-1
        
#         if second_derivative.shape[0]<4:
#             print(z,z.shape)
        
#         v = np.diag(z*(second_derivative@z))
        
#         self.D_k = self.beta2*self.D_k + (1 - self.beta2)*v
        
#         D_k_before_max = self.D_k.copy()

#         D_k = self.D_k.copy()
#         diag_vals = self.D_k.diagonal()
#         diag_vals = np.maximum(abs(diag_vals),self.alpha)
#         np.fill_diagonal(D_k,diag_vals)
        
#         #D_k = np.maximum(abs(self.D_k),self.alpha)
#         #D_k = self.D_k.copy()
#         #D_k[self.D_k==0] = self.alpha
#         try:
#             return D_k_before_max,np.linalg.inv(D_k)
#         except Exception as e:
#             print(e,D_k,D_k_before_max)
#             raise e
        



