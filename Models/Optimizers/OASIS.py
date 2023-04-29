import torch.optim as optim
from torch.optim import Optimizer
import torch

import datetime
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

        v = z*torch.matmul(second_derivative,z)
        return v

    def count_next_D_k(self,prev_D_k,v):
        #assert cur_iter >0,'Invalide iteration value'
        #assert cur_iter == self.current_iteration, "Internal iteration counter is not same as external one"
        if prev_D_k==0:
            return (1 - self.beta2)*v
        D_k = self.beta2*prev_D_k + (1 - self.beta2)*v
        return D_k

    def count_alpha_cut(self,D_k):
        D_k = torch.maximum(D_k.to_dense(),torch.tensor(self.alpha))
        D_k = D_k.to_sparse()

        return D_k

    def count_second_derivative(self,grad,param):
        derivative_array = []
        
        for g in grad:
            
            print("step1.1")
            start_time = datetime.datetime.now()
            ddx = torch.autograd.grad(g,param,retain_graph=True)[0].reshape(-1)
            print((datetime.datetime.now()-start_time).seconds)
            derivative_array.append(ddx)

        print("step1.2")
        start_time = datetime.datetime.now()
        derivative_array = torch.stack(derivative_array)
        print((datetime.datetime.now()-start_time).seconds)

        return derivative_array


    def step(self):
        self.current_iteration += 1

        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict(D_prev = 0)#dict(mom=torch.zeros_like(p.data))

                layer_shape = p.data.shape
                current_papareters = p.data.reshape(-1,1)

                cur_grad = p.grad.data.reshape(-1,1)
                print("step1")
                start_time = datetime.datetime.now()
                second_derivative = self.count_second_derivative(grad = p.grad.reshape(-1),param= p)
                print((datetime.datetime.now()-start_time).seconds)
                
                print("step2")
                start_time = datetime.datetime.now()
                v = self.count_v(second_derivative = second_derivative)
                v = v.to_sparse()
                print((datetime.datetime.now()-start_time).seconds)

                print("step3")
                start_time = datetime.datetime.now()
                D_k = self.count_next_D_k(prev_D_k = self.state[p]['D_prev'],v = v)#,cur_iter=current_iter)
                print((datetime.datetime.now()-start_time).seconds)

                print("step4")
                start_time = datetime.datetime.now()
                D_k = self.count_alpha_cut(D_k = D_k)
                print((datetime.datetime.now()-start_time).seconds)

                print("step5")
                start_time = datetime.datetime.now()
                self.state[p]['D_prev'] = D_k
                #D_k_inv = torch.inverse(D_k**(0.5))
                D_k_inv = D_k**(-1)


                update_val = torch.sparse.mm(D_k_inv,cur_grad)
                update_val = update_val.to_dense()
                update_val = update_val.reshape(-1,1)


                current_papareters = current_papareters - group['lr']*update_val
                current_papareters = current_papareters.to('cuda' if torch.cuda.is_available() else 'cpu')
                
                p.data = current_papareters.reshape(layer_shape)
                print((datetime.datetime.now()-start_time).seconds)



