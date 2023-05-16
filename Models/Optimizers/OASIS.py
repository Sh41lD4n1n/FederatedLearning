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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(D_prev = 0)#dict(mom=torch.zeros_like(p.data))
    

    
    def get_z(self,grad):
        z = torch.full_like(grad, 0.5)
        z = torch.bernoulli(z)
        z[z == 0] = -1

        # v = z*torch.matmul(second_derivative,z)
        return z

    def count_next_D_k(self,prev_D_k,v):
        #assert cur_iter >0,'Invalide iteration value'
        #assert cur_iter == self.current_iteration, "Internal iteration counter is not same as external one"

        if type(prev_D_k) == int:
            return (1 - self.beta2)*v
        D_k = torch.mul(self.beta2,prev_D_k) + torch.mul((1 - self.beta2),v)
        return D_k

    def count_alpha_cut(self,D_k):
        D_k = torch.maximum(D_k,torch.tensor(self.alpha))
        # D_k = D_k.to_sparse()

        return D_k

    def count_v(self,grad,param,z):
        z = z.to(self.device)
        z = z.reshape(-1,1)
        
        # batch stat:
        # batch 1e6, 100
        # time  20,  

        batch = 10000
        size = grad.shape[0]
        steps = size//batch+1

        left_border = -batch
        right_border = 0
        v_list = []

        start_time = datetime.datetime.now()
        # print("Number of Iter:",steps)

        for i in range(steps):
            left_border += batch
            right_border  += batch
            current_size = batch if size>=right_border else size - left_border
            if current_size==0:
                break
            
            current_grad = grad[left_border:right_border]

            matrix = torch.eye(current_size).to(self.device)

            
            start_time1 = datetime.datetime.now()

            ddx = torch.autograd.grad(current_grad,param,retain_graph=True,grad_outputs=matrix,is_grads_batched=True)[0]
            ddx = ddx.reshape(current_grad.shape[0],-1)

            # print(1.1)
            # print(datetime.datetime.now() - start_time1 )
            # start_time1 = datetime.datetime.now()

            ddx = torch.matmul(ddx,z)
            v_list.append(ddx)
            del matrix

            # print(1.2)
            # print(datetime.datetime.now() - start_time1 )

        # print(1.0)
        # print(datetime.datetime.now() - start_time )
        # start_time = datetime.datetime.now()

        z = z.reshape(-1)
        v_list = torch.cat(v_list,dim=0).reshape(-1)
        # print("print(v_list.shape)")
        # print(v_list.shape)
        v = torch.mul(z,v_list)
        v = v.reshape(-1)

        # print(1.4)
        # print(datetime.datetime.now() - start_time )

        return v


    def step(self):
        self.current_iteration += 1

        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict(D_prev = 0)#dict(mom=torch.zeros_like(p.data))

                layer_shape = p.data.shape
                current_papareters = p.data.reshape(-1)
                # current_papareters = current_papareters.cpu()

                cur_grad = p.grad.data.reshape(-1)
                # cur_grad = cur_grad.cpu()

                start_time = datetime.datetime.now()
                
                v = self.count_v(grad = p.grad.reshape(-1),param= p,z=self.get_z(cur_grad))

                # print(1)
                # print( datetime.datetime.now() - start_time )
                # start_time = datetime.datetime.now()


                D_k = self.count_next_D_k(prev_D_k = self.state[p]['D_prev'],v = v)#,cur_iter=current_iter)

                # print(2)
                # print( datetime.datetime.now() - start_time )
                # start_time = datetime.datetime.now()

                
                D_k = self.count_alpha_cut(D_k = D_k)

                # print(3)
                # print( datetime.datetime.now() - start_time )
                # start_time = datetime.datetime.now()

                

                self.state[p]['D_prev'] = D_k
                #D_k_inv = torch.inverse(D_k**(0.5))
                D_k_inv = torch.pow(D_k,-1)
                D_k_inv = D_k_inv.reshape(-1)
                # D_k_inv = D_k_inv.cpu()
                
                


                # update_val = D_k_inv*cur_grad
                # update_val = update_val.to_dense()
                # update_val = update_val.cpu()
                # update_val = update_val.reshape(-1,1)


                # current_papareters = current_papareters - group['lr']*update_val
                # current_papareters = current_papareters.to(self.device)

                current_papareters = torch.addcmul(input = current_papareters, tensor1 = D_k_inv, tensor2 = cur_grad,value = -group['lr'])

                # print(4)
                # print( datetime.datetime.now() - start_time )
                
                
                p.data = current_papareters.reshape(layer_shape)

                del current_papareters
                torch.cuda.empty_cache()
        # print("End of step")



