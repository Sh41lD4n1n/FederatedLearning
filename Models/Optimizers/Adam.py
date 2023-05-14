import torch.optim as optim
from torch.optim import Optimizer
import warnings
import torch
import time


"""
Optimizer_SGD - собственная имплементация 
SGD оптимизатора
"""


class Adam(Optimizer):
    
    def __init__(self, params,beta2, lr=1e-2):
        super(Adam, self).__init__(params, defaults={'lr': lr})
        self.beta2 = beta2
        self.current_iter = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.gradients = []

        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(sum=torch.tensor(0,device = self.device))
    

    def count_sum(self,prev_iter_sum,current_grad):
        assert self.current_iter>-1, "invalide current_iteration"
        
        #D_k_sum = prev_iter_sum*self.beta2+(torch.pow(current_grad, 2)) 
        D_k_sum = torch.mul(prev_iter_sum, self.beta2)+(torch.pow(current_grad, 2)) 

        return D_k_sum#torch.sparse.spdiags(D_k_sum,torch.tensor([0]))

    def count_param_d(self,D_k_sum):
        D_k = torch.mul(1 - self.beta2,D_k_sum)/(1 - self.beta2**self.current_iter)
                
        D_k = torch.pow(D_k, 0.5)
        D_k = torch.add(D_k, 1e-8)
        D_k_inv = torch.pow(D_k,-1)
        return D_k_inv,D_k


    def step(self):

        self.current_iter += 1
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    warnings.warn("New layer state was added")
                    self.state[p] = dict(sum= torch.tensor(0,device = self.device))#dict(mom=torch.zeros_like(p.data))


                
                #a = time.time()

                layer_shape = p.data.shape

                D_k_sum = self.count_sum(prev_iter_sum = self.state[p]["sum"],
                                         current_grad = p.grad.data.detach().reshape(-1))

                self.state[p]["sum"] = D_k_sum
                
                #print("state1",time.time() - a)
                #a = time.time()
                
                D_k_inv,_ = self.count_param_d(D_k_sum = D_k_sum)

                D_k_inv = D_k_inv.reshape(-1)

                #D_k_inv = D_k_inv.cpu()
                #D_k_inv = torch.sparse.spdiags(D_k_inv,torch.tensor([0]),(D_k_inv.shape[0],D_k_inv.shape[0]),layout = torch.sparse_csr).to(self.device)
                #D_k_inv = D_k_inv.to_sparse_csr()
                #D_k_inv = D_k_inv.to(self.device)
                
                #print("state2",time.time() - a)
                #a = time.time()
                
                current_papareters = p.data.reshape(-1)#.to_sparse_csr()
                #current_papareters = current_papareters.to(self.device)
                #current_papareters = current_papareters.to_sparse_csr()

                cur_grad = p.grad.data.reshape(-1)#.to_sparse_csr()
                #cur_grad = cur_grad.to(self.device)
                #cur_grad = cur_grad.to_sparse_csr()

                #update = torch.sparse.mm(D_k_inv,cur_grad)
                #update = update.reshape(-1)
                #update = update.to(self.device)

                
                current_papareters = torch.addcmul(input = current_papareters, tensor1 = D_k_inv, tensor2 = cur_grad,value = -group['lr'])
                #current_papareters = torch.add(current_papareters,update,alpha= -group['lr'])
                #current_papareters = current_papareters - group['lr']*update   
                #current_papareters = current_papareters.cpu().to_sparse().to_dense().to(self.device)
                #current_papareters = current_papareters.to_dense()
                #current_papareters = current_papareters.to(self.device)
                
                #print("state3",time.time() - a)
                #a = time.time()
                
                p.data = current_papareters.reshape(layer_shape)
        #print("step completed")

                

                
