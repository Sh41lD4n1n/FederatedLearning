import numpy as np
import torch

from Models.FederatedLearning.Worker import Worker


class server:     
    def __init__(self,num_workers,data,model_creator):
        self.num_workers = num_workers
        self.workers = self.init_workers(model_creator,data)

    
    def init_workers(self,model_creator,dataloaders):
        w_list = []
        
        for i,data in zip(np.arange(self.num_workers),dataloaders):

            w = Worker(model = model_creator(i),
                       data = data)

            w_list.append(w)
        return w_list
        

    def perform_global_step(self,grad_dicts):
        grad_dict = grad_dicts[0]
        for k in grad_dict.keys():
            grad_dict[k] = torch.zeros_like(grad_dict[k])
            for d in grad_dicts:
                grad_dict[k] += d[k]
            grad_dict[k] = (grad_dict[k]/self.num_workers).to(torch.float64)
        return grad_dict
    
    def run(self,n_iter,T):
        for w in self.workers:
            w.model.init_model()
        #init_grad = np.linspace(0,0.7,self.n_features).reshape((self.n_features,1))
        #init_grad = np.random.rand(self.n_features,1)#np.zeros((self.n_features,1))
        while(n_iter>0):
            
            #Local steps
            n_local_steps = T if n_iter-T >0 else n_iter
            n_iter -=T
            
            grad_list = []
            for i,worker in enumerate(self.workers):
                try:
                    worker.perform_local_steps(n_local_steps)
                    grad_list.append(worker.model.get_parameters())
                except Exception as e:
                    print(e,i)
                    raise e
            
            #Global steps
            if n_iter>=0:
                init_grad = self.perform_global_step(grad_list)
                
                for worker in self.workers:
                  worker.model.set_parameters(init_grad)
        
