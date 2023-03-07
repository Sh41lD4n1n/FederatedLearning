import numpy as np
import pandas as pd

"""
init Worker
w = Worker()
"""

class Worker:
    def __init__(self,model,data):
        self.model = model
        
        self.train_loader = data["trainloader"]
        self.test_loader = data["testloader"]

        self.stat_collectors = None
    
    
    def perform_local_steps(self,epoch):
        
        #self.model.set_parameters(params)
        
        #self.model.preconditioner.reset()

        #if self.m_scheduler !=None:
        #    self.m_scheduler.set_params(n_iter=T,size = (self.X.shape[1]+1))

        self.model.run(epoch,self.train_loader,self.test_loader)
        
        
        return #self.model.get_parameters()
        
    def test(self,X_test):
        return self.model.test(X_test)
    
    def describe_data(self):
        pass



class worker:
    def __init__(self,model,X,y,m_scheduler):
        self.model = model
        
        self.X = X
        self.y = y
        self.m_scheduler = m_scheduler
        
    
    def perform_local_steps(self,W,T):
        self.model.epoch=T
        self.model.set_weights(W)
        self.model.preconditioner.reset()

        if self.m_scheduler !=None:
            self.m_scheduler.set_params(n_iter=T,size = (self.X.shape[1]+1))
            self.model.train(X = self.X,y = self.y,m_scheduler = self.m_scheduler)
            


        else:
            self.model.train(X = self.X,y = self.y)
        
        
        return self.model.get_weights()
        
    def test(self,X_test):
        return self.model.test(X_test)
    
    def describe_data(self):
        description = ""

        #print("Worker:")
        n_true = (self.y==1).sum()
        n_false = (self.y==0).sum()
        n_samples = self.y.shape[0]
        
        description += f"# of true val: {n_true} ({n_true/n_samples})\n"
        description += f"# of true val: {n_false} ({n_false/n_samples})\n"
        description += f"# samples: {n_samples}"
        
        #print("features statistics")
        #print(pd.DataFrame(self.X).describe())
        #return pd.DataFrame(self.X).describe()
        return description
    

