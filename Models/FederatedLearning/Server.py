import numpy as np
#from Models.LogisticRegression.LogisticRegression import Logistic_Regression
#from Models.LogisticRegression.LogisticRegression_layer import Logistic_Regression_train
from Models.LogisticRegression.LogisticRegression_numpy import Logistic_Regression_numpy

from Models.FederatedLearning.Worker import worker


#def create_model_instance(m_name,n_f,lr,epoch,stat_collector):
#    models = {"Logistic_Regression_train":None,#Logistic_Regression_train(n_features = n_f,lr = lr,epoch = epoch,stat_collector=stat_collector),
#              "Logistic_Regression":None,#Logistic_Regression(n_features = n_f,lr = lr,epoch = epoch,stat_collector=stat_collector),
#              "Logistic_Regression_numpy":Logistic_Regression_numpy(n_features = n_f,lr = lr,epoch = epoch,stat_collector=stat_collector)}
#    return models[m_name]

class server:     
    def __init__(self,M_workers,X_train,y_train,
                stat_collectors,models,data_splitter='Het',
                m_scheduler = None):
        self.num_workers = M_workers
        self.n_features = X_train.shape[1]+1
        self.m_scheduler = m_scheduler
        self.stat_collectors = stat_collectors
        
        self.workers = self.init_workers(X_train,y_train,models,data_splitter)

        
        
    def _heterogenious_splitter(self,X,y):
        
        idx_true = np.arange(len(y))[y==1]
        idx_false = np.arange(len(y))[y==0]
        
        total_t = len(idx_true)
        total_f = len(idx_false)
        
        amount_true = int(total_t/self.num_workers)+1
        amount_false = int(total_f/self.num_workers)+1

        init_pos_t = 0
        init_pos_f = 0
        
        datasets_X = []
        datasets_y = []
        for i in range(self.num_workers):
            amount_true = int(total_t/self.num_workers)+1
            amount_false = int(total_f/self.num_workers)+1
            
            if i%2==0:
                amount_true = amount_true + int(amount_true*0.4)
                amount_false = amount_false - int(amount_false*0.4)
            else:
                amount_true = amount_true - int(amount_true*0.3)
                amount_false = amount_false + int(amount_false*0.3)
            
            idx_t = idx_true[init_pos_t:init_pos_t+amount_true]
            idx_f = idx_false[init_pos_f:init_pos_f+amount_false]
            
            idx = np.concatenate([idx_t,idx_f],axis=0)
            np.random.shuffle(idx)
            
            datasets_X.append(X[idx].copy())
            datasets_y.append(y[idx].copy())
            
            init_pos_t += amount_true
            init_pos_f += amount_false
        
        return datasets_X,datasets_y
        
    def _identical_splitter(self,X,y):
        
        idx_true = np.arange(len(y))[y==1]
        idx_false = np.arange(len(y))[y==0]
        
        total_t = len(idx_true)
        total_f = len(idx_false)
        
        amount_true = int(total_t/self.num_workers)+1
        amount_false = int(total_f/self.num_workers)+1

        init_pos_t = 0
        init_pos_f = 0
        
        datasets_X = []
        datasets_y = []
        for i in range(self.num_workers):
            idx_t = idx_true[init_pos_t:init_pos_t+amount_true]
            idx_f = idx_false[init_pos_f:init_pos_f+amount_false]
            
            idx = np.concatenate([idx_t,idx_f],axis=0)
            np.random.shuffle(idx)
            
            datasets_X.append(X[idx].copy())
            datasets_y.append(y[idx].copy())
            
            init_pos_t += amount_true
            init_pos_f += amount_false
        
        return datasets_X,datasets_y
    
    def init_workers(self,X_train,y_train,models,data_splitter):
        w_list = []
        
        dataset_X,dataset_y = [],[]
        if data_splitter=="id":
            dataset_X,dataset_y = self._identical_splitter(X_train,y_train)
        elif data_splitter=="het":
            dataset_X,dataset_y = self._heterogenious_splitter(X_train,y_train)
        
        for i,X,y in zip(np.arange(self.num_workers),dataset_X,dataset_y):
            
            

            w = worker(model = models[i],
                       X = X,y = y,m_scheduler=self.m_scheduler[i])

            self.set_data_descriptor(i,w.describe_data())

            w_list.append(w)
        return w_list
    
    def set_data_descriptor(self,i,description):

        self.stat_collectors[i].collect_data_description(description)
        

    def perform_global_step(self,grad_list):
        grad = grad_list.sum(axis=1)
        grad = grad.reshape(-1,1)
        return grad/self.num_workers
    
    def run(self,n_iter,T):
        init_grad = np.linspace(0,0.7,self.n_features).reshape((self.n_features,1))
        #init_grad = np.random.rand(self.n_features,1)#np.zeros((self.n_features,1))
        while(n_iter>0):
            
            #Local steps
            n_local_steps = T if n_iter-T >0 else n_iter
            n_iter -=T
            
            grad_list = []
            for i,worker in enumerate(self.workers):
                try:
                    grad_list.append(worker.perform_local_steps(init_grad,n_local_steps))
                except Exception as e:
                    print(e,i)
                    raise e
                
            
            grad_list = np.concatenate(grad_list,axis=1)
            
            #Global steps
            if n_iter>=0:
                init_grad = self.perform_global_step(grad_list)
            
        