import numpy as np

class Adam:
    def __init__(self,lr,beta1,beta2,n_features):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr

        self.m = np.zeros((n_features,1))
        self.v = np.zeros((n_features,1))
    
    def get_move(self,gradient,iteration):
        iteration = iteration + 1

        self.m = self.beta1*self.m + (1-self.beta1)*(gradient)
        self.v = self.beta2*self.v + (1-self.beta2)*(gradient**2)

        #print(self.beta1,1 - self.beta1**iteration)
        #print(self.beta2,1 - self.beta2**iteration)

        m = self.m/(1 - self.beta1**iteration)
        v = self.v/(1 - self.beta2**iteration)

        return self.lr*m / (v**0.5 + 1e-8)