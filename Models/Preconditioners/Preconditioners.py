import numpy as np
import abc
from abc import ABC, abstractmethod
import numpy as np

class Precond(ABC):
    
    @abstractmethod
    def get_next_value(self):
        pass

class SGD_preconditioner(Precond):
    
    def __init__(self):
        return
    def reset(self):
        return
    def get_next_value(self,derivative,second_derivative,iteration):
        return np.eye(derivative.shape[0]),np.eye(derivative.shape[0])



class Adam_preconditioner(Precond):
    def __init__(self,beta2):
        self.beta2 = beta2
        self.gradient = []

    def reset(self):
        self.gradient = []

    def get_next_value(self,derivative,second_derivative,iteration):
        iteration += 1
        self.gradient.append(derivative)
        
        k = len(self.gradient)
        
        if iteration != k:
            raise RuntimeError
        
        D_k = 0
        for i in range(1,k+1):
            D_k += (self.beta2**(k-i))*np.diag((self.gradient[i-1]*self.gradient[i-1]).reshape(-1))
        
        #print(D_k,np.diag((self.gradient[-1]*self.gradient[-1]).reshape(-1)))
        D_k = (1 - self.beta2)*D_k/(1 - self.beta2**k)
        
        return D_k,np.linalg.inv(D_k**(0.5))


class OASIS_preconditioner(Precond):
    def __init__(self,beta2,alpha):
        self.beta2 = beta2
        self.alpha = alpha
        self.D_k = 0
        self.k = 1
        #np.random.seed(seed=42)

    def reset(self):
        self.k = 1
        self.D_k = 0

    def get_next_value(self,derivative,second_derivative,iteration):
        iteration += 1
        
        if iteration != self.k:
            raise RuntimeError
        self.k += 1

        z = np.random.binomial(1, 0.5, size = second_derivative.shape[0])
        z[z==0]=-1
        
        if second_derivative.shape[0]<4:
            print(z,z.shape)
        
        v = np.diag(z*(second_derivative@z))
        
        self.D_k = self.beta2*self.D_k + (1 - self.beta2)*v
        
        D_k_before_max = self.D_k.copy()

        D_k = self.D_k.copy()
        diag_vals = self.D_k.diagonal()
        diag_vals = np.maximum(abs(diag_vals),self.alpha)
        np.fill_diagonal(D_k,diag_vals)
        
        #D_k = np.maximum(abs(self.D_k),self.alpha)
        #D_k = self.D_k.copy()
        #D_k[self.D_k==0] = self.alpha
        try:
            return D_k_before_max,np.linalg.inv(D_k)
        except Exception as e:
            print(e,D_k,D_k_before_max)
            raise e
        


        