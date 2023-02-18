import numpy as np
class MatrixScheduler_k_array:
    
    def __init__(self,k):
        self.n_iter = -1
        self.size = -1
        self.k = k
        self.M = None

    def set_params(self,n_iter,size):
        self.n_iter = n_iter
        self.size = size
        self.M = self._generate_schedule()
    
    
    def _generate_schedule(self):
        if self.n_iter<0 and self.size<0:
            raise Exception

        M = []
        seq_length = len(self.k)
        for i in range(self.n_iter):
            
            M.append(np.eye(self.size)*self.k[i%seq_length])
        
        return M

    def get_schedule(self,i):
        return self.M[i]