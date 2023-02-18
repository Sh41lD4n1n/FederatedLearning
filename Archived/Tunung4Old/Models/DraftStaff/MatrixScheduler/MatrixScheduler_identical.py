import numpy as np
class MatrixScheduler_identical:
    
    def __init__(self):
        self.n_iter = -1
        self.size = -1
        self.M = None
    
    def set_params(self,n_iter,size):
        self.n_iter = n_iter
        self.size = size
        self.M = self._generate_schedule()
    
    
    def _generate_schedule(self):
        if self.n_iter<0 and self.size<0:
            raise Exception

        M = []
        for _ in range(self.n_iter):
            M.append(np.eye(self.size))
        
        return M

    def get_schedule(self,i):
        #print(self.M[i])
        return self.M[i]