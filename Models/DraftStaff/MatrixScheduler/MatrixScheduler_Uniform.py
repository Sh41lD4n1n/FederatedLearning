import numpy as np
class MatrixScheduler_Uniform:
    #random_uniform = "random_uniform"
    #uniform = "uniform"
    change_on_each_global_update = "mode1"
    change_on_each_iteration = "mode2"
    change_once = "mode3"
    
    def __init__(self,alpha,gamma,change_mode):
        self.n_iter = -1
        self.size = -1
        self.change_mode = change_mode
        
        self.diagonal_values_history = []
        self.M = None

        self.alpha = alpha
        self.gamma = gamma


    
    def set_params(self,n_iter,size):
        self.n_iter = n_iter
        self.size = size

        self.M = self._generate_schedule()

        
    
    def _generate_schedule(self):
        if self.n_iter<0 and self.size<0:
            raise Exception

        if MatrixScheduler_Uniform.change_on_each_global_update == self.change_mode:

            self.diagonal_values_history.append(
                np.random.uniform(self.alpha,self.gamma,size=self.size))

            M = []
            for _ in range(self.n_iter):
                M.append(np.diag(self.diagonal_values_history[-1]))
            return M

        elif MatrixScheduler_Uniform.change_on_each_iteration == self.change_mode:
            M = []
            for _ in range(self.n_iter):
                self.diagonal_values_history.append(
                np.random.uniform(self.alpha,self.gamma,size=self.size))
                M.append(np.diag(self.diagonal_values_history[-1]))
            return M
        
        elif MatrixScheduler_Uniform.change_once == self.change_mode:
            if len(self.diagonal_values_history)>1:
                raise Exception
            if len(self.diagonal_values_history)==0:
                self.diagonal_values_history.append(
                    np.random.uniform(self.alpha,self.gamma,size=self.size))
            M = []
            for _ in range(self.n_iter):
                M.append(np.diag(self.diagonal_values_history[-1]))
            return M
        else:
            raise Exception

    def get_schedule(self,i):
        #print(self.M[i])
        return self.M[i]



class CommonMatrixScheduler_Uniform:
    #random_uniform = "random_uniform"
    #uniform = "uniform"
    change_on_each_global_update = "mode1"
    change_on_each_iteration = "mode2"
    change_once = "mode3"
    
    def __init__(self,alpha,gamma,change_mode,n_workers):
        self.n_iter = -1
        self.size = -1
        self.change_mode = change_mode
        
        self.diagonal_values_history = []
        self.M = None

        self.alpha = alpha
        self.gamma = gamma

        self.n_workers = n_workers
        self.n_workers_requests = 0


    
    def set_params(self,n_iter,size):
        if self.n_workers_requests==0:
            self.n_iter = n_iter
            self.size = size

            self.M = self._generate_schedule()
            self.n_workers_requests += 1
        
        elif self.n_workers_requests>0:
            if self.n_iter != n_iter and self.size !=size:
                raise Exception
            
            self.n_workers_requests = (self.n_workers_requests + 1)%self.n_workers
    
    def _generate_schedule(self):
        if self.n_iter<0 and self.size<0:
            raise Exception

        if CommonMatrixScheduler_Uniform.change_on_each_global_update == self.change_mode:

            self.diagonal_values_history.append(
                np.random.uniform(self.alpha,self.gamma,size=self.size))

            M = []
            for _ in range(self.n_iter):
                M.append(np.diag(self.diagonal_values_history[-1]))
            return M

        elif CommonMatrixScheduler_Uniform.change_on_each_iteration == self.change_mode:
            M = []
            for _ in range(self.n_iter):
                self.diagonal_values_history.append(
                np.random.uniform(self.alpha,self.gamma,size=self.size))
                M.append(np.diag(self.diagonal_values_history[-1]))
            return M
        
        elif CommonMatrixScheduler_Uniform.change_once == self.change_mode:
            if len(self.diagonal_values_history)>1:
                raise Exception
            if len(self.diagonal_values_history)==0:
                self.diagonal_values_history.append(
                    np.random.uniform(self.alpha,self.gamma,size=self.size))
            M = []
            for _ in range(self.n_iter):
                M.append(np.diag(self.diagonal_values_history[-1]))
            return M
        else:
            raise Exception

    def get_schedule(self,i):
        #print(self.M[i])
        return self.M[i]


