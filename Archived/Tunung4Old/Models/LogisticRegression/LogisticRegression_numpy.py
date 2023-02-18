import numpy as np
from Models.ModelClass import Model
from Models.StatisticClass import Statistic


class Logistic_Regression_numpy(Model):
    def __init__(self,n_features,
                stat_collector,optimizer=None,
                lr = None,preconditioner = None,epoch = 1000) -> None:
        #self.w = np.random.rand(n_features+1,1)
        #self.w = np.zeros((n_features+1,1))
        self.w = np.linspace(0,0.7,11).reshape((n_features+1,1))

        self.lr = lr
        self.optimizer = optimizer

        self.preconditioner = preconditioner
        
        self.epoch = epoch
        
        self.stat_collector = stat_collector


    def train(self, X: np.ndarray, y: np.ndarray,m_scheduler=None):
        X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        for i in range(self.epoch):
          y_pred = self._model(X)
          
          y_pred_val = y_pred.copy()
          y_pred_val[y_pred_val>0.5] = 1
          y_pred_val[y_pred_val<1] = 0

          loss = self._loss(y,y_pred).sum()/n_samples
          der = self._loss_derivative(X,y,y_pred)
          second_der = self._loss_second_derivative(X,y_pred)

          D_k_before_max, precond_value = self.preconditioner.get_next_value(derivative = der,
                                            second_derivative = second_der,
                                            iteration = i)

          if m_scheduler==None:
            M = 1
            if self.optimizer==None:
                #w = w - lr * (D_k)^{-1} * grad
                self.w = self.w - self.lr*(precond_value@der)
            else:
                raise RuntimeError
                self.w = self.w - self.optimizer.get_move(gradient=der,iteration=i)#self.lr*der
            
          else:
            M = m_scheduler.get_schedule(i)
            self.w = self.w - M@(self.lr*(precond_value@der))
            #self.w = self.w - M@self.optimizer.get_move(gradient=der,iteration=i)#self.lr*der
            der = M@der

          calculated_der = self._evaluate_derivative(X,y)
          
          self.log_results(current_loss=loss,
                           current_weights=self.get_weights(),
                           oasis_matrix = D_k_before_max,
                           current_der = der,
                           estimated_der = calculated_der,
                           current_y_train=y_pred_val,
                           current_y_train_proba=y_pred)


    def test(self, X: np.ndarray) -> np.ndarray:
        X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
        y_pred_proba = self._model(X).T
        
        y_pred = y_pred_proba.copy()
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<1] = 0
        return y_pred_proba,y_pred
    
    def set_weights(self, w: np.ndarray):
        self.w = w.copy()
    
    def get_weights(self) -> np.ndarray:
        return self.w.copy()

  
    def _loss(self,y_true,y):
        y[y == 0] = 1e-10
        y[(1-y) == 0] = 1 - 1e-10
        l = -( y_true*np.log(y) + (1 - y_true)*np.log(1-y))/y.shape[0]
        return l # + torch.abs(self.w).sum()*0.1
    
    def _evaluate_derivative(self,X,y_true):
        
        Loss_derivatives = []
        for i in range(self.w.shape[0]):
            basis = np.zeros(self.w.shape)
            basis[i] = 1

            w = self.get_weights()
            epsilon = 1e-10
            
            self.set_weights(w+epsilon*basis)
            y = self._model(X)
            y[y == 0] = 1e-10
            y[(1-y) == 0] = 1 - 1e-10
            l_pluss = -( y_true*np.log(y) + (1 - y_true)*np.log(1-y))/y.shape[0]
            
            self.set_weights(w-epsilon*basis)
            y = self._model(X)
            y[y == 0] = 1e-10
            y[(1-y) == 0] = 1 - 1e-10
            l_minus = -( y_true*np.log(y) + (1 - y_true)*np.log(1-y))/y.shape[0]
            
            self.set_weights(w)
            
            evaluated_derivative = (l_pluss - l_minus)/(2*epsilon)
            evaluated_derivative = evaluated_derivative.mean()
            
            Loss_derivatives.append(evaluated_derivative)
        
        return np.array(Loss_derivatives)
        
        

    def _loss_derivative(self,X,y_true,y):
        y[y == 0] = 1e-10
        y[(1-y) == 0] = 1 - 1e-10
        
        n_samples = X.shape[0]
        #print(X.shape,y_true.shape,(y.reshape(-1,1) - y_true.reshape(-1,1)).shape)
        y,y_true = y.reshape(-1,1),y_true.reshape(-1,1)
        #return (1/n_samples*(X.T@(-y + y_true)))
        return (1/n_samples*(X.T@(y - y_true)))

    def _loss_second_derivative(self,X,y_pred):
        H = np.diag(y_pred.reshape(-1))
        n_samples = X.shape[0]
        

        return (1/n_samples)*(X.T@(H - H*H)@X)

    def _model(self,X):
        z = (self.w.T@X.T)
        #z = (X@self.w)
        return 1/(1. + np.exp(-z))


    def log_results(self,current_loss, current_weights,oasis_matrix,current_der,estimated_der,current_y_train,current_y_train_proba):

        self.stat_collector.handle_train(loss=current_loss,weights=current_weights, der = current_der,
                                        oasis_matrix = oasis_matrix,estimated_der = estimated_der,
                                        y_train = current_y_train,y_train_proba=current_y_train_proba)
        
        
        X = self.stat_collector.X_test
        y_test = self.stat_collector.y_test

        y_pred_proba,y_pred = self.test(X)
        self.stat_collector.handle_test(y_test,y_pred,y_pred_proba)
