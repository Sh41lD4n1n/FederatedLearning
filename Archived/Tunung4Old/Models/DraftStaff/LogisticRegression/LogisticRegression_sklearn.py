import numpy as np
from Models.ModelClass import Model
from sklearn.linear_model import SGDClassifier
from Models.StatisticClass import Statistic


class Logistic_Regression_origin(Model):

    def __init__(self,stat_collector,n_features=-1,lr = 1e-2,epoch = 1000) -> None:
        self.log_regr = SGDClassifier(loss = 'log',max_iter=epoch,
              random_state=42,
              alpha=0,
              learning_rate='constant',eta0=lr,penalty=None)
        self.lr = lr
        self.epoch = epoch
        self.stat_collector = stat_collector


    def train(self, X: np.ndarray, y: np.ndarray,m_scheduler=None):
        #X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        
        for i in range(self.epoch):
            self.log_regr.partial_fit(X=X,y=y,classes=[0,1])
            
            y_pred_proba = self.log_regr.predict_proba(X)[:,1]
            y_pred = self.log_regr.predict(X)

            loss = self._loss(y,y_pred_proba)
            der = self._loss_derivative(X,y,y_pred)

            calculated_der = self._evaluate_derivative(X,y)

            self.log_results(current_loss=loss,
                            current_weights=self.get_weights(),
                            current_der = der,
                            estimated_der = calculated_der,
                            current_y_train = y_pred,
                            current_y_train_proba = y_pred_proba)


    def test(self, X: np.ndarray) -> np.ndarray:
        #X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
        y_proba = self.log_regr.predict_proba(X)[:,1]
        y_pred = self.log_regr.predict(X)

        return y_proba,y_pred 
    
    def set_weights(self, w: np.ndarray):
        pass
    
    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.log_regr.intercept_.reshape((1,-1)),self.log_regr.coef_],axis=1).reshape(-1,1)

    def _evaluate_derivative(self,X,y_true):
        
        Loss_derivatives = []
        n_features = self.get_weights().shape[0]
        for i in range(n_features):
            basis = np.zeros(n_features)
            basis[i] = 1

            w = self.get_weights()
            epsilon = 1e-10
            
            self.set_weights(w+epsilon*basis)
            y = self.log_regr.predict_proba(X)[:,1]
            y[y == 0] = 1e-10
            y[(1-y) == 0] = 1 - 1e-10
            l_pluss = -( y_true*np.log(y) + (1 - y_true)*np.log(1-y))/y.shape[0]
            
            self.set_weights(w-epsilon*basis)
            y = self.log_regr.predict_proba(X)[:,1]
            y[y == 0] = 1e-10
            y[(1-y) == 0] = 1 - 1e-10
            l_minus = -( y_true*np.log(y) + (1 - y_true)*np.log(1-y))/y.shape[0]
            
            self.set_weights(w)
            
            evaluated_derivative = (l_pluss - l_minus)/(2*epsilon)
            evaluated_derivative = evaluated_derivative.mean()
            
            Loss_derivatives.append(evaluated_derivative)
        
        return np.array(Loss_derivatives)
  
    def _loss(self,y_true,y):
        y[y==0] = 1e-20
        l = -( y_true*np.log(y) + (1 - y_true)*np.log(1-y))
        return l.sum()/len(y) # + torch.abs(self.w).sum()*0.1

    def _loss_derivative(self,X,y_true,y):
        n_samples = X.shape[0]
        #print(X.shape,y_true.shape,(y.reshape(-1,1) - y_true.reshape(-1,1)).shape)
        y,y_true = y.reshape(-1,1),y_true.reshape(-1,1)
        return (1/n_samples*(X.T@(y-y_true)))

    def log_results(self,current_loss, current_weights,current_der,estimated_der,current_y_train,current_y_train_proba):
        self.stat_collector.handle_train(loss=current_loss,weights=current_weights, der = current_der,
                                        estimated_der = estimated_der,
                                        y_train = current_y_train,y_train_proba=current_y_train_proba)
        

        X = self.stat_collector.X_test
        y_test = self.stat_collector.y_test

        y_pred_proba,y_pred = self.test(X)
        self.stat_collector.handle_test(y_test,y_pred,y_pred_proba)
        
        
