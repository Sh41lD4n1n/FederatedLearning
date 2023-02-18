import abc
from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):

    """
    params:
        X,y numpy array
    """
    @abstractmethod
    def train(self,X:np.ndarray,y:np.ndarray):
        pass
    
    """
    params:
        X numpy array
    return 
        y Numpy array
    """
    @abstractmethod
    def test(self,X:np.ndarray) ->np.ndarray:
        pass
    
    """
    return:
        numpy array
    """
    @abstractmethod
    def get_weights(self) ->np.ndarray:
        pass
    
    """
    params:
        w: numpy
    """
    @abstractmethod
    def set_weights(self,w:np.ndarray):
        pass
    
    """
    params:
        w: numpy
        loss: float
    """
    @abstractmethod
    def log_results(self,current_loss, current_weights):
        pass