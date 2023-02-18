import numpy as np

class StochasticGradientDescent:
    def __init__(self,lr):
        self.lr = lr

    
    def get_move(self,gradient,iteration):

        return self.lr*gradient