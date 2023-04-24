import numpy as np
import pandas as pd

"""
Worker 
worker, nod-а для локального обучения на части dataset-а
"""
class Worker:
    
    def __init__(self,model,data):
        """
        Инициализация
        Параметры:
            model           - NN модель объект класса Model
            train_loader    - dataloader части dataset-а для тренировки
            test_loader     - dataloader части dataset-а для тестов
        """
        self.model = model
        
        self.train_loader = data["trainloader"]
        self.test_loader = data["testloader"]

        #self.stat_collectors = None
    
    
    
    def perform_local_steps(self,epoch):
        """
        perform_local_steps
        Параметры
            epoch - колличество локальных шагов
        - Запуск модели на данном worker
        """

        self.model.run(epoch,self.train_loader,self.test_loader)

        return #self.model.get_parameters()
    
    """
    test (пока не применяется) 
    Параметры
        epoch - колличество локальных шагов
    - Тест модели на данном worker
    """
    def test(self,dataloader):
        return self.model.test(dataloader)

