import numpy as np
import torch

from Models.FederatedLearning.Worker import Worker
from tqdm import tqdm

class server:   
    """
    server
    Центральная нода
    - хранит всех workers
    - запускает обучение workers 
    - выполняет усреднение весов моделей (perform_global_step)
    """

    def __init__(self,num_workers,data,model_creator,same_init = True):
        """
        Инициализация:
            Параметры:
                num_workers -   количество nodes/workers
                workers -       лист объектов класса Worker
                                (объект Worker выполняет локальные шаги градиетного
                                спуска)
                model_creator - функция которая создает NN модель
                                (и хранит параметры модели: name,
                                learning_rate, optimizer, preconditioner, scaller)
                data -          (dataloaders) массив из частей CIFAR dataset который
                                будет храниться на worker
            
            - инициализирует всех worker-ов
        """  
        self.num_workers = num_workers
        self.workers = self.init_workers(model_creator,data)
        
        if same_init:
            self.initialize_models_with_same_weights()

    '''
    Пример model_creator:
    `def model_creator(w_num):
        name = f"FederatedModel_{w_num}"

        model = Model(name)
        
        #opt = Optimizer_SGD(params=model.net.parameters())
        opt = optim.SGD(model.net.parameters(), lr=0.01)
        preconditioner = None
        scaller = None
        model.set_optimizer(opt)
        #model.set_preconditioner(preconditioner)
        #model.set_scaller(scaller)
        return model`
    '''
    def init_workers(self,model_creator,dataloaders):
        """
        init_workers
        Инициализирует worker-ов
        Параметры:
            model_creator - функция которая создает NN модель
                                (и хранит параметры модели: name,
                                learning_rate, optimizer, preconditioner, scaller)
            dataloaders -   массив из частей CIFAR dataset который будет
                            храниться на worker

            - инициализирует всех worker-ов
        """
        w_list = []
        
        for i,data in zip(np.arange(self.num_workers),dataloaders):
            #инииализация worker-а i данными data, моделью возращенной model_creator


            w = Worker(model = model_creator(i),
                       data = data)

            w_list.append(w)
        return w_list

    def initialize_models_with_same_weights(self):
        parameters = self.workers[0].model.get_parameters()
        
        for w in self.workers:
            w.model.set_parameters(parameters)

    #"""
    def perform_global_step(self):
        """
        perform_global_step
            Усредняет веса моделей на workers
            (данная версия функции принимает от модели параметры (`model.parameters()`))

        """

        new_parameters = []
        #param_size колличество слоев модели
        param_size = len(self.workers[0].model.get_parameters() )
        print("size:",param_size)
        # подсчитать среднее значение для каждого слоя
        for i in range(param_size):
            #инициализировать avg_param
            param = self.workers[0].model.get_parameters()[i].data
            avg_param = torch.zeros_like(param)
            #получить среднее для ондого слоя 
            for w in self.workers:
                avg_param += w.model.get_parameters()[i].data.clone()
            avg_param = (avg_param/self.num_workers)#.float()
            new_parameters.append(avg_param.clone())
        
        return new_parameters
    #"""
    
    def run(self,n_iter,T):
        """
        run
        Запуск обучения на worker-ах
        
        Параметры:
            n_iter - количество эпох тренировки
            T -      колличество локальных шагов перед усреднением
        
        - Запуск локального обучения на workers
        """
        
        with tqdm(total=n_iter) as pbar:
            while(n_iter>0):
                
                #Local steps

                #определить сколько локальных шагов нужно совершить:
                #T если следующее вычетание n_iter-T положительное
                #оставшееся количество итераций n_iter иначе
                n_local_steps = T if n_iter-T >0 else n_iter
                n_iter -=T

                #Запуск локального обучения на workers
                grad_list = []
                for i,worker in enumerate(self.workers):
                    try:
                        worker.perform_local_steps(n_local_steps)
                        grad_list.append(worker.model.get_parameters())
                    except Exception as e:
                        print(e,i)
                        raise e

                #Глобальное усреднение
                init_grad = []
                if n_iter>=0:
                    init_grad = self.perform_global_step()
                #init_grad = self.perform_global_step(grad_list)

                    for worker in self.workers:
                        worker.model.set_parameters(init_grad)
                
                pbar.update(T)
            
