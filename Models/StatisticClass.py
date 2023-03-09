#from sklearn.metrics import balanced_accuracy_score
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import figure
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

import numpy as np

#class TensorFlowStatistic:
#    writer = SummaryWriter()


class Statistic:
    def __init__(self,lable):
        self.lable = lable

        #Weights
        #self.weights = []
        #self.weights_mean = []
        #Loss
        self.loss = []
        #Iterations
        self.iterations = 0
        self.iter_list = []

        #Accuracy
        self.c_matrix = 0
        self.balance_accuracy = 0
        self.accuracy = []

        #Grad_val
        #self.derivative = []
        #self.derivative_mean = []
        #self.estimated_der = []
        
        #Test set
        #self.X_test = X_test
        #self.y_test = y_test
        
        #Train
        #self.y_train = []
        #self.y_train_proba = []
        #self.y_pred = []
        #self.y_pred_proba = []
        self.writer = SummaryWriter(comment=self.lable)
        self.writer.add_hparams(
            {"name": self.lable},
            {"None":1}
        )

#------------Collect
    def collect_data_description(self,data_description):
        self.data_description = data_description


    def log_tensorboard_train(self,loss ,acc):
        self.writer.add_scalar('Loss/train', loss, self.iterations)
        self.writer.add_scalar('Accuracy/train', acc, self.iterations)
        
    
    def log_tensorboard_test(self,loss ,acc):
        self.writer.add_scalar('Loss/test', loss, self.iterations)
        self.writer.add_scalar('Accuracy/test', acc, self.iterations)
    
    def restore_tensorborad(self):
        for l,a in zip(self.loss,self.accuracy):
            self.log_tensorboard_test(l,a)

    def handle_train(self,loss,accuracy):
        self.iter_list.append(self.iterations)
        self.iterations += 1

        self.log_tensorboard_train(loss,accuracy)

    def handle_test(self,loss,accuracy):

        self.loss.append(loss)
        self.accuracy.append(accuracy)
        
        self.log_tensorboard_test(loss,accuracy)

#------------Loss and weights   
    def plot(self,x,y,title_name):
        plt.plot(x,y,label = self.lable)
        plt.legend()
        
        plt.xlabel("iteration")
        plt.ylabel(title_name+" value")
        plt.title(title_name)
        plt.show()
    
    def loss_plot(self):
        self.plot(self.iter_list,self.loss,"loss")

    def weight_plot(self):
        self.plot(self.iter_list,self.weights_mean,"weights")
#------------Derivative
    def der_mean_plot(self):
        self.plot(self.iter_list,self.derivative_mean,"derivative")
#------------Accuracy
    def acc_plot(self):
        self.plot(self.iter_list,self.accuracy,"accuracy")

    def print_accuracy(self):
        print(f"balanced_accuracy_score: {self.balance_accuracy}")
        print(f"accuracy_score: {self.accuracy[-1]}")
    
    def plot_confusion_matrix(self):
        
        plt.imshow(self.c_matrix)

        plt.xticks(np.arange(2), labels=['pred: false','pred: true'])
        plt.yticks(np.arange(2), labels=['real: false','real: true'])
        for i in range(2):
            for j in range(2):
                text = plt.text(j, i, self.c_matrix[i, j],
                                ha="center", va="center", color="w")

        plt.show()
        print(self.c_matrix)

#------------Data Description
    def show_data_descr(self):
        print(self.data_description)

    def show_y_distr(self):
        print("Train:")
        print(f"Min val: {self.y_train.min()},Max val: {self.y_pred_proba.max()},Mean: {self.y_pred_proba.mean()}\
            ,Var: {self.y_pred_proba.var()}")

        print("Test:")
        print(f"Min val: {self.y_pred_proba.min()},Max val: {self.y_pred_proba.max()},Mean: {self.y_pred_proba.mean()}\
            ,Var: {self.y_pred_proba.var()}")
        #unique, counts = np.unique(self.y_pred_proba, return_counts=True)
        #counts = (counts-counts.min())/counts.max()
        #plt.scatter(x = unique,y = counts,marker='.',c=y_classes)
        #plt.scatter(x = self.y_train_proba,y = np.ones(self.y_train_proba.shape),marker='.',c=self.y_train)
        plt.scatter(x = self.y_pred_proba,y = np.zeros(self.y_pred_proba.shape),marker='.',c=self.y_pred)
        plt.show()
#------------Comparison plots
    def comparison_plot(self,x,y1,y2,names,title):
        plt.plot(x,y1,label=names[0])
        plt.plot(x,y2,label=names[1])
        plt.legend(title='Models:')
        plt.title(title)
        plt.show()

    def comparison_der_mean_plot(self,other:object):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]
        self.comparison_plot(iter_list,self.derivative_mean[:iterations],other.derivative_mean[:iterations],
                                names=[self.lable,other.lable], title="mean derivative")

    def comparison_der_plot(self,other:object,num):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]
        
        w1 = [i[num] for i in self.derivative[:iterations]]
        w2 = [i[num] for i in other.derivative[:iterations]]

        self.comparison_plot(iter_list,w1,w2,
                                names=[self.lable,other.lable], title=f"derivative {num}")

    def comparison_loss_plot(self,other:object):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]

        self.comparison_plot(iter_list,self.loss[:iterations],other.loss[:iterations],
                                names=[self.lable,other.lable], title="loss")
        
    
    def comparison_weights_plot(self,other:object):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]

        self.comparison_plot(iter_list,self.weights_mean[:iterations],other.weights_mean[:iterations],
                                names=[self.lable,other.lable], title="weights")
    
    def comparison_weight_plot(self,other:object,num):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]
        
        w1 = [i[num] for i in self.weights[:iterations]]
        w2 = [i[num] for i in other.weights[:iterations]]

        self.comparison_plot(iter_list,w1,w2,
                                names=[self.lable,other.lable], title=f"weights {num}")

    def comparison_accuracy_plot(self,other:object):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]

        self.comparison_plot(iter_list,self.accuracy[:iterations],other.accuracy[:iterations],
                                names=[self.lable,other.lable], title="accuracy")
    
    def comparison_mse(self,other: object):
        elems = np.arange(len(self.y_pred_proba))
        diff = self.y_pred_proba.reshape(-1) - other.y_pred_proba.reshape(-1)
        #print(elems.shape,diff.shape)
        print(f"difference({self.lable},{other.lable}) = {self.lable} - {other.lable}")
        plt.bar(elems,diff)
        plt.show()
        mse_val = (((other.y_pred_proba - self.y_pred_proba)**2).sum()**0.5) / self.y_pred_proba.shape[0]
        print(f"MSE({self.lable},{other.lable}) = {mse_val}")

"""
class Statistic:
    def __init__(self,X_test,y_test,lable):
        self.lable = lable

        #Weights
        self.weights = []
        self.weights_mean = []
        #Loss
        self.loss = []
        #Iterations
        self.iterations = 0
        self.iter_list = []

        #Grad_val
        self.derivative = []
        self.derivative_mean = []
        self.estimated_der = []

        #Temporal Oasis matrix
        self.matrix_stat = np.zeros(1)
        self.matrix_min = []
        self.matrix_frequent = []

        #Accuracy
        self.c_matrix = 0
        self.balance_accuracy = 0
        self.accuracy = []
        

        #Data
        self.data_description = ""
        
        #Test set
        self.X_test = X_test
        self.y_test = y_test
        
        #Train
        self.y_train = []
        self.y_train_proba = []
        self.y_pred = []
        self.y_pred_proba = []


#------------Collect
    def collect_data_description(self,data_description):
        self.data_description = data_description
        
    def matrix_preprocess(self,matrix):
        om = matrix.reshape(-1).copy()
        om = np.around(om,20)
        om = om[om != 0]
        
        ""
        if om.shape[0]==0:
            a = np.array([100])
            self.matrix_stat = np.concatenate([self.matrix_stat,a])
            self.matrix_min.append(100)
            self.matrix_frequent.append(100)
        ""
        
        
        self.matrix_stat = np.concatenate([self.matrix_stat,om])
        self.matrix_min.append(abs(om).min(initial=10))
        
        val = {}
        for i in abs(om):
            if i not in val.keys():
                val[i] = 1
            else:
                val[i] += 1
        if len(list(val.keys()))==0:
            self.matrix_frequent.append(10)    
            #print(matrix)
            return
        idx = np.argmax(np.array(val.values()))
        self.matrix_frequent.append(list(val.keys())[idx])


    def handle_train(self,loss,weights,der,y_train,y_train_proba,estimated_der):
        self.y_train = y_train
        self.y_train_proba = y_train_proba

        self.weights.append(weights)
        self.weights_mean.append(weights.mean())
        
        self.loss.append(loss)

        self.derivative.append(der)
        self.derivative_mean.append(der.mean())
        
        self.iter_list.append(self.iterations)
        self.iterations += 1

        self.estimated_der.append(estimated_der.mean())
        #self.diff.append(diff)

        #self.matrix_preprocess(oasis_matrix)


    def handle_test(self,y_test,y_pred,y_pred_proba):
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        self.c_matrix = confusion_matrix(y_test,y_pred)
        self.balance_accuracy = balanced_accuracy_score(y_test,y_pred)
        self.accuracy.append( accuracy_score(y_test,y_pred) )

#------------Loss and weights   
    def plot(self,x,y,title_name):
        plt.plot(x,y,label = self.lable)
        plt.legend()
        
        plt.xlabel("iteration")
        plt.ylabel(title_name+" value")
        plt.title(title_name)
        plt.show()
    
    def loss_plot(self):
        self.plot(self.iter_list,self.loss,"loss")

    def weight_plot(self):
        self.plot(self.iter_list,self.weights_mean,"weights")
#------------Derivative
    def der_mean_plot(self):
        self.plot(self.iter_list,self.derivative_mean,"derivative")
#------------Accuracy
    def acc_plot(self):
        self.plot(self.iter_list,self.accuracy,"accuracy")

    def print_accuracy(self):
        print(f"balanced_accuracy_score: {self.balance_accuracy}")
        print(f"accuracy_score: {self.accuracy[-1]}")
    
    def plot_confusion_matrix(self):
        
        plt.imshow(self.c_matrix)

        plt.xticks(np.arange(2), labels=['pred: false','pred: true'])
        plt.yticks(np.arange(2), labels=['real: false','real: true'])
        for i in range(2):
            for j in range(2):
                text = plt.text(j, i, self.c_matrix[i, j],
                                ha="center", va="center", color="w")

        plt.show()
        print(self.c_matrix)

#------------Data Description
    def show_data_descr(self):
        print(self.data_description)

    def show_y_distr(self):
        print("Train:")
        print(f"Min val: {self.y_train.min()},Max val: {self.y_pred_proba.max()},Mean: {self.y_pred_proba.mean()}\
            ,Var: {self.y_pred_proba.var()}")

        print("Test:")
        print(f"Min val: {self.y_pred_proba.min()},Max val: {self.y_pred_proba.max()},Mean: {self.y_pred_proba.mean()}\
            ,Var: {self.y_pred_proba.var()}")
        #unique, counts = np.unique(self.y_pred_proba, return_counts=True)
        #counts = (counts-counts.min())/counts.max()
        #plt.scatter(x = unique,y = counts,marker='.',c=y_classes)
        #plt.scatter(x = self.y_train_proba,y = np.ones(self.y_train_proba.shape),marker='.',c=self.y_train)
        plt.scatter(x = self.y_pred_proba,y = np.zeros(self.y_pred_proba.shape),marker='.',c=self.y_pred)
        plt.show()
#------------Comparison plots
    def comparison_plot(self,x,y1,y2,names,title):
        plt.plot(x,y1,label=names[0])
        plt.plot(x,y2,label=names[1])
        plt.legend(title='Models:')
        plt.title(title)
        plt.show()

    def comparison_der_mean_plot(self,other:object):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]
        self.comparison_plot(iter_list,self.derivative_mean[:iterations],other.derivative_mean[:iterations],
                                names=[self.lable,other.lable], title="mean derivative")

    def comparison_der_plot(self,other:object,num):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]
        
        w1 = [i[num] for i in self.derivative[:iterations]]
        w2 = [i[num] for i in other.derivative[:iterations]]

        self.comparison_plot(iter_list,w1,w2,
                                names=[self.lable,other.lable], title=f"derivative {num}")

    def comparison_loss_plot(self,other:object):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]

        self.comparison_plot(iter_list,self.loss[:iterations],other.loss[:iterations],
                                names=[self.lable,other.lable], title="loss")
        
    
    def comparison_weights_plot(self,other:object):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]

        self.comparison_plot(iter_list,self.weights_mean[:iterations],other.weights_mean[:iterations],
                                names=[self.lable,other.lable], title="weights")
    
    def comparison_weight_plot(self,other:object,num):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]
        
        w1 = [i[num] for i in self.weights[:iterations]]
        w2 = [i[num] for i in other.weights[:iterations]]

        self.comparison_plot(iter_list,w1,w2,
                                names=[self.lable,other.lable], title=f"weights {num}")

    def comparison_accuracy_plot(self,other:object):
        iterations = min(self.iterations,other.iterations)
        iter_list = [i for i in range(iterations)]

        self.comparison_plot(iter_list,self.accuracy[:iterations],other.accuracy[:iterations],
                                names=[self.lable,other.lable], title="accuracy")
    
    def comparison_mse(self,other: object):
        elems = np.arange(len(self.y_pred_proba))
        diff = self.y_pred_proba.reshape(-1) - other.y_pred_proba.reshape(-1)
        #print(elems.shape,diff.shape)
        print(f"difference({self.lable},{other.lable}) = {self.lable} - {other.lable}")
        plt.bar(elems,diff)
        plt.show()
        mse_val = (((other.y_pred_proba - self.y_pred_proba)**2).sum()**0.5) / self.y_pred_proba.shape[0]
        print(f"MSE({self.lable},{other.lable}) = {mse_val}")
"""


#------------Comparsion from array
def plot_comparison_array_plot(x_array,y_array,labels):
    n_plots = len(x_array)
    fig, axs = plt.subplots(1,n_plots, figsize=(4*n_plots, 4), sharey=True)

    for i in range(n_plots):
        x,y,l = x_array[i],y_array[i],labels[i]
        
        axs[i].set_title(l)
        axs[i].plot(x,y)
    plt.show()


def comparsion_array_loss(array):
    #array.weights = []
    x_array = []
    y_array = []
    lables_array = []
    
    for model in array:
        x_array.append(model.iter_list)
        #print(model.loss)
        y_array.append(model.loss)
        lables_array.append(model.lable)
        
    plot_comparison_array_plot(x_array,y_array,
                            labels = lables_array)

def comparsion_array_weights(array):
    x_array = []
    y_array = []
    lables_array = []

    for model in array:
        x_array.append(model.iter_list)
        #print(model.loss)
        y_array.append(model.weights_mean)
        lables_array.append(model.lable)
        
    plot_comparison_array_plot(x_array,y_array,
                            labels = lables_array)

def comparsion_array_acc(array:object):
    x_array = []
    y_array = []
    lables_array = []

    for model in array:
        x_array.append(model.iter_list)
        #print(model.loss)
        size = len(model.accuracy)
        y_array.append(model.accuracy)
        lables_array.append(model.lable)
        
    plot_comparison_array_plot(x_array,y_array,
                            labels = lables_array)

#def k_scheduler_statistic_draw():
#    pass

#------------One plot
def oneplot_comparison_array_plot(x,y_array,labels,title,line_type_array,color_array):
    plt.figure(figsize=(12, 12))
    if len(line_type_array)!=len(y_array):
        line_type_array = ['-' for _ in y_array]

    if len(color_array)!=len(y_array):
        color_array = ['r' for _ in y_array]
    
    for n,y,l_t,c_v in zip(labels,y_array,line_type_array,color_array):
        plt.plot(x,y,label=n,ls = l_t,c = c_v)
    
    plt.ylabel(title + " value")
    plt.xlabel("iteration")

    plt.grid(axis='both', color='0.95')
    plt.legend(title='Models:')
    plt.title(title)
    plt.show()

def comparison_array_plots(x,y_array,labels,title,n_array,gamma_array):
    #f = plt.figure(figsize=(50, 50))
    n_size,gamma_size = len(n_array),len(gamma_array)
    fig, axs = plt.subplots(nrows=gamma_size, ncols=n_size,figsize=(32, 34))#, layout="constrained")#,constrained_layout=False)
    for l,y,ax in zip(labels,y_array,axs.flat):
        ax.plot(x,y)
        ax.grid(True)
        
        ax.set_title(l)
        ax.set_xlabel("iteration")
        ax.set_ylabel(title + " value")
    
    fig.suptitle(title)
    plt.show()

def get_min_iter(array):
    x_array = []

    for model in array:
        x_array.append(model.iterations)
    
    iterations = min(x_array)
    return [i for i in range(iterations)],iterations


def comparsion_array_loss_one_plot(array,n_array,gamma_array,line_type_array,color_array):
    
    y_array = []
    lables_array = []
    x_array,iterations = get_min_iter(array)
    
    for model in array:
        y_array.append(model.loss[:iterations])
        lables_array.append(model.lable)
    
    oneplot_comparison_array_plot(x_array,y_array,
                            labels = lables_array,title="Loss",line_type_array = line_type_array,color_array = color_array)

    comparison_array_plots(x_array,y_array,labels = lables_array,title="Loss",
                            n_array=n_array,gamma_array=gamma_array)

def comparsion_array_acc_one_plot(array,n_array,gamma_array,line_type_array,color_array):
    
    y_array = []
    lables_array = []
    x_array,iterations = get_min_iter(array)
    
    for model in array:
        y_array.append(model.accuracy[:iterations])
        lables_array.append(model.lable)
    
    oneplot_comparison_array_plot(x_array,y_array,
                            labels = lables_array,title="Accuracy",line_type_array = line_type_array,color_array=color_array)

    comparison_array_plots(x_array,y_array,labels = lables_array,title="Accuracy",
                            n_array=n_array,gamma_array=gamma_array)

"""
def plot(x,y,title_name,lable):
        plt.plot(x,y,label = lable)
        plt.legend()
        plt.title(title_name)
        plt.show()

def comparison_plot(x1,y1,x2,y2,title_name,lable1,lable2):
        plt.plot(x1,y1,label = lable1)
        plt.plot(x2,y2,label = lable2)
        plt.legend()
        plt.title(title_name)
        plt.show()
"""


#Mean Object-----------------------------------
def array_mean_object(array,label):
    """This function fill: Statistic object with elements:
        loss, c_matrix,accuracy, iterations, iter_list"""
    mean_stat = Statistic(lable = label)
    mean_stat.iter_list,mean_stat.iterations = get_min_iter(array)

    for model in array:
        mean_stat.loss.append(model.loss[:mean_stat.iterations])
        mean_stat.accuracy.append(model.accuracy[:mean_stat.iterations])
    
    mean_stat.loss = np.array(mean_stat.loss)
    mean_stat.loss = mean_stat.loss.mean(axis=0)
    mean_stat.loss = list(mean_stat.loss)

    mean_stat.accuracy = np.array(mean_stat.accuracy)
    mean_stat.accuracy = mean_stat.accuracy.mean(axis=0)
    mean_stat.accuracy = list(mean_stat.accuracy)

    mean_stat.restore_tensorborad()
    return mean_stat

"""
def array_mean_object(array,label):
    This function fill: Statistic object with elements:
        loss, c_matrix,accuracy, iterations, iter_list
    mean_stat = Statistic(X_test=[],y_test=[],lable = label)
    mean_stat.iter_list,mean_stat.iterations = get_min_iter(array)
    mean_stat.c_matrix = np.zeros((2,2))
    b_acc = 0

    for model in array:
        mean_stat.loss.append(model.loss[:mean_stat.iterations])
        mean_stat.accuracy.append(model.accuracy[:mean_stat.iterations])

        mean_stat.data_description += f"\n{model.data_description}" 
        mean_stat.c_matrix += mean_stat.c_matrix
        mean_stat.balance_accuracy += model.balance_accuracy
        
    
    mean_stat.c_matrix = mean_stat.c_matrix/len(array)
    mean_stat.balance_accuracy = mean_stat.balance_accuracy/len(array)
    
    mean_stat.loss = np.array(mean_stat.loss)
    mean_stat.loss = mean_stat.loss.mean(axis=0)
    mean_stat.loss = list(mean_stat.loss)

    mean_stat.accuracy = np.array(mean_stat.accuracy)
    mean_stat.accuracy = mean_stat.accuracy.mean(axis=0)
    mean_stat.accuracy = list(mean_stat.accuracy)

    return mean_stat
"""
"""
def loss_array_mean_plot(array):
    #array.weights = []
    
    y_array = []
    lables = ""
    x,iterations = get_min_iter(array)
    
    for model in array:
        y_array.append(model.loss[:iterations])
        lables = lables +'/'+ model.lable
    
    y_array = np.array(y_array)
    y = y_array.mean(axis=0)

    plot(x,y,title_name="Loss",lable=lables)

def acc_array_mean_plot(array):
    #array.weights = []
    
    y_array = []
    lables = ""
    x,iterations = get_min_iter(array)
    
    for model in array:
        y_array.append(model.accuracy[:iterations])
        lables = lables +'/'+ model.lable
    
    y_array = np.array(y_array)
    y = y_array.mean(axis=0)

    plot(x,y,title_name="Accuracy",lable=lables)
    

def comparison_loss_array_mean_plot(array1,array2):
    
    y_array = []
    lables1 = ""
    x1,iterations = get_min_iter(array1)
    
    for model in array1:
        y_array.append(model.loss[:iterations])
        lables1 = lables1 +'/'+ model.lable
    
    y_array = np.array(y_array)
    y1 = y_array.mean(axis=0)

    #----------------------------------
    y_array = []
    lables2 = ""
    x2,iterations = get_min_iter(array2)
    
    for model in array2:
        y_array.append(model.loss[:iterations])
        lables2 = lables2 +'/'+ model.lable
    
    y_array = np.array(y_array)
    y2 = y_array.mean(axis=0)

    comparison_plot(x1,y1,x2,y2,title_name="Loss",lable1=lables1,lable2=lables2)


def comparison_acc_array_mean_plot(array1,array2):

    y_array = []
    lables1 = ""
    x1,iterations = get_min_iter(array1)
    
    for model in array1:
        y_array.append(model.accuracy[:iterations])
        lables1 = lables1 +'/'+ model.lable
    
    y_array = np.array(y_array)
    y1 = y_array.mean(axis=0)

    #----------------------------------
    y_array = []
    lables2 = ""
    x2,iterations = get_min_iter(array2)
    
    for model in array2:
        y_array.append(model.accuracy[:iterations])
        lables2 = lables2 +'/'+ model.lable
    
    y_array = np.array(y_array)
    y2 = y_array.mean(axis=0)

    comparison_plot(x1,y1,x2,y2,title_name="Loss",lable1=lables1,lable2=lables2)


def k_scheduler_statistic_draw1(stat_array,labels,y_lim=[0,100]):
    
    r = [np.arange(len(stat_array)) for _ in range(len(stat_array[0]))]
    barWidth = 0.3
    for i in range(1,len(stat_array[0])):
        r[i] = [x + barWidth for x in r[i-1]]
    
    
    values = [[] for _ in range(len(stat_array[0]))]
    for iter_stat in stat_array:
        #print(iter_stat)
        for i,worker_stat in enumerate(iter_stat):
            
            acc = worker_stat.accuracy[-1]

            values[i].append(acc*100)
    
    for i in range(len(r)):
        #plt.bar(r[i], values[i], width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='poacee')
        plt.bar(r[i], values[i], width = barWidth, edgecolor = 'black', capsize=7, label=f'worker{i}')
        
    plt.ylim(y_lim[0], y_lim[1])
    
    plt.xticks([r + barWidth for r in range(len(labels))], labels)
    plt.ylabel('acc')
    plt.legend()
 
    # Show graphic
    plt.show()

"""