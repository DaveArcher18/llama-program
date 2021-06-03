import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback
from torchmetrics import ConfusionMatrix

class ConfusionMatrixCallback(Callback):
    ''' 
    Callback to plot a confusion matrix every n_epochs at the end of the train/validation/test epoch. 
    By default it plots a confusion matrix after every 10th validation epoch.

    To setup the callback:
    The following import must be present:
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pytorch_lightning.callbacks import Callback
        from torchmetrics import ConfusionMatrix

    For each step (train/val/test) the following must be present in the model's init (n is the number of classes):
        self.train_cm =  ConfusionMatrix(num_classes = n) #for plotting after training epochs
        self.val_cm =  ConfusionMatrix(num_classes = n) #for plotting after validation epochs
        self.test_cm =  ConfusionMatrix(num_classes = n) #for plotting after testing epochs

    In each step definition (train/val/test) the folowing must be present (target is your target):
        self.train_cm(torch.argmax(F.softmax(logits), dim = 1), target) #for plotting after training epochs
        self.val_cm(torch.argmax(F.softmax(logits), dim = 1), target) #for plotting after validation epochs
        self.test_cm(torch.argmax(F.softmax(logits), dim = 1), target) #for plotting after testing epochs
    
    Arguments:
    train (bool, default = False)                   If set to True a confusion matrix will be plotted after every n_epochs training epochs

    val (bool, default = True)                      If set to True a confusion matrix will be plotted after every n_epochs validation epochs

    test (bool, default = False)                    If set to True a confusion matrix will be plotted after every n_epochs testing epochs

    division (bool, default = False)                If set to True the plotted confusion matrix will be divided by its sum, resulting in percentages
                                                    instead of counts appearing in the plot

    title (str, default = None)                     Allows the user to set a main title for all the plots
    
    labels (list of strings, default = None)        Allows the user to input the category names in order to labels the x and y axes of the plots.
    
    '''
    def __init__(self, train = False, val = True, test = False, division = False, n_epochs = 10, title = None, labels = None):
        self.division = division
        self.n_epochs = n_epochs
        self.title = title
        self.train = train      
        self.val = val
        self.test = test
        if labels:
            self.labels = labels
        
    
    def _plot_confusion_matrix(self, cf_matrix, step_title):
        '''This funtion plots the confusion matrix in accordance with the above arguments '''
        
        cf_matrix = cf_matrix.cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 12))
        
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True Value")
        plt.suptitle(step_title)
        
        if self.title:
            ax.set_title(self.title)
            
        if self.labels:
            ax.set_xticklabels(self.labels)
            ax.set_yticklabels(self.labels)
            
        if self.division:
            sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues', ax = ax)
            
        else:
            sns.heatmap(cf_matrix, annot=True, 
            cmap='Blues', ax = ax)
       
        plt.show()
            
        
    def on_train_epoch_start(self, trainer, pl_module):
         if self.train and pl_module.current_epoch % self.n_epochs == 0:
                pl_module.train_cm.reset()
       #zeros the confusion matrix so it can be repopulated by only this training epoch
    
    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        if self.train and pl_module.current_epoch % self.n_epochs == 0:
            self._plot_confusion_matrix(pl_module.train_cm.compute(), step_title = f'After Train epoch {pl_module.current_epoch}')
    
    
    def on_validation_epoch_start(self, trainer, pl_module):
        if self.val and pl_module.current_epoch % self.n_epochs == 0 and pl_module.current_epoch != 0:
                pl_module.val_cm.reset()
       #zeros the confusion matrix so it can be repopulated by only this validation epoch
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val and pl_module.current_epoch % self.n_epochs == 0 and pl_module.current_epoch != 0:
            self._plot_confusion_matrix(pl_module.val_cm.compute(), step_title = f'After Validation epoch {pl_module.current_epoch}')
    
    
    def on_test_epoch_end(trainer, pl_module):
        if self.test and pl_module.current_epoch % self.n_epochs == 0:
                pl_module.test_cm.reset()
       #zeros the confusion matrix so it can be repopulated by only this testing epoch
    
    def on_test_epoch_end(trainer, pl_module):
        if self.test and pl_module.current_epoch % self.n_epochs == 0:
            self._plot_confusion_matrix(pl_module.test_cm.compute(), step_title = f'After Test epoch {pl_module.current_epoch}')
