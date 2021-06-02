import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback
from torchmetrics import ConfusionMatrix

class ConfusionMatrixCallback(Callback):
    ''' This callback is designed to plot a confusion matrix every n_epochs at the end of the train/validation/test epoch. It requires
    the lines:
    self.confusion_matrix =  ConfusionMatrix(num_classes = n)
    self.cf_matrix = torch.zeros(n, n)
    to be in the models init and the line:
    self.cf_matrix += self.confusion_matrix(torch.argmax(F.softmax(logits), dim = 1), labels) 
    ### Note, in your step definition eg train step the callback assumes you have:
    data, labels = train_batch
    If you have something else as your target eg:
    data, target = train_batch
    then:
    self.cf_matrix += self.confusion_matrix(torch.argmax(F.softmax(logits), dim = 1), labels) 
    must be changed to:
    self.cf_matrix += self.confusion_matrix(torch.argmax(F.softmax(logits), dim = 1), target) 



    to be in each place you'd like a confusion matrix plotted. 
    
    The arguments train, val and test are booleans to allow the user to decide after which epochs a confusion matrix is plotted.
    The argument division takes a boolean to decide if the matrix must be divided by its sum 
    (to show percentages) or not (if set to True the division will occur).
    The optional argument title allows the user to set a main title for all the plots.
    The optional argument labels allows the user to input the categorical names in order to improve the plots.
    '''
    def __init__(self, train, val, test, division, n_epochs, title = None, labels = None):
        self.division = division
        self.n_epochs = n_epochs
        self.title = title
        self.train = train      
        self.val = val
        self.test = test
    
    def _plot_confusion_matrix(self, cf_matrix, step_title, labels = None):
        '''This funtion plots the confusion matrix in accordance with the above arguments '''
        
        cf_matrix = cf_matrix.cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 12))
        
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True Value")
        plt.suptitle(step_title)
        
        if self.title:
            ax.set_title(self.title)
            
        if labels:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            
        if self.division:
            sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues', ax = ax)
            
        else:
            sns.heatmap(cf_matrix, annot=True, 
            cmap='Blues', ax = ax)
       
        plt.show()
            
        
    def on_train_epoch_start(self, trainer, pl_module):
         if self.train and pl_module.current_epoch % self.n_epochs == 0:
                pl_module.cf_matrix = torch.zeros_like(pl_module.cf_matrix)
       #zeros the confusion matrix so it can be repopulated by only this training epoch
    
    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        if self.train and pl_module.current_epoch % self.n_epochs == 0:
            self._plot_confusion_matrix(pl_module.cf_matrix, division = self.division, step_title = f'After Train epoch {pl_module.current_epoch}')
    
    
    def on_validation_epoch_start(self, trainer, pl_module):
        if self.val and pl_module.current_epoch % self.n_epochs == 0 and pl_module.current_epoch != 0:
                pl_module.cf_matrix = torch.zeros_like(pl_module.cf_matrix)
       #zeros the confusion matrix so it can be repopulated by only this validation epoch
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val and pl_module.current_epoch % self.n_epochs == 0 and pl_module.current_epoch != 0:
            self._plot_confusion_matrix(pl_module.cf_matrix, division = self.division, step_title = f'After Validation epoch {pl_module.current_epoch}')
    
    
    def on_test_epoch_end(trainer, pl_module):
        if self.test and pl_module.current_epoch % self.n_epochs == 0:
                pl_module.cf_matrix = torch.zeros_like(pl_module.cf_matrix)
       #zeros the confusion matrix so it can be repopulated by only this testing epoch
    
    def on_test_epoch_end(trainer, pl_module):
        if self.test and pl_module.current_epoch % self.n_epochs == 0:
            self._plot_confusion_matrix(pl_module.cf_matrix, division = self.division, step_title = f'After Test epoch {pl_module.current_epoch}')
