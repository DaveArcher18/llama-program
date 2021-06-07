import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback
from torchmetrics import ConfusionMatrix

from sklearn.decomposition import PCA

class ConfusionMatrixCallback(Callback):
    ''' 
    Callback to plot a confusion matrix every n_epochs at the end of the train/validation/test epoch. 
    By default it plots a confusion matrix after every 10th validation epoch.

    To setup the callback:

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

    n_epochs (int, default = 10)                    A plot will be produced every n_epochs

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



class EmbeddingPlotterCallback(Callback):
    """
    Callback to plot embeddings every n_epochs at the end of the train/validation/test epoch. 
    By default it plots 2D and 3D after every 10th validation epoch.

    To setup the callback (with labelled data): 

    The following must be present in the model's init (n is the dimension of the embeddings):
        self.storage = torch.zeros(1, n + 1).to(device = 'cpu') 

    In each step definition (train/val/test) the folowing must be present after calculting the embeddings:
        #labels is your target, embs is the tensor containing the embeddings
        
        new_storage = torch.cat((embs.to('cpu'), torch.transpose(labels.unsqueeze(0), 0, 1).to('cpu')), 1)
        self.storage = torch.cat((self.storage, new_storage), 0)

    To setup the callback (with unlabelled data):
    
    The following must be present in the model's init (n is the dimension of the embeddings):
        self.storage = torch.zeros(1, n).to(device = 'cpu') 
    
    In each step definition (train/val/test) the folowing must be present after calculting the embeddings:
        #embs is the tensor containing the embeddings
        self.storage = torch.cat((self.storage, embs.to('cpu')), 0)    
        

    Arguments:
    train (bool, default = False)                   If set to True embeddings will be plotted after every n_epochs training epochs

    val (bool, default = True)                      If set to True embeddings will be plotted after every n_epochs validation epochs

    test (bool, default = False)                    If set to True embeddings will be plotted after every n_epochs testing epochs

    dim_2 (bool, default = True)                    If set to True 2 dimensional embeddings will be plotted every n_epochs after the chose steps
    
    dim_3 (bool, default = True)                    If set to True 3 dimensional embeddings will be plotted every n_epochs after the chose steps

    n_epochs (int, default = 10)                    Plot will be produced every n_epochs

    labeled (bool, default = True)                  If true the callback will expect the data to be labelled, else the callback 
                                                    will expect the embeddings to be unlabeled   

    labels (list, default = None)                   Labels expects an ordered list of the class names (so the first entry corresponds to 
                                                    the label 0, etc). If passed the plot will distunguish the embeddings by string label
                                                    instead of by int label (makes the plot much better).
    """ 
    def __init__(self, train = False, val = True, test = False, dim_2 = True, dim_3 = True,  n_epochs = 10, labeled = True, labels = None):
        self.train = train      
        self.val = val
        self.test = test
        self.dim_2 = dim_2
        self.dim_3 = dim_3
        self.n_epochs = n_epochs
        self.labeled = labeled
        self.labels = labels
        
        
    def _plot_embeddings(self, storage, step_title):
        if self.labeled:
            embs, targets = self._unpack_storage(storage)
            
            # 2D case:
            if self.dim_2:
                pca2 = PCA(n_components=2)

                reduced_embs_2 = pca2.fit_transform(embs)

                xs = reduced_embs_2[:,0]
                ys = reduced_embs_2[:,1] 

                fig = px.scatter(x = xs, y = ys, color = targets, title = '2D '+ step_title)
                fig.show()
            
            # 3D case:
            if self.dim_3:
                pca3 = PCA(n_components=3)
                reduced_embs_3 = pca3.fit_transform(embs)

                xs = reduced_embs_3[:,0]
                ys = reduced_embs_3[:,1]
                zs = reduced_embs_3[:,2] 

                fig = px.scatter_3d(x = xs, y = ys, z = zs, color = targets, title = '3D ' + step_title)
                fig.show()
        else:
            embs = self._unpack_storage(storage)
                        # 2D case:
            if self.dim_2:
                pca2 = PCA(n_components=2)

                reduced_embs_2 = pca2.fit_transform(embs)

                xs = reduced_embs_2[:,0]
                ys = reduced_embs_2[:,1] 

                fig = px.scatter(x = xs, y = ys, title = '2D '+ step_title)
                fig.show()
            
            # 3D case:
            if self.dim_3:
                pca3 = PCA(n_components=3)
                reduced_embs_3 = pca3.fit_transform(embs)

                xs = reduced_embs_3[:,0]
                ys = reduced_embs_3[:,1]
                zs = reduced_embs_3[:,2] 

                fig = px.scatter_3d(x = xs, y = ys, z = zs, title = '3D ' + step_title)
                fig.show()         
      
    
    def _unpack_storage(self, storage):
        if self.labeled:
            if self.labels:
                storage = storage[1:].detach().numpy()
                embs = storage[:,0:-1]
                targets = storage[:, -1]
                target_labels = []

                for i in targets:
                    target_labels.append(self.labels[int(i)])
                return embs, target_labels    
            
            else:
                storage = storage[1:].cpu().detach().numpy()
                return storage[:,0:-1], storage[:, -1]
        else:
            embs = storage[1:].detach().numpy()
            return embs

        
        
    def on_train_epoch_start(self, trainer, pl_module):
         if self.train and pl_module.current_epoch % self.n_epochs == 0:
                pl_module.storage = pl_module.storage[0].unsqueeze(0)
                
    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        if self.train and pl_module.current_epoch % self.n_epochs == 0:
            
            self._plot_embeddings(pl_module.storage, f'Plot of embeddings after training epoch {pl_module.current_epoch}')
            
               
    def on_validation_epoch_start(self, trainer, pl_module):
         if self.val and pl_module.current_epoch % self.n_epochs == 0:
                pl_module.storage = pl_module.storage[0].unsqueeze(0)
    
    def on_validation_epoch_end(self, trainer, pl_module, unused=None):
        if self.val and pl_module.current_epoch % self.n_epochs == 0:
            
            self._plot_embeddings(pl_module.storage, f'Plot of embeddings after validation epoch {pl_module.current_epoch}')
            
            
    def on_test_epoch_start(self, trainer, pl_module):
         if self.test and pl_module.current_epoch % self.n_epochs == 0:
                pl_module.storage = pl_module.storage[0].unsqueeze(0)
    
    def on_test_epoch_end(self, trainer, pl_module, unused=None):
        if self.test and pl_module.current_epoch % self.n_epochs == 0:
            
            self._plot_embeddings(pl_module.storage, f'Plot of embeddings after testing epoch {pl_module.current_epoch}')






