import pandas as pd
import numpy as np
import torch
import plotly.express as px
from pytorch_lightning.callbacks import Callback
from sklearn.decomposition import PCA


class EmbeddingPlotterCallback(Callback):
    """
    Callback to plot embeddings every n_epochs at the end of the train/validation/test epoch. 
    By default it plots 2D and 3D after every 10th validation epoch.

    To setup the callback: 
    
    The following imports must be present:
        import numpy as np
        import torch
        from pytorch_lightning.callbacks import Callback
        import plotly.express as px

    The following must be present in the model's init (n is the dimension of the embeddings):
        self.storage = torch.zeros(1, n + 1)

    In each step definition (train/val/test) the folowing must be present after calculting the embeddings:
        #labels is your target, embs is the tensor containing the embeddings
        
        new_storage = torch.cat((embs, torch.transpose(labels.unsqueeze(0), 0, 1)), 1)
        self.storage = torch.cat((self.storage, new_storage), 0)

    Arguments:
    train (bool, default = False)                   If set to True embeddings will be plotted after every n_epochs training epochs

    val (bool, default = True)                      If set to True embeddings will be plotted after every n_epochs validation epochs

    test (bool, default = False)                    If set to True embeddings will be plotted after every n_epochs testing epochs

    dim_2 (bool, default = True)                    If set to True 2 dimensional embeddings will be plotted every n_epochs after the chose steps
    
    dim_3 (bool, default = True)                    If set to True 3 dimensional embeddings will be plotted every n_epochs after the chose steps

    n_epochs (int, default = 10)                    Plot will be produced every n_epochs

    labels (list, default = None)                   Labels expects an ordered list of the class names (so the first entry corresponds to 
                                                    the label 0, etc). If passed the plot will distunguish the embeddings by string label
                                                    instead of by int label (makes the plot much better).
    """ 
    def __init__(self, train = False, val = True, test = False, dim_2 = True, dim_3 = True,  n_epochs = 10, labels = None):
        self.train = train      
        self.val = val
        self.test = test
        self.dim_2 = dim_2
        self.dim_3 = dim_3
        self.n_epochs = n_epochs
        self.labels = labels
        
        
    def _plot_embeddings(self, storage, step_title):
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
      
    
    def _unpack_storage(self, storage):
        if self.labels:
            storage = storage[1:].cpu().detach().numpy()
            embs = storage[:,0:-1]
            targets = storage[:, -1]
            target_labels = []

            for i in targets:
                target_labels.append(self.labels[int(i)])
            return embs, target_labels    
        
        else:
            storage = storage[1:].cpu().detach().numpy()
            return storage[:,0:-1], storage[:, -1]
        
        
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
