'''
Created on Jan 25, 2021

@author: paepcke
'''
# from textwrap import wrap
# 
# import itertools
# import matplotlib
# from sklearn.metrics import confusion_matrix
# 
# import numpy as np
# #****import tensorflow as tf
# import torch
# 
# from . import figure
# from . import util
# from .util import merge_kwargs, decode_bytes_if_necessary

from matplotlib.figure import Figure
import seaborn as sns

class TensorBoardPlotter:
    
    conf_matrices = []

    #------------------------------------
    # fig_from_conf_matrix 
    #-------------------
    
    def fig_from_conf_matrix(self, 
                             conf_matrix,
                             class_names,
                             title='Confusion Matrix'):
        

        self.figure = Figure(figsize=(10, 5))
        self.resize(400, 400)
        
        # create confusion matrix and its peripherals
        ax = self.figure.add_subplot(111)
        ax.set_title(title)
        ax = sns.heatmap(self.calc_norm(conf_matrix), 
                         xticklabels=class_names, 
                         yticklabels=class_names, 
                         center=0.45)
        ax.set_xlabel('actual species')
        ax.set_ylabel('predicted species')
        ax.tick_params(axis='x', labelrotation=90)

        return ax
    #------------------------------------
    # calc_norm
    #-------------------

    def calcNorm(self, conf_matrix, num_classes):
        """
        Calculates a normalized confusion matrix. Normalizes the 
        confusion matrix for the last epoch by the number of samples each species has.
        """
        
        # Get the sum of each row, which is the number
        # of samples in that row's class. Then divide
        # each element in the row by that num of samples
        # to get the percentage of predictions that ended
        # up in each cell:
          
        # Sum the rows, and turn the resulting 
        # row vector into a column vector:
        sample_sizes = conf_matrix.sum(axis=1).resize(num_classes, 1)
        
        norm_cm = conf_matrix.float() / sample_sizes
        return norm_cm 

#     #------------------------------------
#     # plot_confusion_matrix 
#     #-------------------
# 
#     def plot_confusion_matrix(self, 
#                               correct_labels, 
#                               predict_labels, 
#                               labels, 
#                               title='Confusion matrix', 
#                               tensor_name = 'MyFigure/image', 
#                               normalize=False):
#         ''' 
#         Parameters:
#             correct_labels                  : These are your true classification categories.
#             predict_labels                  : These are you predicted classification categories
#             labels                          : This is a list of labels which will be used to display the axix labels
#             title='Confusion matrix'        : Title for your matrix
#             tensor_name = 'MyFigure/image'  : Name for the output summay tensor
#         
#         Returns:
#             summary: TensorFlow summary, ready for addition to 
#                      Tensorboard as an image.
#         
#         Other itema to note:
#             - Depending on the number of category and the data , you may have to 
#               modify the figzize, font sizes etc. 
#             - Currently, some of the ticks dont line up due to rotations.
#             
#         All conf matrices are stored at the class
#         level. So each epoch can generate a CM. All
#         can be submitted to Tensorboard, and animated. 
#         '''
#         cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
#         if normalize:
#             cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
#             cm = np.nan_to_num(cm, copy=True)
#             cm = cm.astype('int')
#         
#         np.set_printoptions(precision=2)
#         ###fig, ax = matplotlib.figure.Figure()
#         
#         fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
#         ax = fig.add_subplot(1, 1, 1)
#         im = ax.imshow(cm, cmap='Oranges')
#         
#         classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
#         classes = ['\n'.join(wrap(l, 40)) for l in classes]
#         
#         tick_marks = np.arange(len(classes))
#         
#         ax.set_xlabel('Predicted', fontsize=7)
#         ax.set_xticks(tick_marks)
#         c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
#         ax.xaxis.set_label_position('bottom')
#         ax.xaxis.tick_bottom()
#         
#         ax.set_ylabel('True Label', fontsize=7)
#         ax.set_yticks(tick_marks)
#         ax.set_yticklabels(classes, fontsize=4, va ='center')
#         ax.yaxis.set_label_position('left')
#         ax.yaxis.tick_left()
#         
#         for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#             ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
#         fig.set_tight_layout(True)
#         summary = tfplot.figure.to_summary(fig, tag=tensor_name)
#         return summary

    #------------------------------------
    # clear 
    #-------------------
    
    def clear(self):
        '''
        Clear confusion matrices from the 
        plotter
        '''
        
        self.conf_matrices = []
        
