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

import os

from matplotlib.figure import Figure
import torch
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.utils import make_grid

import seaborn as sns

#*****************
#
# For remote debugging via pydev and Eclipse:
# If uncommented, will hang if started from
# on Quatro or Quintus, and will send a trigger
# to the Eclipse debugging service on the same
# or different machine:
 
import socket,sys
if socket.gethostname() in ('quintus', 'quatro'):
    # Point to where the pydev server 
    # software is installed on the remote
    # machine:
    sys.path.append(os.path.expandvars("$HOME/Software/Eclipse/PyDevRemote/pysrc"))
 
    import pydevd
    global pydevd
    # Uncomment the following if you
    # want to break right on entry of
    # this module. But you can instead just
    # set normal Eclipse breakpoints:
    #*************
    print("About to call settrace()")
    #*************
    pydevd.settrace('localhost', port=4040)
# **************** 


class TensorBoardPlotter:
    
    conf_matrices = []

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, logdir=None):
        
        self.curr_dir = os.path.dirname(__file__)
        if logdir is None:
            self.logdir = os.path.join(self.curr_dir/'runs')
        self.logdir = logdir

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

    #------------------------------------
    # write_img_grid 
    #-------------------
    
    def write_img_grid(self, 
                       writer, 
                       img_root_dir, 
                       num_imgs=4, 
                       class_sample_file_pairs=None,
                       img_height=200,    # px
                       img_width=400,     # px
                       to_grayscale=True,
                       unittesting=False):
        '''
        Create and log a Tensorboard 'grid' of
        example train images. 

        @param writer: a Tensorboard Pytorch SummaryWriter
        @type writer: SummaryWriter
        @param img_root_dir: directory 
            that contains sub-directories with samples. The 
            sub-directory names are taken to be class names.  
        @type img_root_dir: str
        @param num_imgs: total number of images to
            include in the grid. If None: all images
        @type num_imgs: {None | int}
        @param class_sample_file_pairs: <class>/<img_file_name> for
            individual images
        @type class_sample_file_pairs: {None | str | [str]}
        @param img_height: height of all images
        @type img_height: int (pixels)
        @param img_width: width of all images
        @type img_width: int (pixels)
        @param to_grayscale: whether or not to convert 
            images to grayscale upon import
        @type to_grayscale: bool
        @param unittesting: controls whether grid is
            actually created, or the img tensor that
            would be contained in the grid is returned
            for testing dimensions.
        @type unittesting: bool 
        '''

        if img_root_dir is None:
            raise ValueError("Must provide path to image root dir")

        # Prepare to resize all images to a given dimension,
        # convert to grayscale if requested, and turn into
        # a tensor:
        the_transforms = [transforms.Resize((img_height, img_width))]
        if to_grayscale:
            the_transforms.append(transforms.Grayscale())
        the_transforms.append(transforms.ToTensor())

        img_transform = transforms.Compose(the_transforms)
        
        # Get an ImageFolder instance, from which 
        # we will easily find classes and samples
        
        img_folder  = ImageFolder(img_root_dir,
                                  transform=img_transform,
                                  loader=default_loader
                                  )

        # Get list of full paths to samples:
        sample_idxs = self.get_sample_indices(img_folder,
                                              num_imgs=num_imgs,
                                              class_sample_file_pairs=class_sample_file_pairs
                                              )

        img_tns_list = [img_folder[idx]
                        for idx
                        in sample_idxs]

        # We have in img_tns_list:
        #   [ (tns1, class_idx1),
        #     (tns2, class_idx2),
        #     ...
        #   ]
        # Get just a list of the tensors:
        
        tns_list = [tns_class_idx[0] for tns_class_idx in img_tns_list]

        # A 10px frame around each img:
        grid = make_grid(tns_list, padding=10)
        
        if unittesting:
            return grid
        writer.add_image('Train Examples', grid)
        return grid
            
    #------------------------------------
    # get_sample_indices 
    #-------------------
    
    def get_sample_indices(self,
                         img_folder,
                         class_sample_file_pairs,
                         num_imgs=None
                         ):
        '''
        If class_sample_file_pairs is provided,
        then num_imgs is ignored.
        
        @param img_folder:
        @type img_folder:
        @param class_sample_file_pairs:
        @type class_sample_file_pairs:
        @param num_imgs:
        @type num_imgs:
        '''

        # Caller requests particular images?
        if class_sample_file_pairs is not None:
            
            # Convert the (<class-name>,<sample-file_name>)
            # pairs to (<class_idx>,<sample-file-name>)
            requested_class_idx_sample_pairs = [
                (img_folder.class_to_idx[class_name], sample_file_nm)
                for class_name, sample_file_nm
                in class_sample_file_pairs
                ]

            # Make a more convenient dict
            #   {class-idx : [<sample-file-name>]
            requests = {}
            for class_idx, sample_path in requested_class_idx_sample_pairs:
                try:
                    requests[class_idx].append(sample_path)
                except KeyError:
                    # First sample file for this class:
                    requests[class_idx] = [sample_path]

            found_idxs = []
            for i, (sample_path, class_idx) in enumerate(img_folder.samples):
                try:
                    if os.path.basename(sample_path) in requests[class_idx]:
                        found_idxs.append(i)
                except KeyError:
                    # Not one of the requested samples:
                    continue 
            return found_idxs

        # We are asked to randomly pick images
        # from each class:
        num_samples = len(img_folder)
        num_classes = len(img_folder.classes)
        num_samples_to_get = num_samples \
                            if num_imgs is None \
                            else min(num_samples, num_imgs)
            
        # Create a dict {class-idx : <list of indices into img_folder>}
        # I.e. for each class, list the int indices i 
        # such that img_folder[i] is an img in the class.
        #  

        class_dict = {}
        for i, (sample_path, class_idx) in enumerate(img_folder.samples):
            try:
                class_dict[class_idx].append(i)
            except KeyError:
                # First sample of this class:
                class_dict[class_idx] = [i]
            
        num_imgs_per_class = round(num_samples_to_get / num_classes)
        _remaining_imgs    = num_samples_to_get % num_classes

        to_get_idxs = []
        for class_idx, sample_idx_list in class_dict.items():
            # Get a random sequence into the 
            # sample_idx_list:
            rand_sample_idxs = torch.randperm(num_imgs_per_class)
            if len(sample_idx_list) < len(rand_sample_idxs):
                # Grab them all:
                to_get_idxs.extend(sample_idx_list)
            else:
                to_get_idxs.extend([sample_idx_list[rand_pick]
                                    for rand_pick
                                    in rand_sample_idxs])
        return to_get_idxs
