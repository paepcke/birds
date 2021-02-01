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

import itertools
import os
import socket, sys

from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from matplotlib import cm as col_map
import torch
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.utils import make_grid

import numpy as np
import seaborn as sns


#*****************
#
# For remote debugging via pydev and Eclipse:
# If uncommented, will hang if started from
# on Quatro or Quintus, and will send a trigger
# to the Eclipse debugging service on the same
# or different machine:
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
            self.logdir = os.path.join(self.curr_dir, 'runs')
        self.logdir = logdir

    #------------------------------------
    # fig_from_conf_matrix 
    #-------------------
    
    def fig_from_conf_matrix(self, 
                             conf_matrix,
                             class_names,
                             title='Confusion Matrix'):
        

        # Subplot 111: array of subplots has
        # 1 row, 1 col, and the requested axes
        # is in position 1 (1-based):
        # Need figsize=(10, 5) somewhere
        fig, ax = plt.subplots()
        cmap = col_map.Blues

        fig.set_tight_layout(True)
        fig.suptitle(title)
        ax.set_xlabel('Actual species')
        ax.set_ylabel('Predicted species')
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)

        cm_normed = self.calc_norm(conf_matrix)
        
        heatmap_ax = sns.heatmap(
            cm_normed,
            #*****vmin=0.0, vmax=1.0,
            cmap=cmap,
            annot=True,  # Do write percentages into cells
            fmt='g',     # Avoid scientific notation
            cbar=True,   # Do draw color bar legend
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=1,# Pixel,
            linecolor='black'
            )
        heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), 
                                   rotation = 45 
                                   )
        return heatmap_ax
    
    #------------------------------------
    # calc_norm
    #-------------------

    def calc_norm(self, conf_matrix):
        '''
        Calculates a normalized confusion matrix.
        Normalizes by the number of samples that each 
        species contributed to the confusion matrix.
        Each cell in the returned matrix will be a
        percentage. If no samples were present for a
        particular class, the respective cells will
        contain -1.
        
        It is assumed that rows correspond to the classes 
        truth labels, and cols to the classes of the
        predictions.
        
        @param conf_matrix: confusion matrix to normalize
        @type conf_matrix: numpy.ndarray[int]
        @returned a new confusion matrix with cells replaced
            by the percentage of time that cell's prediction
            was made. Cells of classes without any samples in
            the dataset will contain -1 
        @rtype numpy.ndarray[float]
        '''

        # Get the sum of each row, which is the number
        # of samples in that row's class. Then divide
        # each element in the row by that num of samples
        # to get the percentage of predictions that ended
        # up in each cell:
          
        # Sum the rows, and turn the resulting 
        # row vector into a column vector:
        #****sample_sizes_row_vec = conf_matrix.sum(axis=1)
        #****sample_sizes_col_vec = sample_sizes_row_vec[:, np.newaxis]
        
        # When a class had no samples at all,
        # there will be divide-by-zero occurrences.
        # Suppress related warnings. The respective
        # cells will contain nan:
        
        with np.errstate(divide='ignore', invalid='ignore'):
            #*****norm_cm = conf_matrix / sample_sizes_col_vec
            num_samples_col_vec = conf_matrix.sum(axis=1)[:, np.newaxis]
            norm_cm = ((conf_matrix.astype('float') / num_samples_col_vec)*100).astype(int)
            
        # Replace any nan's with -1:
        norm_cm[np.isnan(norm_cm)] = 0
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
        # Get list of img tensor/class_idx pairs:
        img_tns_list = [img_folder[idx]
                        for idx
                        in sample_idxs]
        
        # Print <class>/file_name onto
        # each spectrogram:
        
        marked_img_tns_list = []
        for i, (img_tns, class_idx) in enumerate(img_tns_list):
            class_name = img_folder.classes[class_idx]
            # img_folder.samples is [ (full_path, class_idx), (..., ...) ]:
            img_file_basename = os.path.basename(img_folder.samples[i][0])
            marked_img_tns_list.append(
                self.print_onto_image(img_tns,
                                      f"{class_name}/{img_file_basename}" 
                                      ))
        # Turn list of img tensors into
        # a single tensor with first dim 
        # being len of list:
        marked_img_tns = torch.cat(marked_img_tns_list)

        # A 10px frame around each img:
        grid = make_grid(marked_img_tns, padding=10)
        
        if unittesting:
            return grid
        writer.add_image('Train Examples', grid)
        return grid


    #------------------------------------
    # print_onto_image 
    #-------------------
    
    def print_onto_image(self, img_src, txt, point=(10,10)):
        
        if type(img_src) == str:
            # Image is a path:
            try:
                pil_img = Image.open(img_src)
            except Exception as e:
                raise ValueError(f"Could not load img from {img_src}: {repr(e)}")
            
        elif type(img_src) == torch.Tensor:
            try:
                pil_img = transforms.ToPILImage()(img_src.squeeze_(0))
            except Exception as e:
                raise ValueError(f"Could not convert tensor to PIL img ({img_src.size()})")
        
        elif not Image.isImageType(img_src):
            raise ValueError(f"Image src must be path to img, tensor, or PIL image; not {type(img_src)}") 

        else:
            pil_img = img_src
            
        # Make a blank image for the text, initialized 
        # to transparent text color:
        #*****txt_img = Image.new("RGBA", pil_img.size, (255,255,255,0))
        txt_img = Image.new("L", pil_img.size, 255)
        
        # get a font
        fnt = ImageFont.load_default()
        # get a drawing context
        drawing = ImageDraw.Draw(txt_img)
        
        # Draw text, half opacity
        drawing.text(point, txt, font=fnt)  #******, fill=(0,0,0,128))
        # Draw text, full opacity
        #drawing.text(point, txt, font=fnt, fill=(255,255,255,255))
        
        #*****out_img = Image.alpha_composite(pil_img, txt_img)
        out_img = Image.blend(pil_img, txt_img, 0.5)

        out_tns = transforms.ToTensor()(out_img).unsqueeze_(0)
        #out_img.show()
        out_img.close()

        return out_tns
            
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
