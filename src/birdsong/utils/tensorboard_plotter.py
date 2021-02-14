'''
Created on Jan 25, 2021

@author: paepcke
'''
from collections import Counter
import os

from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm as col_map
from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.utils import make_grid

from birdsong.rooted_image_dataset import SingleRootImageDataset
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


#*****************
#
# For remote debugging via pydev and Eclipse:
# If uncommented, will hang if started from
# on Quatro or Quintus, and will send a trigger
# to the Eclipse debugging service on the same
# or different machine:
# import socket, sys
# if socket.gethostname() in ('quintus', 'quatro'):
#     # Point to where the pydev server 
#     # software is installed on the remote
#     # machine:
#     sys.path.append(os.path.expandvars("$HOME/Software/Eclipse/PyDevRemote/pysrc"))
#  
#     import pydevd
#     global pydevd
#     # Uncomment the following if you
#     # want to break right on entry of
#     # this module. But you can instead just
#     # set normal Eclipse breakpoints:
#     #*************
#     print("About to call settrace()")
#     #*************
#     pydevd.settrace('localhost', port=4040)
# **************** 
class TensorBoardPlotter:
    '''
    Support functionality for creating custom 
    graphs and images for submission to Tensorboard.
    Services include:
    
        o Create confusion matrix images
        o Bar charts for number of samples in each class
        o Placing a grid of images on Tensorboard
        o Writing (i.e. overlaying) text onto images
        
    No SummaryWriter is created. A writer is always
    passed in
    '''
    
    conf_matrices = []

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self):
        pass

    #------------------------------------
    # conf_matrix_to_tensorboard 
    #-------------------
    
    def conf_matrix_to_tensorboard(self,
                                   writer,
                                   conf_matrix,
                                   class_names,
                                   epoch=0,
                                   title='Confusion Matrix'
                                   ):
        conf_matrix_fig = self.fig_from_conf_matrix(conf_matrix, 
                                                    class_names, 
                                                    title)
        writer.add_figure(title, conf_matrix_fig, global_step=epoch)

    #------------------------------------
    # class_support_to_tensorboard
    #-------------------

    def class_support_to_tensorboard(self,
                                     data_src, 
                                     writer,
                                     title='Class Support'):
        '''
        Create a barchart showing number of samples
        in each class. The chart is converted to
        a tensor, and submitted to tensorboard.
        The data_src may be a dataset in the 
        pytorch sense. Or data_src may be a full path
        the the root of a training data directory.
        
        @param img_folder: either a path to samples,
            or a dataset
        @type img_folder: {str | torch.utils.data.Dataset}
        @param writer: a tensorboard summary writer
        @type writer: tensorboard.SummaryWriter
        @param title: optional title above the figure
        @type title: str
        @return: dict {<class_name> : <num_samples_for_class_name>}
            i.e. number of samples in each class. 
        @rtype: {str : int}
        '''

        if type(data_src) == str:
            # Data source is file path to 
            # root of training data. Create
            # a dataset from that tree:
            dataset = SingleRootImageDataset(data_src)
        elif type(data_src) != SingleRootImageDataset:
            raise ValueError(f"Data source must be path to data root, or a dataset, not {data_src}")
        else:
            dataset = data_src
        
        # Get sample classes ordered by sample_id
        # from 0-num_samples:
        #   array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        sample_classes   = dataset.sample_classes()
        
        # Get dict: {<class_id> : <class_name>}
        class_id_to_name = {class_id : class_name
                            for class_name, class_id
                            in dataset.class_to_id.items() 
                            }
        
        # Get {<class_id> : <num_samples>}:
        # Something like: Counter({0: 6, 1: 6})
        
        support = Counter(sample_classes)
        barchart_class_names = np.array([])
        for class_id in support.keys():
            barchart_class_names = np.append(barchart_class_names, 
                                             [class_id_to_name[class_id]])

        barchart_counts = np.array(list(support.values()))
        
        # Make a horizontal chart, so class names are
        # Y-axis labels:
        y_pos = np.arange(len(barchart_class_names))
        
        fig, ax = plt.subplots()
        fig.suptitle('Number of Samples in Each Class')
        _bar_container = ax.barh(y_pos,
                                 barchart_counts,  # Bar length (i.e. width) 
                                 tick_label=barchart_class_names,
                                 align='center')
        ax.set_xlabel('Number of Samples')
        
        # Convert matplotlib figure into 
        # an image tensor for tensorboard:
        
        # The zero for global-step arg means
        # associate fig with epoch 0:
        writer.add_figure(title, fig, 0)
        
        support_dict = {class_name : num_samples
                        for class_name, num_samples
                        in zip(barchart_class_names, 
                               barchart_counts)
                        }
        return support_dict

    #------------------------------------
    # add_image 
    #-------------------
    
    def add_image(self, 
                  writer, 
                  tag, 
                  img_path,
                  step=0,
                  to_grayscale=True,
                  img_height=200, # px
                  img_width=400 # px
                  ):
        '''
        Writes a single image to tensorboard.
        Can resize image or turn to grayscale
        if requested. If img_width or img_height
        is None, no scaling is done.
        
        @param writer: the SummaryWriter to use
        @type writer: SummaryWriter
        @param tag: the name of the image in 
            tensorboard display
        @type tag: str
        @param img_path: full path to image
        @type img_path: str
        @param step: epoch
        @type step: int
        @param to_grayscale: whether or not to conver
            to grayscale
        @type to_grayscale: bool
        @param img_height: desired image height
        @type img_height: int
        @param img_width: desired image width
        @type img_width: int
        '''
        
        the_transforms = []
        if img_height is not None and img_width is not None:
            the_transforms.append(transforms.Resize((img_height, img_width)))
        if to_grayscale:
            the_transforms.append(transforms.Grayscale())
        the_transforms.append(transforms.ToTensor())

        img_transform = transforms.Compose(the_transforms)
        img = Image.open(img_path)
        img = img_transform(img).float()

        # A 10px frame around each img:
        #grid = make_grid(img, padding=10)
        #writer.add_image(tag, grid, step)
        
        writer.add_image(tag, img, step)

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
        example train images. The img_root_dir must
        be the 'data root': the dir holding one subdir
        per class.

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
            individual images if random choice is not wanted.
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
    # fig_from_conf_matrix 
    #-------------------
    
    def fig_from_conf_matrix(self, 
                             conf_matrix,
                             class_names,
                             title='Confusion Matrix'):
        '''
        Given a confusion matrix and class names,
        return a matplotlib.pyplot axes with a
        heatmap of the matrix.
        
        @param conf_matrix: nxn confusion matrix representing
            rows:truth, cols:predicted for n classes
        @type conf_matrix: numpy.ndarray
        @param class_names: n class names to use for x/y labels
        @type class_names: [str]
        @param title: title at top of figure
        @type title: str
        @return: matplotlib figure with confusion
            matrix heatmap.
        @rtype: pyplot.Figure
        '''

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

        # Later matplotlib versions want us
        # to use the mticker axis tick locator
        # machinery:
        ax.xaxis.set_major_locator(mticker.MaxNLocator('auto'))
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels([class_name for class_name in ticks_loc],
                           rotation=45)
        
        ax.yaxis.set_major_locator(mticker.MaxNLocator('auto'))
        ticks_loc = ax.get_yticks().tolist()
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_yticklabels([class_name for class_name in ticks_loc])

        cm_normed = self.calc_conf_matrix_norm(conf_matrix)
        
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
        return fig


    #------------------------------------
    # calc_conf_matrix_norm
    #-------------------

    def calc_conf_matrix_norm(self, conf_matrix):
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
            if type(conf_matrix) == np.ndarray:
                norm_cm = ((conf_matrix.astype('float') / num_samples_col_vec)*100).astype('int')
                # Replace any nan's with 0:
                norm_cm[np.isnan(norm_cm)] = 0
            else:
                # Assume conf_matrix is a tensor:
                norm_cm = ((conf_matrix.float() / num_samples_col_vec)*100).int()
                # Replace any nan's with 0:
                norm_cm[torch.isnan(norm_cm)] = 0
        return norm_cm 

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
    # print_onto_image 
    #-------------------
    
    def print_onto_image(self, img_src, txt, point=(10,10)):
        '''
        Given an image, writes given text onto the image.
        Returns a tensor of the new image. Acceptable as image
        sources are:
        
            o File path to jpg, png, etc.
            o A tensor
            o A PIL image
            
        @param img_src: image, or a way to get the image
        @type img_src: {str | Tensor | PIL}
        @param txt: text to be printed onto the image
        @type txt: str
        @param point: where to place the text. In pixels,
            origin upper left
        @type point: [int,int]
        @return: new image with text 'burned' onto it
        @rtype: Tensor
        '''
        
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
            
        # Make a blank image for the text.
        # Match the mode (RGB/RGBA/L/...):
        
        txt_img = Image.new(pil_img.mode, pil_img.size, 255)
        
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

# -------------------- Class SummaryWriterPlus ----------

class SummaryWriterPlus(SummaryWriter):
    '''
    Identical to the standard Pytorch SummaryWriter,
    except that it does not add an additional subdirectory
    when add_hparams is called
    '''
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)

