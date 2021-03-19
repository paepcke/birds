'''
Created on Jan 25, 2021

@author: paepcke
'''
import os
import random

from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm as col_map
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.utils import make_grid

from birdsong.rooted_image_dataset import SingleRootImageDataset
from birdsong.utils.learning_phase import LearningPhase
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from birdsong.utils.github_table_maker import GithubTableMaker


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
    DISPLAY_HISTORY_LEN = 8
    
    #------------------------------------
    # collection_to_tensorboard 
    #-------------------
    
    @classmethod
    def collection_to_tensorboard(cls, 
                                  tally_coll,
                                  writer,
                                  phases,
                                  epoch
                                  ):
        '''
        Reports standard results from all tallies
        in the given collection to a tensorboard.
        Included are:
        
            o Various charts
            o Result text tables
            o hparams
        
        @param tally_coll:
        @type tally_coll:
        '''
        cls.visualize_epoch(tally_coll,
                            writer,
                            phases,
                            epoch)
    #------------------------------------
    # visualize_epoch 
    #-------------------
    
    @classmethod
    def visualize_epoch(cls,
                        tally_coll,
                        writer,
                        phases,
                        epoch,
                        class_names
                        ):
        '''
        Take the ResultTally instances from
        the same epoch from the tally_coll, 
        and report appropriate aggregates to 
        tensorboard. 
        
        Computes f1 scores, accuracies, etc. for 
        given epoch.

        Separately for train and validation
        results: build one long array 
        of predictions, and a corresponding
        array of labels. Also, average the
        loss across all instances.
        
        The the preds and labels as rows to csv 
        files.
        
        @return: a ResultTally instance with all
            metrics computed for display
        @rtype: ResultTally
        '''
        
        try:
            tallies = {str(phase) : tally_coll[(epoch, phase)]
                       for phase in phases
                       }
        except KeyError as e:
            print(f"Epoch: {epoch}, phase: foo")
        

        for phase in phases:
            
            # Need learning phase in string forms
            # below:
            phase_str = str(phase)
            
            tally = tallies[phase_str]
            writer.add_scalar(f"loss/{phase_str}", 
                                   tally.mean_loss, 
                                   global_step=epoch
                                   )
            
            writer.add_scalar(f"balanced_accuracy_score/{phase_str}", 
                                   tally.balanced_acc, 
                                   global_step=epoch
                                   )
    
            writer.add_scalar(f"accuracy_score/{phase_str}", 
                                   tally.accuracy, 
                                   global_step=epoch
                                   )

            # The following are only sent to the
            # tensorboard for validation and test
            # phases.

            if phase in (LearningPhase.VALIDATING, LearningPhase.TESTING):

                # Submit the confusion matrix image
                # to the tensorboard. In the following:
                # do not provide a separate title, such as
                #  title=f"Confusion Matrix (Validation): Epoch{epoch}"
                # That would put each matrix into its own slot
                # on tensorboard, rather than having a time slider

                TensorBoardPlotter.conf_matrix_to_tensorboard(
                    writer,
                    tally.conf_matrix,
                    class_names,
                    epoch=epoch,
                    title=f"Confusion Matrix Series"
                    )

                # Versions of the f1 score:
                
                writer.add_scalar(f"{phase_str}_f1/macro", 
                                       tally.f1_macro, 
                                       global_step=epoch)
                writer.add_scalar(f"{phase_str}_f1/micro", 
                                       tally.f1_micro,
                                       global_step=epoch)
                writer.add_scalar(f"{phase_str}_f1/weighted", 
                                       tally.f1_weighted,
                                       global_step=epoch)

                # Versions of precision/recall:
                
                writer.add_scalar(f"{phase_str}_prec/macro", 
                                       tally.prec_macro, 
                                       global_step=epoch)
                writer.add_scalar(f"{phase_str}_prec/micro", 
                                       tally.prec_micro,
                                       global_step=epoch)
                writer.add_scalar(f"{phase_str}_prec/weighted", 
                                       tally.prec_weighted,
                                       global_step=epoch)
        
                writer.add_scalar(f"{phase_str}_recall/macro", 
                                       tally.recall_macro, 
                                       global_step=epoch)
                writer.add_scalar(f"{phase_str}_recall/micro", 
                                       tally.recall_micro,
                                       global_step=epoch)
                writer.add_scalar(f"{phase_str}_recall/weighted", 
                                       tally.recall_weighted,
                                       global_step=epoch)
        
        return tally

    #------------------------------------
    # conf_matrix_to_tensorboard 
    #-------------------
    
    @classmethod
    def conf_matrix_to_tensorboard(cls,
                                   writer,
                                   conf_matrix,
                                   class_names,
                                   epoch=0,
                                   title='Confusion Matrix'
                                   ):
        conf_matrix_fig = cls.fig_from_conf_matrix(conf_matrix, 
                                                   class_names, 
                                                   title)
        writer.add_figure(title, conf_matrix_fig, global_step=epoch)

    #------------------------------------
    # class_support_to_tensorboard
    #-------------------

    @classmethod
    def class_support_to_tensorboard(cls,
                                     data_src, 
                                     writer,
                                     epoch=0,
                                     title='Class Support'):
        '''
        Create a barchart showing number of training samples
        in each class. The chart is converted to
        a tensor, and submitted to tensorboard.
        
        The data_src may be:
        
           o a dataset in the pytorch sense, or 
           o a full path the root of a training data directory, or
           
        If custom_data is None, a barchart with number of samples
        in each class is created. Else custom_data is expected
        to be a dict mapping class-id => num-samples in that class.
        If provided, this data is bar-charted instead of the
        entire dataset's distribution

        @param data_src: either a path to samples,
            or a dataset
        @type data_src: {str | {int : int} | torch.utils.data.Dataset}
        @param writer: a tensorboard summary writer
        @type writer: tensorboard.SummaryWriter
        @param epoch: epoch for which support is shown
        @type epoch: int
        @param custom_data: an optional dict {class-id : sample-count} whose
            per-class count is to be bar-charted instead of the entire
            dataset
        @type custom_data: {int : int}
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
        
        # Get dict: {<class_id> : <class_name>}
        class_id_to_name = {class_id : class_name
                            for class_name, class_id
                            in dataset.class_to_id.items() 
                            }
        
        # Goal is corresponding np arrays: 
        #    class-name, num-samples-in-class.
        
        # First, get correponding tuples of 
        # class *ids* and sample counts. The
        # 'zip(*<list-of-tuples>) notation is
        # the inverse of a zip(): 
        # take [(c1,n1), (c2,n2),...] that is returned
        # from sample_distribution(), and create two
        # arrays: [c1,c2,...], and [n1,n2,...]
        
        [class_id_tuple, 
         sample_count_tuple] = zip(*dataset.sample_distribution())

        # Create np array of class *names* from the class ID tuple:
        class_names   = np.array([class_id_to_name[class_id]
                                  for class_id
                                  in class_id_tuple
                                  ])
        sample_counts = np.array(sample_count_tuple)

        # Make a horizontal chart, so class names are
        # Y-axis labels:
        y_pos = np.arange(len(class_names))
        
        fig, ax = plt.subplots()
        fig.suptitle('Number of Samples in Each Class')
        _bar_container = ax.barh(y_pos,
                                 sample_counts,  # Bar length (i.e. width) 
                                 tick_label=class_names,
                                 align='center')
        ax.set_xlabel('Number of Samples')
        
        # Convert matplotlib figure into 
        # an image tensor for tensorboard:
        
        writer.add_figure(title, fig, epoch)
        
        support_dict = {class_name : num_samples
                        for class_name, num_samples
                        in zip(class_names, sample_counts)
                        }
        return support_dict

    #------------------------------------
    # add_image 
    #-------------------
    
    @classmethod
    def add_image(cls, 
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
    
    @classmethod
    def write_img_grid(cls, 
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
        sample_idxs = cls.get_sample_indices(img_folder,
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
                cls.print_onto_image(img_tns,
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
        writer.add_image('Train Input Examples', grid)
        return grid

    #------------------------------------
    # fig_from_conf_matrix 
    #-------------------
    
    @classmethod
    def fig_from_conf_matrix(cls, 
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

        cm_normed = cls.calc_conf_matrix_norm(conf_matrix)
        
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
    # compute_confusion_matrix
    #-------------------
    
    @classmethod
    def compute_confusion_matrix(cls, 
                                 truth_labels, 
                                 predicted_class_ids,
                                 num_classes,
                                 normalize=False
                                 ):
        # Example Confustion matrix for 16 samples,
        # in 3 classes:
        # 
        #              C_1-pred, C_2-pred, C_3-pred
        #  C_1-true        3         1        0
        #  C_2-true        2         6        1
        #  C_3-true        0         0        3
        
        # The class IDs (labels kwarg) is needed for
        # sklearn to know about classes that were not
        # encountered:
        
        conf_matrix = torch.tensor(confusion_matrix(
            truth_labels,          # Truth
            predicted_class_ids,   # Prediction
            labels=list(range(num_classes)) # Class labels
            ))

        if normalize:
            conf_matrix = cls.calc_conf_matrix_norm(conf_matrix)
             
        return conf_matrix

    #------------------------------------
    # calc_conf_matrix_norm
    #-------------------

    @classmethod
    def calc_conf_matrix_norm(cls, conf_matrix):
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
    # make_f1_train_val_table 
    #-------------------
    
    @classmethod
    def make_f1_train_val_table(cls, res_list):
        '''
        Return a github flavored table:
           |phase|ep0  |ep1 |ep2 |
           |-----|-----|----|----|
           |train| f1_0|f1_1|f1_2|
           |  val| f1_0|f1_1|f1_2|
           
        for half as many epochs back as there are
        tallies available in the list of ResultTally
        instances in epoch_results.
        
        Assumption: exactly two ResultTallies are provided
        in res_list. One each for train and validation 
        results.
           
        @param res_list: list of ResultTally
            instances in oldest-epoch-first order
        @type res_list: [ResultTally]
        @return: a table
        @rtype: str
        '''
        res_len    = len(res_list)

        # Could catch the following error.
        # But it's just a special case of 
        # num train tallies unequal to num 
        # of val tallies. Wait till we catch
        # that root problem later:
        
        # Should be an even number of result
        # objs:
        #if res_len % 2 != 0:
        #    raise ValueError("Must provide two ResultTally instances per epoch")
        
        
        num_epochs = res_len // 2
        
        # First the header:
        header = []
        for i in range(num_epochs):
            header.append(f"f1-macro ep{i}")
            
        # The f1 value results for both
        # train and val:
        train_f1s = filter(lambda res_tally: res_tally.phase == LearningPhase.TRAINING,
                           res_list)
        val_f1s   = filter(lambda res_tally: res_tally.phase == LearningPhase.VALIDATING,
                           res_list)

        train_row = []
        for res in train_f1s:
            train_row.append(str(round(res.f1_macro, 1)))
            
        val_row = []
        # Second row: f1's for validation results:
        for res in val_f1s:
            val_row.append(str(round(res.f1_macro, 1)))
            
        if len(val_row) != len(train_row):
            raise ValueError(f"Must have equal num of train/val tallies; have {len(val_row)} vals and {len(train_row)} trains")
            
        tbl_content = {'col_header' : header,
                       'row_labels' : ['training', 'validation'],
                       'rows'       : [train_row, val_row] 
                       }
        tbl = GithubTableMaker.make_table(tbl_content)

        return tbl

    #------------------------------------
    # make_all_classes_f1_table
    #-------------------
    
    @classmethod
    def make_all_classes_f1_table(cls, latest_result, class_names):
        '''
        Return a github flavored table with
        with train and val f1 values for every
        class:
        
            |class|weighted mean f1 train|weighted mean f1 val|
            |-----|----------------------|--------------------|
            |  c1 |     0.1              |   0.6              |
            |  c2 |     0.1              |   0.6              |
            |  c3 |     0.1              |   0.6              |
            ---------------------------------------------------
        
        '''

        # Get the 'all-classes' version of f1 from
        # the last ResultTally for both train and val:
        t_f1s = latest_result['train'].f1_all_classes
        v_f1s = latest_result['val'].f1_all_classes
        
        if t_f1s is None or \
           v_f1s is None or \
           len(t_f1s) == 0 or\
           len(t_f1s) == 0:
            raise ValueError("Both, train and val values of f1_all_classes must be non-empty lists")
        
        # Get [[c1_train, c1_val],
        #      [c2_train, c2_val],
        #             ...
        #      ]
        res = torch.tensor([t_f1s, v_f1s]).T
        
        header = ['weighted mean f1 train','weighted mean f1 val']
        
        # And the f1 train/val numbers, one
        # class in each row:
        row_labels = []
        rows       = []
        for class_name, (f1_train, f1_val) in zip(class_names, res):
            f1_train = round(float(f1_train),1)
            f1_val   = round(float(f1_val),1)
            row_labels.append(class_name)
            rows.append([f1_train, f1_val])

        tbl_content = {
            'col_header' : header,
            'row_labels' : row_labels,
            'rows'       : rows
            }
        tbl = GithubTableMaker.make_table(tbl_content) 
        return tbl

    #------------------------------------
    # print_onto_image 
    #-------------------
    
    @classmethod
    def print_onto_image(cls, img_src, txt, point=(10,10)):
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
    
    @classmethod
    def get_sample_indices(cls,
                           img_folder,
                           class_sample_file_pairs,
                           num_imgs=None
                           ):
        '''
        If class_sample_file_pairs is provided,
        then num_imgs is ignored.
        
        @param img_folder: folder instance with training images
        @type img_folder: ImageFolder
        @param class_sample_file_pairs: optionally, pairs of 
            class-name and path to training images
        @type class_sample_file_pairs: [(<class-name>, <sample-file-name>)]
        @param num_imgs: for how many images to create spectrograms 
        @type num_imgs: int
        @return: a list of sample IDs
        @rtype: int
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
        
        # Rough number of images to get per class:
        num_imgs_per_class = round(num_samples_to_get / num_classes)
        _remaining_imgs    = num_samples_to_get % num_classes

        to_get_idxs = []
        for class_idx, sample_idx_list in class_dict.items():
            # Get as many random picks from round's classs
            # sample IDs as we want samples per class:
            
            # Do we have fewer samples in this class than
            # we want from each class?
            if len(sample_idx_list) < num_imgs_per_class:
                # Yes: grab them all:
                to_get_idxs.extend(sample_idx_list)
            else:
                sample_idxs = random.sample(sample_idx_list, num_imgs_per_class)
                to_get_idxs.extend(sample_idxs)

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
