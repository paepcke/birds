'''
Created on Jan 25, 2021

@author: paepcke
'''
import os
import random
import warnings

from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm as col_map
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics._classification import precision_score, recall_score
from sklearn.metrics._ranking import average_precision_score
from sklearn.preprocessing._label import label_binarize
import torch
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.utils import make_grid

from birdsong.rooted_image_dataset import SingleRootImageDataset
from birdsong.utils.github_table_maker import GithubTableMaker
from birdsong.utils.learning_phase import LearningPhase
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
    # visualize_testing_result 
    #-------------------
    
    @classmethod
    def visualize_testing_result(cls,
                                 truth_labels, 
                                 pred_class_ids
                                 ):
        '''
        Use to visualize results from using a 
        saved model on a set of test-set samples.
        
        Draws a PR curve, and adds a table with 
        the average precison (AP) of each class.
        '''
        # Find number of classes involved:
        all_class_ids = set(truth_labels)
        num_classes   = len(all_class_ids)
        
        # Will alternately treat each class 
        # prediction as a one-vs-all binary
        # classification. For each class ID (cid<n>), 
        # get 0/1 guess separately for each sample:
        # 
        #                 cid0      cid1 
        #   pred_sample0   1          0 
        #   pred_sample1   0          0
        #   pred_sample2   0          1
        #             ...
        # Same with labels:
        #                 cid0      cid1 
        #   labl_sample0   1          0 
        #   labl_sample1   0          0
        #   labl_sample2   0          1
        #             ...

        bin_labels = label_binarize(truth_labels,
                                    classes=list(range(num_classes)))
        
        bin_preds  = label_binarize(pred_class_ids,
                                    classes=list(range(num_classes)))

        # Make tensors just for manipulation
        # convenience:
        
        bin_labels_tn = torch.tensor(bin_labels)
        bin_preds_tn  = torch.tensor(bin_preds)
        
        precisions = dict()
        recalls = dict()
        average_precisions = dict()

        # Go through each column, i.e. the
        # 1/0 labels/preds for one class at
        # a time, and get the prec/rec numbers.
        # The [1] in prec & rec is b/c precision_recall_curve
        # returns a triplet for binary classification:
        # prec/rec at thresholds 0, 1, putting 1 as the 
        # last element. The prec/rec we want is the 
        # where 1 is the thresholds:
        
        for i in range(num_classes):
            
            bin_labels_arr = bin_labels_tn[:,i].tolist()
            bin_preds_arr  = bin_preds_tn[:,i].tolist()
            
            # Get precision and recall at each
            # of threshold = [0.2, 0.4, 0.6, 0.8, 1.0]
            precs, recs = \
                cls.compute_binary_pr_curve(bin_labels_arr,
                                     bin_preds_arr
                                     )
            precisions[i] = precs
            recalls[i]    = recs
            
            # Avg prec is:
            #
            #      AP = SUM_ovr_n((R_n - R_n-1)*P_n
            #
            # I.e. the increase in recalls times current
            # precisions as each pred/sample pair is 
            # processed:
            
            average_precisions[i] = \
                average_precision_score(bin_labels_arr,
                                        bin_preds_arr,
                                        average='macro',
                                        )
                
        mAP = np.mean(list(average_precisions.values()))
        
        return (mAP, 
                precisions, 
                recalls, 
                average_precisions
                )


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
        sample_idxs = cls._get_sample_indices(img_folder,
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
    # compute_multiclass_pr_curve 
    #-------------------
    
    @classmethod
    def compute_multiclass_pr_curve(cls,
                                    truth_labels, 
                                    raw_preds,
                                    thresholds=[0.2, 0.4, 0.6, 0.8, 1.0]
                                    ):
        '''
        Computes the data needed to draw
        a family of PR curves for the results
        of multiclass classifier output.
        
        Returns a dict of the constituent 
        single-class curve specs, and a
        mean average precision (mAP) score
        for all curves combined.
        
        Each result dict maps a class ID
        to all info needed for one of the
        curves:

          1:
              {'best_op_pt' : best_operating_pt,
               'precisions' : precisions,
               'recalls'    : recalls,
               'thresholds' : thresholds,
               'avg_prec'   : avg_precision
               }
          2:
              {'best_op_pt' : best_operating_pt,
               'precisions' : precisions,
               'recalls'    : recalls,
               'thresholds' : thresholds,
               'avg_prec'   : avg_precision
               }

        where best_op_pt is:

               {'threshold' : <optimal decision probability value>
                'f1'        : <f1 at the optimal threshold>
                'prec'      : <precision at the optimal threshold>
                'thresholds' : thresholds,
                'rec'       : <recall at the optimal threshold>
                }

        Each of the avg_prec is the 
        the average of precisions across the 
        samples of one class (AP). I.e. there will
        be as many elements in average_precisions
        as there are classes. 
        
        The Mean Average Precision (mAP) is 
        the mean of the average_precision values.
        This measure summarizes the family of PR curves.
        It is comparable to AUC ROC.
        
        The precisions and recalls returns are dicts.
        The keys are class IDs, and the values are the
        precisions for that class. They are the quantities
        from which the average_precision values are 
        computed.
        
        Summary: 
            o precisions/recalls are the lowest granularity
              of information: the per class precs and recs
              at different thresholds.
              
              There are as many entries in these dicts as
              there are classes. And prec/rec value pair
              from the precisions and recalls dict are results
              of one threshold. 

               TODO: o finish this sentence by running and
                       seeing what's what
                     o A unit test for this method
                     o Finally: the actual drawing of the 
                         curves with pyplot
                         
            o average_precision aggregates the precisions
              of one class across multiple thresholds. There 
              will be as many entries in this dict as there 
              are classes.
              
            o mAP aggregates the average_precision values
              across all classes. This is one number.

        @param truth_labels: all truth labels shaped
            torch.Size([num-batches, batch-size])
        @type truth_labels: Tensor
        @param raw_preds: the logits for each class for
            each sample as 
            torch.Shape([num-batches, batch-size, num-classes])
        @type raw_preds: Tensor
        @return: (precisions, recalls, average_precisions, mAP)
        @rtype: ({int : [floats]}, {int : [floats]}, [floats], float)
        '''
        # REMOVE
#         # Find number of classes involved:
#         all_class_ids = set([tn.item() 
#                              for tn 
#                              in torch.flatten(truth_labels)])

        (num_batches, 
         batch_size, 
         num_classes) = raw_preds.shape
        
        num_samples = num_batches * batch_size
        
        # Will alternately treat each class 
        # prediction as a one-vs-all binary
        # classification.
        #
        # Ex. let labels = [1,0,0,1,2]
        #      and preds = [0.3,0.6,0.1,0.7,0.9]
        #
        # Convert the labels to a one-hot vector;
        # the width of the binarized labels is 
        # num_classes:
        #
        #       L A B E L S               P R E D S
        #       ------------              ----------
        #     [1,         [[0, 1, 0],       [0.3,
        #      0,          [1, 0, 0],        0.6,
        #      0,   ==>    [1, 0, 0],        0.1,
        #      1,          [0, 1, 0],        0.7,
        #      2]          [0, 0, 1]]        0.9]
        #
        # Then evaluate each label column vector 
        # separately.
        
        bin_labels = label_binarize(truth_labels.flatten(),
                                    classes=list(range(num_classes)))

        assert(bin_labels.shape == torch.Size([num_samples, num_classes])) 
        assert(raw_preds.shape == \
               torch.Size([num_batches, batch_size, num_classes])
               )
        
        # Want straight down: logits for each class, for
        # each sample ('lst' for 'list'):
        
        raw_preds_lst = raw_preds.reshape([num_samples, num_classes])

        assert(raw_preds_lst.shape == bin_labels.shape)
        
        # Turn logits into probs, rowise:
        preds = torch.softmax(raw_preds_lst, dim=1) 

        # Place to hold the result dicts 
        # from compute_binary_pr_curve()
        # for each of the classes. This
        # will be class-name : binary-result-dict
        
        all_curves_info = {}
        
        # Go through each column, class_id i.e. the
        # 1/0-vector label columns and preds 
        # columns for one class at
        # a time, and get the prec/rec numbers.

        for col_idx in range(num_classes):
            bin_label_col = torch.tensor(bin_labels[:,col_idx])
            preds_col     = preds[:,col_idx]

            # Get all info for this single, binary
            # classification: list of 1/0 labels, and
            # list of floats, which are the preds for
            # the current class:
            
            one_class_curve = cls.compute_binary_pr_curve(bin_label_col,
                                                          preds_col,
                                                          thresholds)
            
            # Series of precs/recs at the 
            # different thresholds for the 
            # class in this loop
            all_curves_info[col_idx] = one_class_curve

        avg_precs = [binary_curve_info['avg_prec']
                     for binary_curve_info
                     in all_curves_info.values()
                     ]
        mAP = np.mean(np.array(avg_precs)).tolist()
        
        return (all_curves_info, mAP) 

    #------------------------------------
    # compute_binary_pr_curve 
    #-------------------

    @classmethod
    def compute_binary_pr_curve(cls, 
                                labels, 
                                preds, 
                                thresholds=None,
                                ):
        '''
        Return the recall (x-axis) and precision (y-axis)
        values of a PR curve, its average precision (AP),
        and a dict specifying the optimal threshold, 
        with corresponding f1, precision, and recall values:
        
        best_op_pt: 
            {'threshold' : <optimal decision probability value>
             'f1'        : <f1 at the optimal threshold>
             'prec'      : <precision at the optimal threshold>
             'rec'       : <recall at the optimal threshold>
            }

        The result is packaged as a dict:
        
        res = {'best_op_pt' : best_operating_pt,
               'precisions' : precisions,
               'recalls'    : recalls,
               'thresholds' : thresholds,
               'avg_prec'   : avg_precision
               }

        Procedure:
        
        A prec/rec point is computed for each 
        threshold point. 
        
        Works for binary classification.
        But can use sklearn's label_binaries to compute 
        separate curves for each class 
        (see compute_multiclass_pr_curve())
        
        Differs from sklearn.precision_recall_curve() in
        that the sklearn method does not take a list
        of thresholds.  
        
        Example:
        (preds are probabilities, but they
         are from one class, different samples.
         So dont' add to 1):
        
               labels  = [1,1,0,1]
               preds  = [0.2, 0.4, 0.1, 0.2] 
          thresholds  = [0.3, 0.7]
          
          The predictions are turned into decisions like this:
               preds_decided_0.3 = [0, 1, 0, 0]
               preds_decided_0.5 = [0, 0, 0, 0]
          
          Two prec and rec computations are executed:
          
            pr0:  prec and rec from [1, 1, 0, 1] 
                                    [0, 1, 0, 0]
        
            pr1:  prec and rec from [1, 1, 0, 1]
                                    [0, 0, 0, 0]

           resulting in:
              precs = [p0, p1]
              recs  = [r0, r1]

          F1 values fs = [f0, f1] are computed for p0/r0,
          and p1/r1. The position idx (argmax) of 
          the highest f1 is determined. 
          
          best_op_pt = {
             'threshold' : thresholds[idx], 
             'f1'        : fs[idx], 
             'prec'      : precs[idx] 
             'rec'       : recs[idx]
            }

          Finally the average precision (AP) is
          computed. It derives from precs and recs:
          
          for k=0 to k=n-1
          AP = sum_ovr_k((recs_k - recs_[k-1]) * preds_k)
          
          where n is number of thresholds, 
          recs_k and precs_k are precision and 
          recall at the kth threshold. By definition,
          preds_n = 1, recs_n = 0.

          Returned:
              {'best_op_pt' : best_operating_pt,
               'precisions' : precs,
               'recalls'    : recs
               'avg_prec'   : AP
               }

                  
        @param labels: integer binary class labels.
            Exs.: [1,1,0,0], ['yes', 'yes', 'no', 'yes']
        @type labels: [int | str]
        @param preds: predictions output from a classifier.
            May be floats or integers
        @type preds: [float | int]
        @param thresholds: list of decision thresholds to
            decide whether preds are one class or the other.
            If None, uses [0.2, 0.4, 0.6, 0.8, 1]
        @type thresholds: [float | int]
        @return: dict with optimal operating point,
            and lists with prec and recall ready 
            for drawing a PR curve
        @rtype: ({str : float}, [float], [float])
        @raises ValueError if labels hold more than 
            two distinct values
        '''
        uniq_classes = set(labels.tolist())
        if len(uniq_classes) > 2:
            raise ValueError(f"Labels limited to up to two distinct values; got {uniq_classes}")

        if thresholds is None:
            thresholds = [0.2, 0.4, 0.6, 0.8, 1]
        precisions = []
        recalls = []
        class_list = list(uniq_classes)
        
        for threshold in thresholds:
            y_pred = []
            for pred in preds:
                if pred >= threshold:
                    y_pred.append(class_list[1])
                else:
                    y_pred.append(class_list[0])
    
            y_pred_tn = torch.tensor(y_pred)
            precision = precision_score(y_true=labels, 
                                        y_pred=y_pred_tn, 
                                        pos_label=class_list[1])
            
            recall    = recall_score(y_true=labels, 
                                     y_pred=y_pred_tn, 
                                     pos_label=class_list[1])
            
            precisions.append(precision)
            recalls.append(recall)

        precs_np = np.array(precisions)
        recs_np  = np.array(recalls)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    category=RuntimeWarning,
                                    message='invalid value encountered in true_divide'
                                    )
            f1_scores = 2 * (precs_np * recs_np) / (precs_np + recs_np)
            # When both prec and recall are 0,
            # set f1 to zero:
            np.nan_to_num(f1_scores, copy=False)
            
        best_op_idx = np.argmax(f1_scores)
        
        best_operating_pt = {'threshold' : thresholds[best_op_idx],
                             'f1'        : f1_scores[best_op_idx],
                             'prec'      : precisions[best_op_idx],
                             'rec'       : recalls[best_op_idx]
                             }

        # To make average_precision computation
        # work:
        recs_np_padded = np.append(recs_np, [0])
        precs_np_padded = np.append(precs_np, [1])
        
        avg_precision = \
            np.sum((recs_np_padded[:-1] - recs_np_padded[1:]) * precs_np_padded[:-1])
        
        res = {'best_op_pt' : best_operating_pt,
               'precisions' : precisions,
               'recalls'    : recalls,
               'thresholds' : thresholds,
               'avg_prec'   : avg_precision
               }
        return res

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
    # _get_sample_indices 
    #-------------------
    
    @classmethod
    def _get_sample_indices(cls,
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
        if not isinstance(hparam_dict, dict) or \
            not isinstance(metric_dict, dict):
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)
