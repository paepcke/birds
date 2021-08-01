'''
Created on May 6, 2021

@author: paepcke
'''
import argparse
import copy
import csv
from enum import Enum
import os
import sys
import warnings

from logging_service.logging_service import LoggingService
from matplotlib import cm as col_map
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, precision_recall_curve
#from sklearn.metrics._classification import precision_score, recall_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing._label import label_binarize
import torch

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


class CELL_LABELING(Enum):

    # The default for the number of species
    # above which confusion matrices will not
    # have cell values written in the cells. 
    # For too many species cells get too small
    # for the numbers:
    CONF_MATRIX_CELL_LABEL_LIMIT=10

    ALWAYS = 0
    NEVER  = 1
    DIAGONAL = 2
    AUTO = CONF_MATRIX_CELL_LABEL_LIMIT  # If dimension > this, no cell lables

# ----------------------- Class Charter --------------

class Charter:

    # The minimum threshold that is 
    # included in PR curves:
    PR_CURVE_MIN_THRESHOLD = 0.1
    
    # Value to assign to precision,
    # recall, or f1 when divide by 0
    DIV_BY_ZERO = 0
    
    log = LoggingService()

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self,
                 actions=None
                 ):
        
        if actions is None:
            return
        
        for action in actions:
            try:
                if type(action) == VizConfMatrixReq:
                    cm = Charter.read_conf_matrix_from_file(action.path)
                    fig = self.fig_from_conf_matrix(cm,
                                                    supertitle=action.supertitle,
                                                    title=action.title,
                                                    write_in_fields=action.write_in_fields
                                                    )
                    fig.show()
                    
                elif action == 'pr_curves':
                    pass #data = 0 #********************
            except Exception as _e:
                pass

    #------------------------------------
    # visualize_testing_result
    #-------------------

    @classmethod
    def visualize_testing_result(cls,
                                 truth_labels,
                                 pred_probs
                                 ):
        '''
        Use to visualize results from using a 
        saved model on a set of test-set samples.
        
        Draws a PR curve, and adds a table with 
        the average precison (AP) of each class.
        
        The truth_labels are expected to be a Pandas
        Series of integers of length num_test_samples.
        
        The pred_probs must be a 1D or 2D dataframe.
        Each row contains the set of probabilities assigned
        to one class. So: width is number of classes,
        height must be num_test_samples, class_id.e. same as
        the length of the truth Series.
                
        :param truth_labels: the class IDs corresponding to 
            the true classification
        :type truth_labels: pd.Series
        :param pred_probs: probabilities assigned for each 
            class_id
        :type pred_probs: pd.DataFrame
        :return the mAP, the number of classes on which the 
            mAP was computed, and a dict mapping class_ids
            to of curve specification instances
        :rtype: ((float, int), {int : CurveSpecification})
        '''

        # Find number of classes involved:
        _num_samples, num_classes = pred_probs.shape
        pred_class_ids = list(range(num_classes))
        
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
                                    classes=pred_class_ids
                                    )

        # The sklearn label_binarize() method returns
        # an  n x c array of 1s and 0s, where n is the number of
        # samples, and c is the number of distinct classes.
        # This is a one-hot encoding. However, when there are
        # only two classes a column vector is returned instead.
        # It has a 1 when a sample is of the first class. When
        # the sample is of the second class, the col vec is 0.
        # Be sure to consider this in any code below.
        
        bin_labels_df = pd.DataFrame(bin_labels) 

        pr_curve_specs = {}

        # Go through each column, class_id, i.e. the
        # 1/0 labels/preds for one class at
        # a time, and get the prec/rec numbers at different
        # thresholds.
        
        for class_id in range(num_classes):
            
            bin_labels_series = bin_labels_df.iloc[:,class_id]
            preds_series  = pred_probs.iloc[:,class_id]
            
            # Get precision and recall at each
            # of the default thresholds:
            pr_curve_spec = \
                cls.compute_binary_pr_curve(bin_labels_series,
                                            preds_series,
                                            class_id)
            
            pr_curve_specs[class_id] = pr_curve_spec

        # Get list of all curves' average precision
        # excluding the ones that are nan:
        all_defined_APs = [pr_curve_spec['avg_prec']
                           for pr_curve_spec
                           in list(pr_curve_specs.values())
                           if not np.isnan(pr_curve_spec['avg_prec'])
                           ]
        mAP = np.mean(all_defined_APs)
        
        return ((mAP, len(all_defined_APs)), pr_curve_specs) 

# ----------------- Computations ---------------


    #------------------------------------
    # compute_binary_pr_curve 
    #-------------------

    @classmethod
    def compute_binary_pr_curve(cls, 
                                truth_series,
                                pred_probs_series,
                                class_id
                                ):
        '''
        Given a series of prediction probabilities for
        a single class (against all others), and corresponding
        1/0 truth values. Return a CurveSpecification instance
        that contains all information needed to draw a PR curve.
        Also includes the optimal operating point for this class,
        i.e. the threshold that leads to the optimal f1.
        
        Each element in pred_probs_series corresponds to a
        classifier's output probability for one test sample.
        The probability is that the sample is of target class 
        class_id (as opposed to some other class). The corresponding
        truth_series element is 1 if the sample truly is of class
        class_id, else the element is 0.
        
          pred_probs_series:                truth_series:
              [for sample0]  0.1              1
              [for sample1]  0.4              0
                  ...                        ...

        

        :param truth_series: 1 or 0 indicating whether or not sample
            at index i is of class class_id
        :type truth_series: pd.Series[int]
        :param pred_probs_series: probabilities that sample at
            index i is of class class_id
        :type pred_probs_series: pd.Series[float]
        :param class_id: the class id for which this 
            PR curve is being computed.
        :type class_id: int
        :return all information needed to draw the PR curve
            for class_id, plus indicator of the optimal operating point
            for class_id
        :rtype: CurveSpecification
        '''

        # The last prec will be 1, the last rec will
        # be 0, and will not have a corresponding 
        # threshold. This ensures that the graph starts
        # on the y axis:
        precs_np, recs_np, thresholds = \
            precision_recall_curve(truth_series, pred_probs_series)

        # Remove NaNs by interpolating from neighboring
        # values. Works for all but NaN at position 0:
        precs_series = pd.Series(precs_np).interpolate()
        recs_series  = pd.Series(recs_np).interpolate()
        
        pr_curve_df = pd.concat([precs_series, recs_series], axis=1)
        pr_curve_df.columns = ['Precision', 'Recall'] 
        thresholds_series = pd.Series(thresholds, name='Threshold')
        
        # Compute f1 for each corresponding precision/recall pair
        # i.e. across each row of pr_curve_df. Take care
        # cases when both recall and precision are zero:
        # Suppress warnings:
        with warnings.catch_warnings():
            try:
                warnings.filterwarnings("error",
                                        #category=UndefinedMetricWarning
                                        category=UserWarning
                                        )
                warnings.filterwarnings("ignore",
                                        #category=UndefinedMetricWarning
                                        category=RuntimeWarning
                                        )
                
                # Fix precision and recall NaN values 
                # by interpolating from their neighbors:
                
                
                f1_scores_series = 2 * (precs_series * recs_series) / (precs_series + recs_series)
                # Add these f1s to the pr_curve info:
                pr_curve_df['f1'] = f1_scores_series
                
                # There is always one fewer thresholds
                # than precision/recall pairs (see above).
                # Make handling the data easier by adding
                # thresholds as a column to pr_curve_df,
                # using NaN for the missing threshold in 
                # the final position. Since by definition,
                # in that row prec==1, rec==0, f1 will be 0,
                # and not the optimal op pt.
                # The ignore_index makes the row number match
                # the row number of the df in the last position:
                
                pr_curve_df['Threshold'] = thresholds_series.append(pd.Series(np.nan), 
                                                                    ignore_index=True)
                
                # Remove rows in which f1 is undefined,
                # b/c both prec and rec were 0:
                pr_curve_df = pr_curve_df.dropna(axis='index', subset=['f1'])

                
            except Exception as e:
                raise type(e)(f"Error during f1 computation: {e}") from e

        avg_prec = average_precision_score(truth_series, pred_probs_series)

        # Create a crv spec instance, but in the 
        # process, remove all rows with threshold
        # values that are overly small, and therefore
        # approach 0, and precision of 1:
        
        res = CurveSpecification(pr_curve_df,
                                 avg_prec,
                                 class_id,
                                 cull_threshold_value=cls.PR_CURVE_MIN_THRESHOLD
                                 )
        return res
    
    #------------------------------------
    # compute_confusion_matrix
    #-------------------
    
    @classmethod
    def compute_confusion_matrix(cls, 
                                 truth_labels, 
                                 predicted_class_ids,
                                 class_names,
                                 normalize=False
                                 ):
        '''
        Example Confustion matrix for 16 samples,
        in 3 classes:
        
                     C_1-pred, C_2-pred, C_3-pred
         C_1-true        3         1        0
         C_2-true        2         6        1
         C_3-true        0         0        3
        
        The number of classes is needed to let 
        sklearn to know even about classes that were not
        encountered.
        
        Assumption: self.class_names contains list 
        of class names, i.e. not the numeric IDs, but the
        ones to use when labeling the matrix.

        :param truth_labels: truth labels as list of class ids
        :type truth_labels: [int]
        :param predicted_class_ids: list of class_ids that were
            predicted, in same order as truth_labels
        :type predicted_class_ids: [int]
        :param class_names: list of class names as known to the
            user, i.e. not the numeric class ints. But the names
            to use as matrix labels in class id order!
        :type class_names: [str]
        :param normalize: whether or not to normalize ROWS
            to add to 1. I.e. turn cells into percentages
        :type normalize: bool
        :return: a dataframe of the confusion matrix; columns 
            and rows (i.e. index) set to class ids
        :rtype: pd.DataFrame 
        '''
        conf_matrix = torch.tensor(confusion_matrix(
            truth_labels,          # Truth
            predicted_class_ids,   # Prediction
            labels=list(range(len(class_names))) # Numeric class ID labels
            ))

        if normalize:
            conf_matrix = cls.calc_conf_matrix_norm(conf_matrix)

        # Turn conf matrix from tensors to numpy, and
        # from there to a dataframe:
        conf_matrix_df = pd.DataFrame(conf_matrix.numpy(),
                                      index=class_names,
                                      columns=class_names
                                      )        
        return conf_matrix_df

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
        percentage of the number of samples for the
        row. If no samples were present for a
        particular class, the respective cells will
        contain -1.
        
        It is assumed that rows correspond to the classes 
        truth labels, and cols to the classes of the
        predictions.
        
        :param conf_matrix: confusion matrix to normalize
        :type conf_matrix: {pd.DataFrame[int] | np.array | torch.Tensor}
        :returned a new confusion matrix with cells replaced
            by the percentage of time that cell's prediction
            was made. Cells of classes without any samples in
            the dataset will contain -1 
        :rtype matches input type
        '''

        # Get the sum of each row, which is the number
        # of samples in that row's class. Then divide
        # each element in the row by that num of samples
        # to get the percentage of predictions that ended
        # up in each cell:
        
        # When a class had no samples at all,
        # there will be divide-by-zero occurrences.
        # Suppress related warnings. The respective
        # cells will contain nan:
        
        with np.errstate(divide='ignore', invalid='ignore'):
            if type(conf_matrix) == np.ndarray:
                return sklearn.preprocessing.normalize(conf_matrix,norm='l1')
            elif type(conf_matrix) == torch.Tensor:
                return conf_matrix.true_divide(torch.sum(conf_matrix, axis=1).unsqueeze(-1))
            elif type(conf_matrix) == pd.DataFrame:
                return conf_matrix.div(conf_matrix.sum(axis='columns'), axis='rows')
            else:
                raise TypeError(f"Matrix must be a dataframe, numpy array, or tensor, not {type(conf_matrix)}")


# ----------------- Visualizations ---------------

    #------------------------------------
    # fig_from_conf_matrix 
    #-------------------
    
    @classmethod
    def fig_from_conf_matrix(cls, 
                             conf_matrix,
                             supertitle='Confusion Matrix\n',
                             subtitle='',
                             write_in_fields=CELL_LABELING.DIAGONAL
                             ):
        '''
        Given a confusion matrix, return a 
        matplotlib.pyplot Figure with a heatmap of the matrix.
        
        The write_in_fields arg controls whether or not
        each cell is filled with a label indicating its
        value. If:
        
            o CELL_LABELING.ALWAYS    : always write the labels
            o CELL_LABELING.NEVER     : never write the labels
            o CELL_LABELING.DIAGONAL  : only label the diagonals
            o CELL_LABELING.AUTO      : only write labels if number of classes
                                        is <= CELL_LABELING.AUTO.value
                
        Result form:
        	             C_1-pred, C_2-pred, C_3-pred
        	 C_1-true        3         1        0
        	 C_2-true        2         6        1
        	 C_3-true        0         0        3
        	         
        
        :param conf_matrix: nxn confusion matrix representing
            rows:truth, cols:predicted for n classes
        :type conf_matrix: pd.DataFrame
        :param supertitle: main title at top of figure
        :type supertitle: str
        :param subtitle: title for the confusion matrix
            only. Ex: "data normalized to percentages"
        :type subtitle: str
        :param write_in_fields: how many cells, if any should 
            contain labels with the cell values. 
        :type write_in_fields: CELL_LABELING
        :return: matplotlib figure with confusion
            matrix heatmap.
        :rtype: pyplot.Figure
        '''

        if type(write_in_fields) != CELL_LABELING:
            raise TypeError(f"Arg write_in_fields must be a CELL_LABELING enum member, not {write_in_fields}")
        
        class_names = conf_matrix.columns
        # Subplot 111: array of subplots has
        # 1 row, 1 col, and the requested axes
        # is in position 1 (1-based):
        # Need figsize=(10, 5) somewhere
        fig, ax = plt.subplots()
        # Make a copy of the cmap, so
        # we can modify it:
        cmap = copy.copy(col_map.Blues)

        fig.set_tight_layout(True)
        fig.suptitle(supertitle, fontsize='large', fontweight='extra bold')

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

        # Add cell labels if requested:
        if write_in_fields == CELL_LABELING.ALWAYS or \
                              (CELL_LABELING.AUTO and len(class_names) <= CELL_LABELING.AUTO.value):
            annot = conf_matrix.copy()
            mask = None
        elif write_in_fields == CELL_LABELING.DIAGONAL:
            # Fill with a copy of conf matrix with strings, 
            # but room for up to 3 chars:
            #**************
            #annot = np.full_like(conf_matrix, '', dtype='U4')
            #np.fill_diagonal(annot, np.diag(conf_matrix).astype(str))
            #annot = np.empty_like(conf_matrix)
            #np.fill_diagonal(annot, np.diag(conf_matrix).astype(str))
            # Fill a new df with True, where df is same
            # dimensions as another df: annot:
            annot = conf_matrix.copy()
            mask = pd.DataFrame(np.array([True]*annot.size).reshape(annot.shape))
            mask.index    = conf_matrix.index
            mask.columns  = conf_matrix.columns
            np.fill_diagonal(mask.values, False)
            
            
            #**************
        else:
            annot = None
            mask  = None

        cmap.set_bad('gray')

        heatmap_ax = sns.heatmap(
            conf_matrix,
            cmap=cmap,
            xticklabels=True,
            yticklabels=True,
            square=True,
            annot=annot,  # Cell labels
            mask=mask,
            fmt='.0%',    # Round to int, multiply by 100, add '%'
            cbar=True,    # Do draw color bar legend
            ax=ax,
            linewidths=1,# Pixel,
            linecolor='gray',
            robust=True   # Compute colors from quantiles istead of 
                          # most extreme values
            )
        
        # Add '%' after cell numbers; note that fmt='d%' 
        # leads to an error; I suspect there is a seaborn
        # heatmap fmt value that would add '%', but I don't
        # have time for frigging format strings:
        
        #*****for txt in heatmap_ax.texts: txt.set_text(txt.get_text() + " %")

        heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), 
                                   rotation = 45 
                                   )
        
        heatmap_ax.set_title(subtitle,
                             fontdict={'fontsize'   : 'medium',
                                       'fontweight' : 'bold',
                                 },
                             pad=12 # Distance above matrix in pt
                             )
        
        # Label x and y to clarify what's predicted,
        # and what's truth; also: have the x-axis label
        # at the top:
        
        heatmap_ax.set_xlabel('True Classes', fontweight='bold')
        heatmap_ax.xaxis.set_label_position('top')
        heatmap_ax.set_ylabel('Predicted Classes', fontweight='bold')

        fig = heatmap_ax.get_figure()
        fig.suptitle(supertitle, fontsize='large', fontweight='extra bold')
        
        return fig
    
# -------------------- Utilities for Charter Class --------------


    #------------------------------------
    # read_conf_matrix_from_file
    #-------------------
    

    @classmethod
    def read_conf_matrix_from_file(cls, cm_path):
        '''
        Read a previously computed confusion matrix from
        file. Return a dataframe containing the cm.
        
        NOTE: this method is for reading files saved to csv
              by numpy or pandas.

        NOTE: if arrays of predicted and truth classes are
              available, rather than an already computed confusion
              matrix saved to file, see compute_confusion_matrix()
              
        NOTE: to read raw results saved during training (usually into
              src/birdsong/runs_raw_results), use: 
              confusion_matrices_from_raw_results(), followed by compute_confusion_matrix()
        
        Depending on the original dataframe/tensor,np_array
        from which which the .csv was created, the first line
        has a leading comma. This results in:
        
              Unnamed: 0  foo  bar  fum
            0        foo    1    2    3
            1        bar    4    5    6
            2        fum    7    8    9        
        
        Rather than the correct:
        
                 foo  bar  fum
            foo    1    2    3
            bar    4    5    6
            fum    7    8    9

        Since conf matrices are square, we can check
        and correct for that.
        
        NOTE:  
        
        :param cm_path: path to confusion matrix in csv format
        :type cm_path: str
        :return: confusion matrix as dataframe; no processing on numbers
        :rtype: pd.DataFrame
        '''
        
        df = pd.read_csv(cm_path)
        # If comma was missing, we have one fewer
        # col names than row names:
        if len(df.columns) != len(df.index):
            df_good = df.iloc[:, 1:]
            df_good.index = df.columns[1:]
        else:
            df_good = df

        return df_good

    #------------------------------------
    # confusion_matrices_from_raw_results()
    #-------------------
    
    @classmethod
    def confusion_matrices_from_raw_results(cls, fname, class_names=None, normalize=False):
        '''
        Read csv prediction/truth files created by the training 
        process. The files are by default written to
        src/birdsong/runs_raw_results.
        
        File format:
        
			step,train_preds,train_labels,val_preds,val_labels
			0,"[2, 2, 3, 2, 3]","[2, 3, 1, 1, 1]","[4, 4, 4, 4, 4,]","[3, 4, 4, 2, 2]"
			2,"[5, 5, 4, 5, 4]","[2, 3, 2, 2, 2]","[4, 4, 4, 4, 4]","[2, 4, 4, 1, 1]"

        
        Target cass names: if provided, the value should be an 
        array of strings with human-readable class names, or a list 
        of integers that correspond to target classes. If None, the
        method looks for a file class_names.txt in fname's directory,
        where it expects to find a single row like:
        
           ['DYSMEN_S', 'HENLES_S', 'audi', 'bmw', 'diving_gear', 'office_supplies']
            
        If that file is unavailable, and class_names is None: ValueError.
        
        Returns a list of TrainHistoryCMs instances whose API is:
        
              <inst>.step         ---> int
              <inst>.training     ---> dataframe confusion matrix
              <inst>.validation   ---> dataframe confusion matrix

        :param fname: name of csv file to read
        :type fname: str
        :param class_names: if provided, an array of strings with
            human-readable class names, or a list of integers 
            that correspond to target classes.
        :type class_names: {[str] | [int]}
        :param normalize: normalize: whether or not to normalize ROWS
            to add to 1. I.e. turn cells into percentages
        :type normalize: bool
        :returns: list of TrainHistoryCMs instances, each 
            containing one training step's train and validation
            outcome as dataframes
        :rtype: [TrainHistoryCMs]
        :raise ValueError if class_names cannot be found
        :raise FileNotFoundError if fname does not exist
        '''
        
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Cannot find file {fname}")

        if class_names is None:
            # Get class names from file, if available:
            class_name_file = os.path.join(os.path.dirname(fname), 
                                           'class_names.txt')
            if os.path.exists(class_name_file):
                with open(class_name_file, 'r') as fd:
                    class_names_str = fd.read()
                # Now have something like:
                #
                #   "['DYSMEN_S', 'HENLES_S', 'audi', 'bmw', 'diving_gear', 'office_supplies']\n"
                #
                # Use eval to retrieve the list, but 
                # do it safely to avoid attacks by content
                # placed in the class_names.txt file:
    
                class_names = eval(class_names_str,
                   {"__builtins__":None},    # No built-ins at all
                   {}                        # No additional func
                   )
            else:
                raise ValueError("Must provide class names as argument, or in class_name.txt file")

        results = []
        # Get one row with step, train-preds, train_labels,
        # val_preds, and val_labels at a time:
        with open(fname, 'r') as data_fd:
            reader = csv.DictReader(data_fd)
            for outcome_dict in reader:
                # Safely turn string formatted
                # lists into lists of ints:
                train_labels = eval(outcome_dict['train_labels'],
                   {"__builtins__":None},    # No built-ins at all
                   {}                        # No additional func
                   )
                train_preds = eval(outcome_dict['train_preds'],
                   {"__builtins__":None},    # No built-ins at all
                   {}                        # No additional func
                   )
                train_preds_df = cls.compute_confusion_matrix(train_labels, 
                                                              train_preds, 
                                                              class_names, 
                                                              normalize)

                val_labels = eval(outcome_dict['val_labels'],
                   {"__builtins__":None},    # No built-ins at all
                   {}                        # No additional func
                   )
                val_preds = eval(outcome_dict['val_preds'],
                   {"__builtins__":None},    # No built-ins at all
                   {}                        # No additional func
                   )

                val_preds_df = cls.compute_confusion_matrix(val_labels,
                                                            val_preds,
                                                            class_names,
                                                            normalize)
                results.append(TrainHistoryCMs(outcome_dict['step'], 
                                                 train_preds_df, 
                                                 val_preds_df,
                                                 normalize))
        return results

    #------------------------------------
    # barchart 
    #-------------------
    
    @classmethod
    def barchart(cls, 
                 data, 
                 rotation=0,
                 ylabel=None, 
                 color_groups=None,
                 ):
        '''
        Returns a matplotlib axes with a bar chart
        that can be added to a figure. The data is
        a pandas Series with column labels like:
        
                'prec_micro'
                'prec_macro'
                'f1'
                'accuracy_balanced'
                    ...
                    
        The axes used is the present matplotlib axes.
        This may be the default axes that is implicitly
        created by a simple:
           
           fig = plt.figure()
           
        Example:
        
           fig = plt.figure()
           data = pd.Series([1,2,3], index=['foo','bar','fum'])
           Charter.barchartdata) 
           
        The color_groups allows coloring related bars
        the same. Example: if bars include micro-precision,
        macro-precision, micro-recall, and macro-recall, one
        might want the precisions bars to have the same color,
        and a different color for the recalls. Example:
        
          color_groups: {'green' : ['micro_prec', macro_prec'],
                         'brown' : ['micro_recall', macro_recall'],
                         'blue'  : ['accuracy']
                         }
        
        :param data: values to plot
        :type data: ordinal values
        :param rotation: rotation of x labels in degrees; ex: 45
        :type rotation: int
        :param ylabel: y axis label; None is OK
        :type ylabel: str
        :param color_groups: groupings of colors for the bars
        :type color_groups: {str : [str]}
        :return axes with chart
        :rtype matplotlib.axes
        '''
        
        ax = data.plot.bar()
        plt.tight_layout()
        if rotation != 0:
            ax.set_xticklabels(data.index, rotation=rotation)
            
        if ylabel is not None:
            ax.set_ylabel(ylabel)
            
        if color_groups is not None:
            # Make the bars of related values the
            # same color: the bar chart's X labels 
            # are the names of the measures. They are
            # each contained in a text object; got those
            # out into a list: 
            #   ['prec_macro', 'prec_micro', 'prec_weighted', 'recall_macro',...]
            bars = ax.get_children()
            measures = [txt_obj.get_text() for txt_obj in ax.get_xticklabels()]
            for color in color_groups.keys():
                for like_colored_meas in color_groups[color]:
                    try:
                        bar_idx = measures.index(like_colored_meas)
                        bars[bar_idx].set_color(color)
                    except ValueError:
                        raise ValueError(f"Could not find {like_colored_meas} in barplot X labels")

        return ax


# ------------------ Class TrainHistoryCMs -----------

class TrainHistoryCMs:
    '''
    Instances hold a training step (int), and
    two confusion matrices as dataframes: the step's
    training outcome CM, and the corresponding validation
    CM. 
    
    Attributes: step, training, validation, normalized
    '''
    
    def __init__(self, step, train_cm, val_cm, normalized):
        self.step = int(step)
        self.training   = train_cm
        self.validation = val_cm
        self.normalized = normalized

# ------------------ Class Best Operating Point -----------

class BestOperatingPoint(dict):
    '''
    Instances hold information about the 
    optimal point on a PR curve. That is
    the decision threshold where the corresponding
    recall/precision yields the maximum F1 score
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, op_pt_info):
        '''
        A pd series containing:
			 Threshold    0.000853
			 Precision    0.115044
			 Recall       0.928571
			 f1           0.204724
        
        where:
            Threshold is probability value that is the best
                      decision threshold between two classes
            Precision is the precision value of the 
                      best operating point on the PR curve
            Recall    is the recall value of the 
                      best operating point on the PR curve
            f1        is the f1 score computed from the given
                      precision/recall
        
        :param op_pt_info: the parameters of the operating point
            precision, recall, f1, threshold
        :type op_pt_info: pd.Series
        '''
        super().__init__()

        f1_score  = op_pt_info.f1
        precision = op_pt_info.Precision
        recall    = op_pt_info.Recall
        threshold = op_pt_info.Threshold

        self.__setitem__('threshold', threshold)
        self.__setitem__('f1', f1_score)
        self.__setitem__('precision', precision)
        self.__setitem__('recall', recall)

    #------------------------------------
    # __repr__ 
    #-------------------
    
    def __repr__(self):
        uniq_id = hex(id(self))
        f1 = round(self.__getitem__('f1'), 2)
        rep = f"<BestOpPoint f1={f1} {uniq_id}>"
        return(rep)

    #------------------------------------
    # __str__ 
    #-------------------
    
    def __str__(self):
        return repr(self)


# --------------------- Class CurveSpecification ----------------

class CurveSpecification(dict):
    '''
    Instances hold information about one 
    classifier evaluation curve. Enough
    information is included to draw 
    a precision-recall curve.
    
    Instances behave like dicts with keys:
    
         best_op_pt:        {'threshold' : thresholds[best_op_idx],
                             'f1'        : f1_scores[best_op_idx],
                             'precision' : precisions[best_op_idx],
                             'recall'    : recalls[best_op_idx]
                             }
         recalls          : list of recall values
         precisions       : list of precision values
         thresholds       : list of probability decision thresholds
                            at which precions/recall pairs were computed
         f1               : list of f1 values
         avg_precision    : the Average Precision (AP) of all the points
         
         class_id         : ID (int or str) for which instances is a curve
         
         curve_df         : all recall/precision/threshold/f1
                            values in one dataframe
         
            
        The best_op_pt is an instance of BestOperatingPoint, which
        contains the precision, recall, f1, and threshold for 
        which f1 is optimal.
         
        The curve_df holds for each decision threshold
        the probability threshold value, the corresponding
        precision and recall, and the f1 value:
        
            ['Threshold', 'Precision', 'Recall', 'f1']

        The recalls/precision/threshold lists are extracted
        from curve_df.

        Instance initialization includes computation of the
        best operating point, which is the threshold at which
        the given class yields the best f1. 

        In addition, methods:
        
            copy
            undef_prec
            undef_rec
            undef_f1
            
        return the number of undefined (nan)
        precisions, recalls, and f1 values, respectively.
        These values can be use to discard curves that
        include nan values.
        	
        It is recommended that the caller removes
        such NaN rows ahead of time.

    
    NOTE: For now equality (__eq__) is implemented
          as 'identical'. I.e. the very same id number
    '''
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self,
                 pr_curve_df,
                 avg_precision,
                 class_id,
                 cull_threshold_value=False,
                 **kwargs
                 ):
        '''
        The pr_curve_df's columns must be:
        ['Threshold', 'Precision', 'Recall', 'f1']
        
        The avg_precision must be the AP computed from
        the original data, for instance via sklearn.average_precision_score()
        
        The class_id is the integer target class ID for
        which this curve is the PR-curve. 

        If cull_threshold_value is provided, it must be a float. In that
        case all rows for which threshold is less than the
        given value are removed before the best operating
        point is computed. Subsequent retrievals such as
        crv_spec_inst['recalls'] will not include the culled
        rows.

        :param pr_curve_df: dataframe with columns
            ['Precision', 'Recall', 'f1', 'Threshold']
        :type pr_curve_df: pd.DataFrame
        :param avg_precision: AP summarizes a precision-recall curve as 
            the weighted mean of precisions achieved at each threshold, 
            with the increase in recall from the previous threshold 
            used as the weight:
               SUM_n((R_n - R_(n-1)) * P_n)
        :type avg_precision: [float]
        :param class_id: class for which this obj
            is the pr-curve
        :type class_id: {int | str}
        :param cull_threshold_value: if True, all prec/recall pairs for
            thresholds < the given value are removed.
        :type cull_threshold_value: float
        '''
        
        super().__init__()
        if cull_threshold_value is not None:
            
            if type(cull_threshold_value) != float:
                raise TypeError(f"The culling threshold value must be a float, not {cull_threshold_value}")
            
            self.pr_curve_df = self._cull_low_thresholds(pr_curve_df)
        else:
            self.pr_curve_df = pr_curve_df
        
        self.__setitem__('curve_df', self.pr_curve_df)
        self.__setitem__('best_op_pt', self._compute_best_op_pt())
        self.__setitem__('avg_prec', avg_precision)
        self.__setitem__('class_id', class_id)
        self.update(**kwargs)

    #------------------------------------
    # well_defined 
    #-------------------
    
    def well_defined(self):
        '''
        Returns true if all this curve's precision and recall
        points are non-zero, and the best operating point is
        well defined.
        '''
        
        status_all_points = self.undef_prec() or \
                            self.undef_rec() or \
                            self.undef_f1() or \
                            np.isnan(self['avg_prec'])
        if status_all_points > 0:
            return False
        # For now, assume that the best operating
        # point is well defined, b/c all precs/recs/f1
        # values are OK. One alternative would be to
        # declare BOPs with threshold 1 to be ill defined.
        # But leaving that judgement up to clients.
        
        return True

    #------------------------------------
    # _cull_low_thresholds
    #-------------------
    
    def _cull_low_thresholds(self, crv_df):
        '''
        Takes a df ('Precision', 'Recall', 'f1', 'Threshold').
        Removes rows with very low thresholds, subject to 
        at least 10 rows left. 
        
        :param crv_df: curve information
        :type crv_df: pd.DataFrame
        :return: new dataframe with rows culled
        :rtype: pd.DataFrame
        '''
        min_rows = 10
        num_rows, _num_cols = crv_df.shape
        if num_rows <= min_rows:
            return crv_df
        
        new_df = crv_df.copy()
        cur_cutoff = Charter.PR_CURVE_MIN_THRESHOLD
        
        while True:
            candidate = new_df[new_df['Threshold'] >= cur_cutoff]
            new_num_rows, _new_num_cols = candidate.shape 
            if new_num_rows >= min_rows:
                return candidate
            # Getting too few rows; lower the cutoff a bit:
            cur_cutoff -= 0.01

    #------------------------------------
    # copy 
    #-------------------

    def copy(self, pr_curve_df=None, **kwargs):
        '''
        Special copy method with an optional
        replacement of pr_curve_df. The new
        copy will contain all the same values,
        except that the given pr_curve_df is
        replaced. If none is provided, the copy
        will contain a copy of the this instance's
        pr_curve_df.
        '''
        
        if pr_curve_df is None:
            pr_curve_df = self.pr_curve_df.copy()

        new_inst = CurveSpecification(pr_curve_df,
                                      self['avg_precision'],
                                      self['class_id'],
                                      **kwargs
                                      )
    
        return new_inst
    
    #------------------------------------
    # __getitem__ 
    #-------------------
    
    def __getitem__(self, key):
        if key == 'recalls':
            return self.pr_curve_df['Recall']
        elif key == 'precisions':
            return self.pr_curve_df['Precision']
        elif key == 'thresholds':
            return self.pr_curve_df['Threshold']
        elif key == 'f1_scores' or key == 'f1':
            return self.pr_curve_df['f1']
        else:
            return super().__getitem__(key)

    #------------------------------------
    # undef_prec 
    #-------------------

    def undef_prec(self):
        '''
        Returns the number of precision values
        that are 0
        '''
        return sum(self['precisions'].isna())
        
    #------------------------------------
    # def undef_prec 
    #-------------------

    def undef_rec(self):
        '''
        Returns the number of recall values
        that are 0
        '''
        return sum(self['recalls'].isna())
    
    #------------------------------------
    # def undef_f1
    #-------------------

    def undef_f1(self):
        '''
        Returns the number of undefined (nan)
        f1 values
        '''
        return sum(self['f1_scores'].isna())

    #------------------------------------
    # __repr__ 
    #-------------------
    
    def __repr__(self):
        uniq_id = hex(id(self))
        AP = self.__getitem__('avg_prec')
        rounded_AP = round(AP, 2)
        rep = f"<CurveSpecification AP={rounded_AP} {uniq_id}>"
        return(rep)

    #------------------------------------
    # __str__ 
    #-------------------
    
    def __str__(self):
        return repr(self)

    #------------------------------------
    # __eq__ 
    #-------------------

    def __eq__(self, other):
        return id(self) == id(other)
    
    #------------------------------------
    # __neq__ 
    #-------------------

    def __neq__(self, other):
        return not self.__eq__(other)
    
    #------------------------------------
    # _compute_best_op_pt 
    #-------------------
    
    def _compute_best_op_pt(self):
        best_op_idx = self.pr_curve_df['f1'].argmax()
        optimal_parms_series = self.pr_curve_df.iloc[best_op_idx,:]
        # A nice object containing all about the optimal point:
        best_operating_pt = BestOperatingPoint(optimal_parms_series)
        return best_operating_pt

    

# ---------------------- Classes to Specify Visualization Requests ------

# The following (sub)classes are just information holders
# that are passed to the __init__() method of Charter to 
# keep args for different visualizations tidy:

class VizRequest:
    
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        
        self.path = path
        
class VizConfMatrixReq(VizRequest):
    
    def __init__(self, 
                 path, 
                 write_in_fields=CELL_LABELING.DIAGONAL, 
                 supertitle='Confusion Matrix', 
                 title=''):
        
        super().__init__(path)
        
        if type(write_in_fields) != CELL_LABELING:
            raise TypeError(f"Confusion matrix cell label handling specification must be a CELL_LABELING, not {type(write_in_fields)}")
        self.write_in_fields = write_in_fields
        
        self.supertitle = str(supertitle)
        self.title = str(title)
        
class VizPRCurvesReq(VizRequest):
    def __init__(self, 
                 path
                 ): 
        super().__init__(path)

# --------------- Main -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Charts and other analyses from inferencing."
                                     )

    parser.add_argument('--supertitle',
                        help='title above the figure',
                        default=None
                        )

    parser.add_argument('--title',
                        help='title just above the main chart of the figure',
                        default=None
                        )
    
    parser.add_argument('--conf_matrix',
                        help='heatmap confusion matrix; value: path to csv file',
                        default=None
                        )
    parser.add_argument('--pr_curves',
                        help='draw (family of) PR curves; value: path to csv file',
                        default=None
                        )

    args = parser.parse_args()
    
    actions = []
    if args.conf_matrix is not None:
        actions.append(VizConfMatrixReq(path=args.conf_matrix,
                                        supertitle=args.supertitle,
                                        title=args.title,
                                        write_in_fields=CELL_LABELING.DIAGONAL
                                        ))
    if args.pr_curves is not None:
        actions.append(VizPRCurvesReq(path=args.pr_curves))
        
    Charter(actions)