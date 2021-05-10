'''
Created on May 6, 2021

@author: paepcke
'''
import argparse
import copy
import os
import sys
import warnings

from logging_service.logging_service import LoggingService
from matplotlib import cm as col_map
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics._classification import precision_score, recall_score
from sklearn.metrics._ranking import average_precision_score
from sklearn.preprocessing._label import label_binarize
import torch

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


class Charter:
    
    # Value to assign to precision,
    # recall, or f1 when divide by 0
    DIV_BY_ZERO = 0
    
    log = LoggingService()

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

        # Make tensors just for manipulation
        # convenience:
        
        bin_labels_tn = torch.tensor(bin_labels)
        preds_tn      = torch.tensor(pred_class_ids)
        
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
            preds_arr  = preds_tn.tolist()
            
            # Get precision and recall at each
            # of the default thresholds:
            precs, recs = \
                cls.compute_binary_pr_curve(bin_labels_arr,
                                            preds_arr
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
                                        preds_arr,
                                        average='macro',
                                        )
                
        mAP = np.mean(list(average_precisions.values()))
        
        return (mAP, 
                precisions, 
                recalls, 
                average_precisions
                )

# ----------------- Computations ---------------

    #------------------------------------
    # compute_binary_pr_curve 
    #-------------------

    @classmethod
    def compute_binary_pr_curve(cls, 
                                labels, 
                                preds,
                                class_id,
                                thresholds=None,
                                ):
        '''
        Return the recall (x-axis) and precision (y-axis)
        values of a PR curve, its average precision (AP),
        and optimal threshold with corresponding f1, precision, 
        and recall values
        
        The optimal threshold's prec and rec yield the
        maximum f1 score. Information provided in the 
        BestOperatingPoint instance that is part of this
        method's return:
        
            threshold
            f1
            prec
            rec

        The result is packaged as a CurveSpecification
        that contains:
        
        	best_op_pt
        	precisions
        	recalls
        	thresholds
        	avg_prec'

        Procedure:
        
        A prec/rec point is computed for each 
        threshold point. 
        
        Works for binary classification.
        But can use sklearn's label_binaries to compute 
        separate curves for each class 
        (see compute_multiclass_pr_curves())
        
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

          Returned: a CurveSpecification instance
              containing:
                  best_op_pt
                  precisions
                  recalls
                  avg_prec

        :param labels: integer binary class labels.
            Exs.: [1,1,0,0], ['yes', 'yes', 'no', 'yes']
        :type labels: [int | str]
        :param preds: predictions output from a classifier.
            May be floats or integers
        :type preds: [float | int]
        :param class_id: ID of target class for which this
            curve is being constructed
        :type class_id: {int | str}
        :param thresholds: list of decision thresholds to
            decide whether preds are one class or the other.
            If None, uses [0.2, 0.4, 0.6, 0.8, 1]
        :type thresholds: [float | int]
        :return: CurveSpecification instances with optimal 
            operating point, and lists with prec and recall 
            ready for drawing a PR curve
        :rtype: CurveSpecification
        :raises ValueError if labels hold more than 
            two distinct values
        '''
        if type(labels) != list:
            labels = labels.tolist()
            
        uniq_classes = set(labels)
        
        if len(uniq_classes) > 2:
            raise ValueError(f"Labels limited to up to two distinct values; got {uniq_classes}")

        if thresholds is None:
            thresholds = [0.2, 0.4, 0.6, 0.8]
        precisions = []
        recalls = []
        class_list = list(uniq_classes)
        # Degenerate case: Only a single
        # class ever occurs in the labels.
        # To make the code below work, we
        # add a copy of that only class to
        # the class list of known classes,
        # and log a warning:
        if len(class_list) == 1:
            cls.log.warn(f"Only label {class_list[0]} occurs; always guessing that value.")
        class_list.append(class_list[0])
        
        # So far, no undefined recall or precision
        # i.e. no 0-denominator found:
        undef_prec = False
        undef_rec  = False
        undef_f1   = False
        
        for threshold in thresholds:
            y_pred = []
            for pred in preds:
                # Instead of just class_list[1],
                # must guard against only one
                # class (ID=0) in the labels.
                # In that special case, we always
                # predict 0:
                if pred >= threshold:
                    y_pred.append(class_list[1])
                else:
                    y_pred.append(class_list[0])
    
            y_pred_tn = torch.tensor(y_pred)
            
            # For 'No positive exist and classifier
            # properly doesn't predict a positive,
            # use:
            #      precision=1
            #      recall   =1
            # In this case prec and rec are undefined,
            # causing division by 0:
            
            try:
                
                with warnings.catch_warnings():
                    # Action to take: Ignore 
                    warnings.filterwarnings("error",
                                            #category=UndefinedMetricWarning,
                                            category=UserWarning,
                                            )
                    precision = precision_score(y_true=labels, 
                                                y_pred=y_pred_tn, 
                                                pos_label=class_list[1],
                                                    zero_division='warn')
            except Exception as _e:
                # Was it a div by zero from the prec calc?
                undef_prec = True
                precision  = Charter.DIV_BY_ZERO
                
            try:
                
                with warnings.catch_warnings():
                    # Action to take: Ignore 
                    warnings.filterwarnings("error",
                                            #category=UndefinedMetricWarning
                                            category=UserWarning
                                            )
                    recall    = recall_score(y_true=labels, 
                                             y_pred=y_pred_tn, 
                                             pos_label=class_list[1],
                                             zero_division=Charter.DIV_BY_ZERO)
            except Exception as _e:
                # Was it a div by zero from the prec calc?
                undef_rec = True
                recall  = Charter.DIV_BY_ZERO

            precisions.append(precision)
            recalls.append(recall)

        precs_np = np.array(precisions)
        recs_np  = np.array(recalls)

        with warnings.catch_warnings():
            try:
                warnings.filterwarnings("error",
                                        #category=UndefinedMetricWarning
                                        category=UserWarning
                                        )
                warnings.filterwarnings("true_divide",
                                        #category=UndefinedMetricWarning
                                        category=RuntimeWarning
                                        )
                f1_scores = 2 * (precs_np * recs_np) / (precs_np + recs_np)
            except Exception as _e:
                # Was it a div by zero from the prec calc?
                undef_f1= True
                # When both prec and recall are 0,
                # set f1 to zero:
                f1_scores = torch.tensor(
                    [Charter.DIV_BY_ZERO]*len(precs_np))

        best_op_idx = np.argmax(f1_scores)
        
        best_operating_pt = BestOperatingPoint(
            thresholds[best_op_idx],
            f1_scores[best_op_idx],
            precisions[best_op_idx],
            recalls[best_op_idx]
            )            

        # To make average_precision computation
        # work:
        recs_np_padded = np.append(recs_np, [0])
        precs_np_padded = np.append(precs_np, [1])
        
        avg_precision = \
            np.sum((recs_np_padded[:-1] - recs_np_padded[1:]) * precs_np_padded[:-1])
        
        res = CurveSpecification(best_operating_pt,
                                 recs_np_padded,
                                 precs_np_padded,
                                 thresholds,
                                 avg_precision,
                                 class_id,
                                 undef_prec=undef_prec,
                                 undef_rec=undef_rec,
                                 undef_f1=undef_f1
                                 )
        
        return res


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

    #------------------------------------
    # compute_multiclass_pr_curves 
    #-------------------
    
    @classmethod
    def compute_multiclass_pr_curves(cls,
                                    truth_labels, 
                                    raw_preds,
                                    thresholds=[0.2, 0.4, 0.6, 0.8]
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

        :param truth_labels: all truth labels shaped
            torch.Size([num-batches, batch-size])
        :type truth_labels: Tensor
        :param raw_preds: the logits for each class for
            each sample as 
            torch.Shape([num-batches, batch-size, num-classes])
        :type raw_preds: Tensor
        :return: (precisions, recalls, average_precisions, mAP)
        :rtype: ({int : [floats]}, {int : [floats]}, [floats], float)
        '''

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
            
            #**************
            # # Using sklearn's precision_recall_curve,
            # # which determines thresholds by its own
            # # algorithm:
            #
            # from sklearn.metrics import precision_recall_curve 
            # sklearn_precs,\
            # sklearn_recs,\
            # sklearn_thresholds = \
            #     precision_recall_curve(bin_label_col, preds_col)
            #**************

            # Obtain the information needed to 
            # draw one PR curve: a CurveSpecification
            # instance:
            one_class_curve = cls.compute_binary_pr_curve(bin_label_col,
                                                          preds_col,
                                                          col_idx,   # class_id
                                                          thresholds
                                                          )

            # Accumulate the curve indices 
            # in a dict, keyed by class ID:
            all_curves_info[col_idx] = one_class_curve

        avg_precs = [binary_curve_info['avg_prec']
                     for binary_curve_info
                     in all_curves_info.values()
                     ]
        mAP = np.mean(np.array(avg_precs)).tolist()
        
        return (all_curves_info, mAP) 


# ----------------- Visualizations ---------------

    #------------------------------------
    # fig_from_conf_matrix 
    #-------------------
    
    @classmethod
    def fig_from_conf_matrix(cls, 
                             conf_matrix,
                             title='Confusion Matrix',
                             write_in_fields=5
                             ):
        '''
        Given a confusion matrix, return a 
        matplotlib.pyplot Figure with a heatmap of the matrix.
        
        The write_in_fields arg controls whether or not
        each cell is filled with a label indicating its
        percentage. If:
        
            o True : always write the labels
            o False: never write the labels
            o int  : only write labels if number of classes
                is <= to int.
        
        :param conf_matrix: nxn confusion matrix representing
            rows:truth, cols:predicted for n classes
        :type conf_matrix: pd.DataFrame
        :param title: title at top of figure
        :type title: str
        :return: matplotlib figure with confusion
            matrix heatmap.
        :rtype: pyplot.Figure
        '''

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
        # Turn the tensor-typed dataframe elements into
        # floats:
        cm_normed_floats = cm_normed.astype(float)
        
        if type(write_in_fields) == int:
            annot = True if len(class_names) <= write_in_fields else False
        else:
            annot = write_in_fields 

        cmap.set_bad('gray')
        heatmap_ax = sns.heatmap(
            cm_normed_floats,
            #*****vmin=0.0, vmax=1.0,
            cmap=cmap,
            annot=annot,  # Do write percentages into cells
            fmt='g',      # Avoid scientific notation
            cbar=True,    # Do draw color bar legend
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=1,# Pixel,
            linecolor='gray'
            )
        heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), 
                                   rotation = 45 
                                   )
        return fig
    
    


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
    
    def __init__(self,
                 threshold,
                 f1_score,
                 precision,
                 recall
                 ):
        '''
        
        :param threshold: probability value that is the best
            decision threshold between two classes
        :type threshold: float
        :param f1_score: the F1 score achieved at the 
            given threshold's recall and precision
        :type f1_score: float
        :param precision: the precision value of the 
            best operating point on the PR curve
        :type precision: float
        :param recall: the recall value of the 
            best operating point on the PR curve
        :type recall: float
        '''
        super().__init__()

        if type(f1_score) == torch.Tensor:
            f1_score = float(f1_score)
        if type(precision) == torch.Tensor:
            precision = float(precision)
        if type(recall) == torch.Tensor:
            recall = float(recall)

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
    
         best_operating_pt: {'threshold' : thresholds[best_op_idx],
                             'f1'        : f1_scores[best_op_idx],
                             'prec'      : precisions[best_op_idx],
                             'rec'       : recalls[best_op_idx]
                             }
         recalls          : list of recall values
         precisions       : list of precision values
         thresholds       : list of probability decision thresholds
                            at which precions/recall pairs were computed
         avg_precision    : the Average Precision (AP) of all the points
         class_id         : ID (int or str) for which instances is a curve
         
    The precisions and recalls array-likes form
    the x/y pairs when zipped. 
    '''
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self,
                 best_operating_pt,
                 recalls,
                 precisions,
                 thresholds,
                 avg_precision,
                 class_id,
                 **kwargs
                 ):
        '''
        
        :param best_operating_pt: information where on the 
            curve the optimal F1 score is achieved
        :type best_operating_pt: BestOperatingPoint
        :param recalls: all recall values that with corresponding
            precisions define the curve
        :type recalls: [float]
        :param precisions: all recall values that with corresponding
            recall define the curve
        :type precisions: [float]
        :param thresholds: the decision thresholds at which the
            (recall/precision) points on the curve were computed. 
        :type thresholds: [float]
        :param avg_precision: the average of all precisions (AP)
        :type avg_precision: [float]
        :param class_id: class for which this obj
            is the pr-curve
        :type class_id: {int | str}
        
        '''
        
        super().__init__()
        self.__setitem__('best_op_pt', best_operating_pt)
        self.__setitem__('recalls', recalls)
        self.__setitem__('precisions', precisions)
        self.__setitem__('thresholds', thresholds)
        self.__setitem__('avg_prec', avg_precision)
        self.__setitem__('class_id', class_id)
        self.update(**kwargs)

    #------------------------------------
    # __repr__ 
    #-------------------
    
    def __repr__(self):
        uniq_id = hex(id(self))
        rep = f"<CurveSpecification AP={self.__getitem__('avg_prec')} {uniq_id}>"
        return(rep)

    #------------------------------------
    # __str__ 
    #-------------------
    
    def __str__(self):
        return repr(self)


# --------------- Main -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Charts and other analyses from inferencing."
                                     )

    # parser.add_argument('-l', '--errLogFile',
                        # help='fully qualified log file name to which info and error messages \n' +\
                             # 'are directed. Default: stdout.',
                        # dest='errLogFile',
                        # default=None)
    # parser.add_argument('-d', '--dryRun',
                        # help='show what script would do if run normally; no actual downloads \nor other changes are performed.',
                        # action='store_true')
    # parser.add_argument('my_integers',
                        # type=int,
                        # nargs='+',
                        # help='Repeatable: integers. Will show as list in my_integers')
    
    
    parser.add_argument('--conf_matrix',
                        help='create confusion matrix; value: path to csv file',
                        default=None
                        )
    parser.add_argument('--pr_curves',
                        help='draw (family of) PR curves; value: path to csv file',
                        default=None
                        )

    args = parser.parse_args()
    
    actions = {}
    if args.conf_matrix is not None:
        actions['conf_matrix'] = args.conf_matrix
    if args.pr_curves is not None:
        actions['pr_curves'] = args.pr_curves
        
    Charter(actions)