'''
Created on Dec 23, 2020

@author: paepcke
'''

import os, sys
packet_root = os.path.abspath(__file__.split('/')[0])
sys.path.insert(0,packet_root)

import copy
import datetime
from collections import UserDict

from sklearn import metrics
from sklearn.metrics import confusion_matrix
import torch

import numpy as np

from birdsong.utils.learning_phase import LearningPhase


# ---------------------- Class Train Result Collection --------
class TrainResultCollection(dict):
    
    #------------------------------------
    # Contructor
    #-------------------
    
    def __init__(self, initial_train_result=None):
        
        if initial_train_result is not None:
            self.results[initial_train_result.split_num] = initial_train_result
        self.epoch_losses_training    = {}
        self.epoch_losses_validation  = {}
        self.epoch_losses_testing     = {}

    #------------------------------------
    # tallies
    #-------------------

    def tallies(self, epoch=None, learning_phase=None):
        '''
        Retrieve tallies, optionally filtering by
        epoch and/or learning phase.
        
        @param epoch: epoch to filter by
        @type epoch: int
        @param learning_phase: learning phase to filter by
        @type learning_phase: LearningPhase {TRAINING | VALIDATING | TESTING}

        '''
        
        all_tallies = self.values()
        if epoch is not None:
            all_tallies = filter(lambda t: t.epoch == epoch, 
                                 all_tallies)
        if learning_phase is not None:
            all_tallies = filter(lambda t: t.learning_phase == learning_phase,
                                 all_tallies)
        
        for tally in sorted(list(all_tallies), key=lambda t: t.created_at):
            yield tally

    # ----------------- Epoch-Level Aggregation Methods -----------

    #------------------------------------
    # cumulative_loss 
    #-------------------
    
    def cumulative_loss(self, epoch=None, learning_phase=LearningPhase.TRAINING):
        '''
        Return the accumulated loss, either
        summed over all epochs, or by epoch
        
        @param epoch: optionally an epoch whose
            accumulated loss is to be returned
        @type epoch: int
        @param learning_phase: from which phase, 
            training, validating, testing the loss
            is to be returned
        @type learning_phase: LearningPhase
        @returned cumulative loss
        @rtype: float
        '''
        
        loss_dict = self.fetch_loss_dict(learning_phase)
        if epoch is None:
            res = torch.sum(torch.tensor(list(loss_dict.values()), 
                                         dtype=float))
        else:
            res = loss_dict[epoch]
        return float(res)

    #------------------------------------
    # mean_accuracy
    #-------------------

    def mean_accuracy(self, epoch=None, learning_phase=LearningPhase.TRAINING):
        '''
        Computes the mean of all accuracies in the given 
        epoch and learning phase.
        
        @param epoch: epoch during which tallies must have
            been created to be included in the mean:
        @type epoch: int
        @param learning_phase: learning phase during which 
            included tallies must have been produced.
        @type learning_phase: LearningPhase
        @return mean accuracy over the specified tallies
        @rtype float
        '''

        if epoch is None:
            m = np.mean([tally.accuracy
                         for tally 
                         in self.values()
                         if tally.learning_phase == learning_phase
                         ])
        else:
            m = np.mean([tally.accuracy 
                         for tally 
                         in self.values() 
                         if tally.epoch == epoch and \
                            tally.learning_phase == learning_phase
                         ])
        # m is an np.float.
        # We want to return a Python float. The conversion
        # starts being inaccurate around the 6th digit.
        # Example:
        #      torch.tensor(0.9).item() --> 0.8999999761581421  
        # This inaccuracy is 'inherited' into the np.float32. 
        # The following gymnastics are a work-around:
        # Round in numpy country:
        
        mean_acc_tensor = (m * 10**6).round() / (10**6)
        
        # Convert to Python float, and round that
        # float to 6 digits:
        
        mean_acc = round(mean_acc_tensor.item(), 6) 
        
        return mean_acc

    #------------------------------------
    # mean_weighted_precision 
    #-------------------
    
    
    def mean_weighted_precision(self, 
                                epoch, 
                                learning_phase=LearningPhase.VALIDATING):
        if epoch is None:
            m = np.mean([tally.precision_weighted
                         for tally 
                         in self.values()
                         if tally.learning_phase == learning_phase
                         ])
        else:
            m = np.mean([tally.precision_weighted
                         for tally 
                         in self.values() 
                         if tally.epoch == epoch and \
                            tally.learning_phase == learning_phase
                         ])

        # m is an np.float.
        # We want to return a Python float. The conversion
        # starts being inaccurate around the 6th digit.
        # Example:
        #      torch.tensor(0.9).item() --> 0.8999999761581421  
        # This inaccuracy is 'inherited' into the np.float32. 
        # The following gymnastics are a work-around:
        # Round in numpy country:
        
        mean_prec_tensor = (m * 10**6).round() / (10**6)
        
        # Convert to Python float, and round that
        # float to 6 digits:
        
        mean_precision = round(mean_prec_tensor.item(), 6)
        return mean_precision

    #------------------------------------
    # mean_weighted_recall
    #-------------------

    def mean_weighted_recall(self, epoch, learning_phase=LearningPhase.VALIDATING):
        if epoch is None:
            m = np.mean([tally.recall_weighted
                         for tally 
                         in self.values()
                         if tally.learning_phase == learning_phase
                         ])
        else:
            m = np.mean([tally.recall_weighted
                         for tally 
                         in self.values() 
                         if tally.epoch == epoch and \
                            tally.learning_phase == learning_phase
                         ])

        # m is an np.float.
        # We want to return a Python float. The conversion
        # starts being inaccurate around the 6th digit.
        # Example:
        #      torch.tensor(0.9).item() --> 0.8999999761581421  
        # This inaccuracy is 'inherited' into the np.float32. 
        # The following gymnastics are a work-around:
        # Round in numpy country:
        
        mean_recall_tensor = (m * 10**6).round() / (10**6)
        
        # Convert to Python float, and round that
        # float to 6 digits:
        
        mean_recall = round(mean_recall_tensor.item(), 6)
        return mean_recall

    #------------------------------------
    # conf_matrix_aggregated 
    #-------------------
    
    def conf_matrix_aggregated(self, 
                               epoch=None, 
                               learning_phase=LearningPhase.VALIDATING):
        
        conf_matrix_sum = torch.zeros((self.num_classes, self.num_classes), dtype=int)
        for tally in self.tallies(epoch, learning_phase):
            conf_matrix_sum += tally.conf_matrix
        return conf_matrix_sum

    #------------------------------------
    # num_classes 
    #-------------------
    
    @property
    def num_classes(self):
        '''
        Return the number of target
        classes we know about.
        
        @return number of target classes
        @rtype int
        '''
        
        if len(self) == 0:
            return 0
        try:
            return self._num_classes
        except AttributeError:
            # Just use the number of classes 
            # in the first tally instance that
            # this collection instance contains:
            tallies_iter = iter(self.values())
            self._num_classes = next(tallies_iter).num_classes 
            return self._num_classes 

    #******* Needs thinking and debugging
#     #------------------------------------
#     # mean_within_class_recall
#     #-------------------
#     
#     def mean_within_class_recall(self, epoch=None):
#         
#         if epoch is None:
#             m = np.mean([tally.within_class_recalls()
#                          for tally
#                          in self.values()])
#         else:
#             m = np.mean([tally.within_class_recalls()
#                          for tally
#                          in self.values() 
#                          if tally.epoch == epoch])
#         return m
# 
#     #------------------------------------
#     # mean_within_class_precision 
#     #-------------------
#     
#     def mean_within_class_precision(self, epoch=None):
#         
#         if epoch is None:
#             m = torch.mean([tally.within_class_precisions() 
#                             for tally
#                             in self.values()]) 
#         else:
#             m = torch.mean([tally.within_class_precisions 
#                             for tally
#                             in self.values() 
#                             if tally.epoch == epoch])
#         return m

    #------------------------------------
    # add 
    #-------------------
    
    def add(self, tally_result):
        '''
        Same as TrainResultCollection-instance[split_id] = tally_result,
        but a bit more catering to a mental collection
        model.
        @param tally_result: the result to add
        @type tally_result: TrainResult
        '''
        # Need as retrieval key the epoch, the split_num, 
        # and the learning phase, because split nums restart
        # with each epoch. The key must be hashable, so
        # cannot use the raw learning_phase enum instance.
        # Convert it to 'Training', 'Validating', or 'Testing':
        
        learning_phase_textual = tally_result.learning_phase.name.capitalize()
        tally_key = (tally_result.epoch, 
                     tally_result.split_num,
                     learning_phase_textual,
                     ) 
        self[tally_key] = tally_result

    #------------------------------------
    # add_loss 
    #-------------------
    
    def add_loss(self, epoch, loss, learning_phase=LearningPhase.TRAINING):

        loss_dict = self.fetch_loss_dict(learning_phase)

        try:
            loss_dict[epoch] += loss
        except KeyError:
            # First addition of a loss. If we
            # don't store a copy, then future
            # additions will modify the passed-in
            # loss variable:
            loss_dict[epoch] = loss.detach().clone()

    #------------------------------------
    # epochs 
    #-------------------
    
    def epochs(self):
        '''
        Return a sorted list of epochs
        that are represented in tallies
        within the collection. 
        '''
        return sorted(set(self.keys()))

    #------------------------------------
    # create_from 
    #-------------------
    
    @classmethod
    def create_from(cls, other):
        new_inst = cls()
        new_inst.epoch_losses_training = other.epoch_losses_training.copy()
        new_inst.epoch_losses_validation = other.epoch_losses_validation.copy()
        new_inst.epoch_losses_testing = other.epoch_losses_testing.copy()
        
        for tally in other.tallies():
            new_inst.add(copy.deepcopy(tally))
        
        return new_inst

    #------------------------------------
    # num_tallies 
    #-------------------

    def num_tallies(self, epoch=None, learning_phase=LearningPhase.TRAINING):
        '''
        Return number of tallies contained in
        this collection. Available filters are
        epoch and learning phase:
         
        @param epoch: epoch to which counted tallies
            are to belong
        @type epoch: {None | int}
        @param learning_phase: learning phase to which 
            counted tallies are to belong 
        @type learning_phase: {None | LearningPhase}
        @return number of tallies held in collection
        @rtype int
        '''

        if epoch is None:
            l = len([tally
                     for tally 
                     in self.values()
                     if tally.learning_phase() == learning_phase
                     ])
        else:
            l = len([tally
                     for tally 
                     in self.values() 
                     if tally.epoch() == epoch and \
                        tally.learning_phase() == learning_phase
                     ])
            
        return l

    #------------------------------------
    # fetch_loss_dict
    #-------------------
    
    def fetch_loss_dict(self, learning_phase):
        
        if learning_phase == LearningPhase.TRAINING:
            loss_dict = self.epoch_losses_training
        elif learning_phase == LearningPhase.VALIDATING:
            loss_dict = self.epoch_losses_validation
        elif learning_phase == LearningPhase.TESTING:
            loss_dict = self.epoch_losses_testing
        else:
            raise ValueError(f"Learning phase must be a LearningPhase enum element, not '{learning_phase}'")

        return loss_dict
        

# ---------------------- class EpochSummary ---------------

class EpochSummary(UserDict):
    '''
    Constructs and stores all measurements
    from one epoch. Uses a given tally collection
    for its computational work.
    
    Behaves like a dict.
    '''

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, tally_collection, epoch):
        '''
        Given a filled-in tally_collection, filter
        the measurements by the given epoch. The
        resulting instance acts like a dict with 
        the following keys:
        
           o mean_accuracy_training
           o mean_accuracy_validating
           o epoch_loss
           o epoch_mean_weighted_precision
           o epoch_mean_weighted_recall
           o epoch_conf_matrix
        
        @param tally_collection:
        @type tally_collection:
        @param epoch:
        @type epoch:
        '''

        super().__init__()
        
        # Mean of accuracies among the 
        # training splits of this epoch
        self['mean_accuracy_training'] = \
           tally_collection.mean_accuracy(epoch, 
                                               learning_phase=LearningPhase.TRAINING)
           
        self['mean_accuracy_validating'] = \
           tally_collection.mean_accuracy(epoch, 
                                               learning_phase=LearningPhase.VALIDATING)

        self['epoch_loss'] = tally_collection.cumulative_loss(epoch=epoch, 
                                                           learning_phase=LearningPhase.VALIDATING)
        self['epoch_mean_weighted_precision'] = \
           tally_collection.mean_weighted_precision(epoch,
                                                          learning_phase=LearningPhase.VALIDATING
                                                          )
            
        self['epoch_mean_weighted_recall'] = \
           tally_collection.mean_weighted_recall(epoch,
                                                       learning_phase=LearningPhase.VALIDATING
                                                       )

        # For the confusion matrix: add all 
        # the confusion matrices from Validation
        # runs:
        self['epoch_conf_matrix'] = tally_collection.conf_matrix_aggregated(epoch=epoch,
                                                                         learning_phase=LearningPhase.VALIDATING
                                                                         )

        # Maybe not greatest style but:
        # Allow clients to use dot notation in addition
        # to dict format:
        #     my_epoch_summary.epoch_loss
        # in addition to: 
        #     my_epoch_summary['epoch_loss']
        for instVarName,instVarValue in list(self.items()):
            setattr(self, instVarName, instVarValue)
 
# ----------------------- Class Train Results -----------

class TrainResult:
    '''
    Instances of this class hold results from training,
    validating, and testing for each split.
    
    See also class TrainResultCollection, which 
    holds TrainResult instances, and provides
    overview statistics. 
    '''
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 split_num, 
                 epoch, 
                 learning_phase, 
                 loss, 
                 predicted_class_ids,
                 truth_labels,
                 num_classes,
                 badly_predicted_labels=None):
        '''
        Organize results from one train, validate,
        or test split.

        @param split_num: split number within current epoch
        @type split_num: int
        @param epoch: current epoch
        @type epoch: int
        @param learning_phase: whether result is from training,
            validation, or testing phase
        @type learning_phase: LearningPhase
        @param loss: result of loss function
        @type loss: Tensor
        @param predicted_class_ids:
        @type predicted_class_ids:
        @param num_classes: number of target classes
        @type num_classes: int
        @param badly_predicted_labels: optionally, the labels that
            were incorrectly predicted
        @type badly_predicted_labels: Tensor
        '''
        
        # Some of the following assignments to instance
        # variables are just straight transfers from
        # arguments to inst vars. 
        
        self.created_at     = datetime.datetime.now()
        self.split_num      = split_num
        self.epoch          = epoch
        self.learning_phase = learning_phase
        self.loss           = loss
        self.badly_predicted_labels = badly_predicted_labels
        self.num_classes    = num_classes
                
        self.conf_matrix = self.compute_confusion_matrix(predicted_class_ids,
                                                         truth_labels)

        # Find classes that are present in the
        # truth labels; all others will be excluded
        # from metrics in this result.
        # Use set subtraction to get the non-represented:

        # Precision

        self.precision_macro = metrics.precision_score(truth_labels, 
                                                       predicted_class_ids,
                                                       average='macro',
                                                       zero_division=0
                                                       )

        self.precision_micro = metrics.precision_score(truth_labels, 
                                                       predicted_class_ids,
                                                       average='micro',
                                                       zero_division=0
                                                       )

        self.precision_weighted = metrics.precision_score(truth_labels, 
                                                          predicted_class_ids,
                                                          average='weighted',
                                                          zero_division=0
                                                          )
        
        # Recall
        
        self.recall_macro = metrics.recall_score(truth_labels, 
                                                 predicted_class_ids,
                                                 average='macro',
                                                 zero_division=0
                                                 )

        self.recall_micro = metrics.recall_score(truth_labels, 
                                                 predicted_class_ids,
                                                 average='micro',
                                                 zero_division=0
                                                 )

        self.recall_weighted = metrics.recall_score(truth_labels, 
                                                    predicted_class_ids,
                                                    average='weighted',
                                                    zero_division=0
                                                    )
                
        self.f1_score_weighted = metrics.precision_score(truth_labels, 
                                                         predicted_class_ids,
                                                         average='weighted',
                                                         zero_division=0
                                                         )

    #------------------------------------
    # compute_confusion_matrix
    #-------------------
    
    def compute_confusion_matrix(self, predicted_class_ids, truth_labels):
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
            labels=list(range(self.num_classes)) # Class labels
            ))

        return conf_matrix

    #------------------------------------
    # num_correct 
    #-------------------

    @property
    def num_correct(self):
        try:
            return self._num_correct
        except AttributeError:
            self._num_correct = torch.sum(torch.diagonal(self.conf_matrix))
            return self._num_correct

    #------------------------------------
    # num_wrong
    #-------------------

    @property
    def num_wrong(self):
        try:
            return self._num_wrong
        except AttributeError:
            self._num_wrong = self.num_samples - self.num_correct
            return self._num_wrong

    #------------------------------------
    # precision 
    #-------------------

    @property
    def precision(self):
        return self.precision_weighted

    #------------------------------------
    # recall 
    #-------------------

    @property
    def recall(self):
        return self.recall_weighted

    #------------------------------------
    # within_class_recalls 
    #-------------------

    @property
    def within_class_recalls(self):
        '''
        A tensor with a recall for each
        of the target classes. Length of the
        tensor is therefore the number of 
        the classes: 
        '''
        try:
            return self._within_class_recalls
        except AttributeError:
            #  For each class C: 
            #     num_correctly_predicted-C-samples / num-samples-in-class-C
            diag = torch.diagonal(self.conf_matrix)
            self._within_class_recalls = diag / torch.sum(self.conf_matrix, 
                                                          axis = 0)
            return self._within_class_recalls
            
    #------------------------------------
    # within_class_precisions 
    #-------------------
    
    @property
    def within_class_precisions(self):
        '''
        A tensor with a precision for each
        of the target classes. Length of the
        tensor is therefore the number of 
        the classes: 
        '''
        try:
            return self._within_class_precisions
        except AttributeError:
            #  For each class C:
            #     For each class C: num_correctly_predicted-C-samples / num-samples-predicted-to-be-in-class-C
            diag = torch.diagonal(self.conf_matrix)
            self._within_class_precisions = diag / torch.sum(self.conf_matrix, 
                                                            axis = 1)
            return self._within_class_precisions

    #------------------------------------
    # accuracy 
    #-------------------

    @property
    def accuracy(self):
        try:
            return self._accuracy
        except AttributeError:            
            self._accuracy = self.num_correct / self.num_samples
            return self._accuracy

    #------------------------------------
    # num_samples
    #-------------------

    @property
    def num_samples(self):
        try:
            return self._num_samples
        except AttributeError:
            self._num_samples = int(torch.sum(self.conf_matrix))
            return self.num_samples

    #------------------------------------
    # __repr__ 
    #-------------------
    
    def __repr__(self):
        (cm_num_rows, cm_num_cols) = self.conf_matrix.size()
        cm_dim = f"{cm_num_rows} x {cm_num_cols}"
        lp = self.learning_phase
        if lp == LearningPhase.TRAINING:
            learning_phase = 'Train'
        elif lp == LearningPhase.VALIDATING:
            learning_phase = 'Val'
        elif lp == LearningPhase.TESTING:
            learning_phase = 'Test'
        else:
            raise TypeError(f"Wrong type for {self.learning_phase}")
            
        human_readable = (f"<TrainResult epoch {self.epoch} " +
                          f"split {self.split_num} " +
                          f"phase {learning_phase} " +
                          f"conf_matrix {cm_dim}>")
        return human_readable 

    #------------------------------------
    # __str__ 
    #-------------------
    
    def __str__(self):
        '''
        String representation guaranteed to be unique within a session
        '''
        return f"<TrainResult object at {self.id()}>"

            
#===============================================================================
# 
#===============================================================================
#     #------------------------------------
#     # __eq__ 
#     #-------------------
#     
#     def __eq__(self, other):
#         '''
#         # *********Needs update
#         
#         Return True if given TrainResult instance
#         is equal to self in all but loss and weights
#         @param other: instance to compare to
#         @type other: TrainResult
#         @return: True for equality
#         @rtype: bool
#         '''
#         if not isinstance(other, TrainResult):
#             return False
#         
#         if  round(self.best_valid_acc,4)       ==  round(other.best_valid_acc,4)         and \
#             round(self.best_valid_fscore,4)    ==  round(other.best_valid_fscore,4)      and \
#             round(self.best_valid_precision,4) ==  round(other.best_valid_precision,4)   and \
#             round(self.best_valid_recall,4)    ==  round(other.best_valid_recall,4):
#             return True
#         else:
#             return False
