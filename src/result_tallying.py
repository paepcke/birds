'''
Created on Dec 23, 2020

@author: paepcke
'''

import datetime
import json
import os

import torch

import numpy as np
from utils.learning_phase import LearningPhase


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
        


# ----------------------- Class Train Results -----------

class TrainResult:
    '''
    Instances of this class hold results from training,
    validating, and testing.
    
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
                 conf_matrix,
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
        @param conf_matrix: confusion matrix containing counts
            of correctly and incorrectly predicted labels
        @type conf_matrix: Tensor
        @param badly_predicted_labels: optionally, the labels that
            were incorrectly predicted
        @type badly_predicted_labels: Tensor
        '''
        
        self._split_num      = split_num
        self._epoch          = epoch
        self._learning_phase = learning_phase
        self._loss           = loss
        self._conf_matrix    = conf_matrix
        self._badly_predicted_labels = badly_predicted_labels
        
        self._num_samples = self._num_correct = self._num_wrong = None
        self._within_class_recalls = self._within_class_precisions = None
        self._accuracy = None
        
        self._created_at = datetime.datetime.now() 

    #------------------------------------
    # num_correct 
    #-------------------

    @property
    def num_correct(self):
        if self._num_correct is None:
            self._num_correct = torch.sum(torch.diagonal(self.conf_matrix))
        return self._num_correct

    #------------------------------------
    # num_wrong
    #-------------------

    @property
    def num_wrong(self):
        if self._num_wrong is None:
            self._num_wrong = self.num_samples - self.num_correct
        return self._num_wrong

    #------------------------------------
    # precision 
    #-------------------

    @property
    def precision(self):
        return torch.mean(self.within_class_precisions)

    #------------------------------------
    # recall 
    #-------------------

    @property
    def recall(self):
        return torch.mean(self.within_class_recalls)

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
        if self._within_class_recalls is None:
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
        if self._within_class_precisions is None:
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
        if self._accuracy is None:
            self._accuracy = self.num_correct / self.num_samples
        return self._accuracy

    #------------------------------------
    # epoch 
    #-------------------

    @property
    def epoch(self):
        return self._epoch

    #------------------------------------
    # split_num 
    #-------------------

    @property
    def split_num(self):
        return self._split_num

    #------------------------------------
    # num_samples
    #-------------------

    @property
    def num_samples(self):
        if self._num_samples is None:
            self._num_samples = int(torch.sum(self.conf_matrix))
        return self._num_samples

    #------------------------------------
    # learning_phase 
    #-------------------
    
    @property
    def learning_phase(self):
        return self._learning_phase
    
    #------------------------------------
    # loss 
    #-------------------
    
    @property
    def loss(self):
        return self._loss

    #------------------------------------
    # conf_matrix 
    #-------------------
    
    @property
    def conf_matrix(self):
        return self._conf_matrix
    
    #------------------------------------
    # created_at 
    #-------------------
    
    @property
    def created_at(self):
        return(self._created_at)

    #------------------------------------
    # badly_predicted_labels 
    #-------------------

    @property
    def badly_predicted_labels(self):
        return self._badly_predicted_labels

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
