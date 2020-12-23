'''
Created on Dec 23, 2020

@author: paepcke
'''

import numpy as np
import torch

# ---------------------- Class Train Result Collection --------

class TrainResultCollection(dict):
    
    #------------------------------------
    # Contructor
    #-------------------
    
    def __init__(self, initial_train_result=None):
        
        if initial_train_result is not None:
            self.results[initial_train_result.split_num] = initial_train_result
        self.epoch_losses = {}

    # ----------------- Epoch-Level Aggregation Methods -----------
    
    #------------------------------------
    # accuracy
    #-------------------

    def accuracy(self, epoch):
        np.mean([tally.accuracy() for tally 
                                  in self.values() 
                                  if tally.epoch == epoch])

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
        self[tally_result.split_num] = tally_result


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

    def __init__(self, split_num, epoch, learning_phase, conf_matrix):
        
        self._split_num      = split_num
        self._epoch          = epoch
        self._learning_phase = learning_phase
        self._conf_matrix    = conf_matrix
        
        self._num_samples = self._num_correct = self._num_wrong = None
        self._within_class_recalls = self._within_class_precision = None
        self._accuracy = None

    #------------------------------------
    # num_correct 
    #-------------------
    
    def num_correct(self):
        if self._num_correct is None:
            self._num_correct = torch.sum(torch.diagonal(self.conf_matrix()))
        return self._num_correct

    #------------------------------------
    # num_wrong
    #-------------------

    def num_wrong(self):
        if self._num_wrong is None:
            self._num_wrong = self.num_samples() - self.num_correct()
        return self._num_wrong

    #------------------------------------
    # within_class_recalls 
    #-------------------
    
    def within_class_recalls(self):
        if self._within_class_recalls is None:
            #  For each class C: 
            #     num_correctly_predicted-C-samples / num-samples-in-class-C
            diag = torch.diagonal(self.conf_matrix())
            self._within_class_recalls = diag / torch.sum(self.conf_matrix(), 
                                                          axis = 0)
        return self._within_class_recalls
            
    #------------------------------------
    # within_class_precisions 
    #-------------------
    
    def within_class_precisions(self):
        if self._within_class_precision is None:
            #  For each class C:
            #     For each class C: num_correctly_predicted-C-samples / num-samples-predicted-to-be-in-class-C
            diag = torch.diagonal(self.conf_matrix())
            self._within_class_precision = diag / torch.sum(self.conf_matrix(), 
                                                            axis = 1)
        return self._within_class_precision

    #------------------------------------
    # accuracy 
    #-------------------

    def accuracy(self):
        if self._accuracy is None:
            self._accuracy = self.num_correct() / self.num_samples()
        return self._accuracy

    #------------------------------------
    # epoch 
    #-------------------

    def epoch(self):
        return self._epoch

    #------------------------------------
    # split_num 
    #-------------------
    
    def split_num(self):
        return self._split_num

    #------------------------------------
    # num_samples
    #-------------------

    def num_samples(self):
        if self._num_samples is None:
            self._num_samples = int(torch.sum(self.conf_matrix()))
        return self._num_samples

    #------------------------------------
    # learning_phase 
    #-------------------
    
    def learning_phase(self):
        return self._learning_phase
    
    #------------------------------------
    # conf_matrix 
    #-------------------
    
    def conf_matrix(self):
        return self._conf_matrix

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

    #------------------------------------
    # print 
    #-------------------
    
    def print(self, include_weights=False):
        msg = (f"best_valid_acc      : {self.best_valid_acc}\n"
               f"best_valid_fscore   : {self.best_valid_fscore}\n"
               f"best_valid_precision: {self.best_valid_precision}\n"
               f"best_valid_recall   : {self.best_valid_recall}\n"
               f"best_valid_loss     : {self.best_valid_loss}\n"
               )
        if include_weights:
            msg += f"best_valid_weights: {self.best_valid_wts}\n" 
        print(msg)
