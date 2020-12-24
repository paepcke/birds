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
    # mean_accuracy
    #-------------------

    def mean_accuracy(self, epoch=None):
        
        if epoch is None:
            m = np.mean([tally.accuracy() 
                         for tally 
                         in self.values()])
        else:
            m = np.mean([tally.accuracy() 
                         for tally 
                         in self.values() 
                         if tally.epoch() == epoch])
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
        self._within_class_recalls = self._within_class_precisions = None
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
        if self._within_class_precisions is None:
            #  For each class C:
            #     For each class C: num_correctly_predicted-C-samples / num-samples-predicted-to-be-in-class-C
            diag = torch.diagonal(self.conf_matrix())
            self._within_class_precisions = diag / torch.sum(self.conf_matrix(), 
                                                            axis = 1)
        return self._within_class_precisions

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
