'''
Created on Dec 24, 2020

Used as indicator of whether an action
is taken within the context of training, 
validation, or testing.

@author: paepcke
'''
from enum import Enum

class LearningPhase(Enum):
    TRAINING   = 1
    VALIDATING = 2
    TESTING    = 3

    #------------------------------------
    # __eq__ 
    #-------------------

    def __eq__(self, other):
        if type(other) != LearningPhase:
            raise TypeError("Equality of a LearningPhase can only be tested with another LearningPhase")
        return self.value == other.value
