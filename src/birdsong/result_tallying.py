'''
Created on Dec 23, 2020

@author: paepcke
'''

import copy
import datetime
import os, sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import torch

from birdsong.utils.learning_phase import LearningPhase
from birdsong.utils.tensorboard_plotter import TensorBoardPlotter


packet_root = os.path.abspath(__file__.split('/')[0])
sys.path.insert(0,packet_root)

# ---------------------- Class Train Result Collection --------
class ResultCollection(dict):
    '''
    Hold an arbitrary number of ResultTally
    instances. Services:
        o acts as dict with keys being the epoch number
        o acts as a list as follows:

    '''
    
    #------------------------------------
    # Contructor
    #-------------------
    
    def __init__(self):
        self._sorted_tallies = []
        self._sorted_train_tallies = []
        self._sorted_val_tallies = []

    #------------------------------------
    # tallies
    #-------------------

    def tallies(self, 
                epoch=None, 
                phase=None,
                newest_first=False):
        '''
        Iterator for tallies, optionally filtering by
        epoch and/or phase (training vs. validation)
        
        :param epoch: epoch to filter by
        :type epoch: int
        '''
        
        # Make a shallow copy in case 
        # _sorted_tallies is modified in between
        # yields

        if phase is not None:
            if phase == LearningPhase.TRAINING:
                all_tallies = self._sorted_train_tallies.copy()
            elif phase == LearningPhase.VALIDATING:
                all_tallies = self._sorted_val_tallies.copy()
            elif phase == LearningPhase.TESTING:
                # We are not maintaining a separate
                # tally instance list:
                all_tallies = self._sorted_tallies.copy()
                all_tallies = filter(lambda t: t.phase == LearningPhase.TESTING,
                                     all_tallies
                                     )
            else:
                raise ValueError(f"Only TRAINING, VALIDATING, and TESTING learning phases supported, not {str(phase)}")
        else:
            all_tallies = self._sorted_tallies.copy()
            
        if epoch is not None:
            all_tallies = filter(lambda t: t.epoch == epoch, 
                                 all_tallies)

        if newest_first:
            all_tallies = all_tallies.reverse()
            
        for tally in all_tallies:
            yield tally

    #------------------------------------
    # __getitem__ 
    #-------------------

    def __getitem__(self, item):
        '''
        Even though ResultCollection is a dict,
        allow indexing into it, and especially

          my_coll[-1] to get the latest-epoch tally
          my_col[3:5] get tallies of epoch 3-5
          
        The latter only if tallies of all epoch's
        are present in this collection. Reality
        of the interface is a list of ResultTally
        sorted by epoch.
         
        :param item: integer or slice
        :type item: {int | list}
        :return element of the contained ResultTally
            instances
        :rtype: ResultTally
        '''
        if type(item) == tuple:
            # A tuple (epoch, LearningPhase).
            # Treat as a dict after
            # making the LearningPhase hashable
            #return super().__getitem__(item)
            return super().__getitem__((item[0], str(item[1])))
        
        return self._sorted_tallies.__getitem__(item)

    #------------------------------------
    # __setitem__ 
    #-------------------
    
    def __setitem__(self, epoch, tally):
        
        self.add(tally, epoch=epoch)

    #------------------------------------
    # __iter__ 
    #-------------------
    
    def __iter__(self):
        # As the iterator, return the 
        # sorted list of ResultTally instances:
        
        return self._sorted_tallies

    #------------------------------------
    # conf_matrix_aggregated 
    #-------------------
    
    def conf_matrix_aggregated(self, epoch=None): 
        
        conf_matrix_sum = torch.zeros((self.num_classes, self.num_classes), dtype=int)
        for tally in self.tallies(epoch):
            conf_matrix_sum += tally.conf_matrix
        return conf_matrix_sum / len(self)

    #------------------------------------
    # add 
    #-------------------
    
    def add(self, new_tally_result, epoch=None):
        '''
        Add new_tally_result instance to this collection:
        update the (self)dict with key being the tally's
        epoch, and value being the tally.
        
        Then update self._sorted_tallies, the
        epoch-sorted list of tallies in this collection
        
        If a tally for the given epoch and learning
        phase already exists, augment the existing
        tally with the content of the given arg.
        
        Normally the new result's epoch will be contained
        in its .epoch attr. But the epoch kwarg allows
        placing a ResultTally into any epoch key
        
        :param new_tally_result: the result to add
        :type new_tally_result: ResultTally
        '''

        if epoch is None:
            key = (new_tally_result.epoch, 
                   str(new_tally_result.phase))
        else:
            key = (epoch, str(new_tally_result.phase))

        # Does a result from an earlier batch in the
        # same epoch already exist?
        
        try:
            tally = self[key]
            if tally == new_tally_result:
                # The very tally was already
                # in this collection 
                return
            # Yes, already have a tally for this epoch
            # and training phase:
            tally.preds.extend(new_tally_result.preds)
            tally.labels.extend(new_tally_result.labels)
            tally.losses = torch.cat((tally.losses,
                                    new_tally_result.losses
                                    )
                                   )
            # The metrics of this existing ResultTally
            # will need to be recomputed if any
            # of its computed attrs is accessed:
            tally.metrics_stale = True
        except KeyError:
            # First ResultTally of given epoch in
            # this collection:
            super().__setitem__(key, new_tally_result)

        self._sort_tallies()

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
    # clear 
    #-------------------
    
    def clear(self):
        '''
        Removes all tallies from this collection
        '''
        
        super().clear()
        self._sorted_tallies     = []
        self._sorted_val_tallies = []
        self._sorted_train_tallies = []

    #------------------------------------
    # create_from 
    #-------------------
    
    @classmethod
    def create_from(cls, other):
        new_inst = cls()
        
        for tally in other.tallies():
            new_inst.add(copy.deepcopy(tally))
        
        return new_inst
    
    #------------------------------------
    # _sort_tallies 
    #-------------------
    
    def _sort_tallies(self):
        '''
        Sorts the result tallies in this 
        collection by their epoch, and 
        creates/updates _sorted_tallies
        '''
        
        unsorted_tallies = list(self.values())
        self._sorted_tallies = sorted(unsorted_tallies, 
                                      key=lambda tally: tally.epoch)
        self._sorted_train_tallies = filter(lambda tally: tally.phase == LearningPhase.TRAINING,
                                            self._sorted_tallies
                                            )
        self._sorted_val_tallies = filter(lambda tally: tally.phase == LearningPhase.VALIDATING,
                                          self._sorted_tallies
                                          )


# ----------------------- Class Train Results -----------

class ResultTally:
    '''
    Instances of this class hold results from training,
    validating, or testing.
    '''
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 epoch,
                 phase, 
                 outputs, 
                 labels, 
                 loss,
                 num_classes,
                 batch_size,
                 testing=False,
                 **kwargs
                 ):
        '''
        Create a new ResultTally. Any 
        name/val pairs in kwargs will be
        available in instances:
        
          tally = ResultTally(3,...,32, my_info1='tulip', my_info2='daffodil')
          
        tally.my_info1 ==> tulip
        tally.my_info2 ==> daffodil
        
        Outputs is either a list of class IDs, or, 
        for the example of a batch size of 2:
        
            tensor([[0.0162, 0.0096, 0.0925, 0.0157],
                    [0.0208, 0.0087, 0.0922, 0.0141]], grad_fn=<AddmmBackward>)
        
        That is, outputs may be raw logits, one per
        class in each row.
        
        Note: the number of target classes cannot be
              gleaned from outputs or labels, b/c some
              classes might not have been involved in this
              epoch
        
        :param epoch: epoch from which produced the results 
        :type epoch: int
        :param phase: the LearningPhase (e.g. LearningPhase.TRAINING)
        :type phase: LearningPhase
        :param outputs: predictions that model produced.
            These are the raw logits from the model.
        :type outputs: [float]
        :param labels: truth values
        :type labels: [int]
        :param loss: loss computed by the loss function
        :type loss: float
        :param num_classes: number of target classes
        :type num_classes: int
        '''

        # Some of the following assignments to instance
        # variables are just straight transfers from
        # arguments to inst vars. 
        
        self.created_at     = datetime.datetime.now()
        self.phase          = phase
        self.epoch          = epoch
        self.num_classes    = num_classes
        self.batch_size     = batch_size
        self.preds          = None
        self.labels         = labels.tolist()
        
        # Remember the attrs that don't need
        # to be computed, and access to which 
        # therefore does not need to trigger a 
        # computation:
        
        self._static_attrs = list(self.__dict__.keys())
        
        self.mean_loss      = None
        self.losses         = None

        # If unittesting:
        if testing:
            self.metrics_stale  = False
            return

        # Lazy computation of metrics:
        self.metrics_stale  = True
        
        self._initialize_loss(loss)
        
        
        # See whether the passed-in output
        # are integers, meaning class IDs,
        # or one logit for each class per
        # sample: 
        if type(outputs) == list and type(outputs[0]) == int:
            # Nothing to do, outputs are
            # already class IDs:
            self.preds = outputs
        else:
            # Turn the logits in outputs into
            # class predictions, filling in 
            # self.preds
            self._set_initial_preds(outputs)
        
        if len(kwargs) > 0:
            self.__dict__.update(kwargs)

    #------------------------------------
    # __getattribute__ 
    #-------------------
    
    def __getattr__(self, attr_name):

        # Asking for one of the instance
        # vars used below? If so, just return
        # their values (else infinite recursion):
        
        if attr_name in ['_static_attrs', 'metrics_stale']:
            return super().__getattr__(attr_name)

        # Is access to one of the computed attrs,
        # and the cache for those attrs is stale?
        
        if attr_name not in self._static_attrs and \
                self.metrics_stale:
            # (Re)compute attrs from the (presumably)
            # expanced preds and labels list:
            
            self.update_tally_metrics(self.__getattribute__('labels'),
                                      self.__getattribute__('preds'),
                                      )
            
        return self.__getattribute__(attr_name)

    #------------------------------------
    # update_tally_metrics 
    #-------------------
    
    def update_tally_metrics(self, labels, preds):
        '''
        Called from __getattr__(), so don't retrieve
        instance vars; pass them in.
        :param labels:
        :type labels:
        :param preds:
        :type preds:
        '''

        # Compute accuracy, adjust for chance, given 
        # number of classes, and shift to [-1,1] with
        # zero being chance:
         
        self.balanced_acc = balanced_accuracy_score(labels, 
                                                    preds,
                                                    adjusted=True)
        
        self.accuracy = accuracy_score(labels, 
                                       preds,
                                       normalize=True)

        # The following metrics are only 
        # reported for validation set:
        
        # For 'No positive exist and classifier
        # properly doesn't predict a positive,
        # use:
        #      precision=1
        #      recall   =1
        # In this case prec and rec are undefined,
        # causing division by 0:
        
        self.f1_macro    = f1_score(labels, preds, average='macro',
                                    zero_division=1
                                    )
        self.f1_micro   = precision_score(labels, preds, average='micro',
                                          zero_division=1
                                          )
        self.f1_weighted  = f1_score(labels, preds, average='weighted',
                                     zero_division=1
                                     )

        self.f1_all_classes  = f1_score(labels, preds, average=None,
                                     zero_division=1
                                     )

        
        self.prec_macro   = precision_score(labels, preds, average='macro',
                                            zero_division=1
                                            )
        self.prec_micro   = precision_score(labels, preds, average='micro',
                                            zero_division=1
                                            )
        self.prec_weighted= precision_score(labels, preds, average='weighted',
                                            zero_division=1
                                           )
        self.prec_all_classes = precision_score(labels, preds, average=None,
                                            zero_division=1
                                           )


        self.recall_macro   = recall_score(labels, preds, average='macro',
                                           zero_division=1
                                           )
        self.recall_micro   = recall_score(labels, preds, average='micro',
                                           zero_division=1
                                           )
        self.recall_weighted= recall_score(labels, preds, average='weighted',
                                           zero_division=1
                                           )
        self.recall_all_classes = recall_score(labels, preds, average=None,
                                           zero_division=1
                                           )

        # A confusion matrix whose entries
        # are normalized to show percentage
        # of all samples in a row the classifier
        # got rigth:
        
        self.conf_matrix = TensorBoardPlotter\
            .compute_confusion_matrix(labels,
                                      preds,
                                      self.num_classes,
                                      normalize=True)
            
        self.mean_loss = torch.mean(self.losses)

        self.metrics_stale = False

    #------------------------------------
    # _initialize_loss 
    #-------------------
    
    def _initialize_loss(self, loss):
        '''
        Called by ResultTally constructor to
        initialize self.losses and self.mean_loss.
        
        The loss is as encountered during one batch.
        Turn that into loss/sample, and 
        to this ResultTally's loss instance var, and
        return it as the other part of the return
        tuple. 

        :param loss: initial loss from training
        :type loss: torch.Tensor
        '''

        # Loss is per batch. Convert the number
        # to loss per sample (within each batch):
        new_loss = loss.detach() / self.batch_size
        # Add a dimension so we can later
        # add more loss tensors:
        self.losses = new_loss.unsqueeze(dim=0)
        self.mean_loss = torch.mean(self.losses)
        
    #------------------------------------
    # _set_initial_preds 
    #-------------------
    
    def _set_initial_preds(self, outputs):
        '''
        Convert outputs logits first into probabilities
        via softmax, then into class IDs via argmax. 
        The result is assigned to self.preds.
       
        :param outputs: raw outputs from model
        :type outputs: torch.Tensor
        '''
        
        # Ex.: For a batch_size of 2 we output logits like:
        #
        #     tensor([[0.0162, 0.0096, 0.0925, 0.0157],
        #             [0.0208, 0.0087, 0.0922, 0.0141]], grad_fn=<AddmmBackward>)
        #
        # Turn into probabilities along each row:
        
        pred_probs = torch.softmax(outputs, dim=1)
        
        # Now have:
        #
        #     tensor([[0.2456, 0.2439, 0.2650, 0.2454],
        #             [0.2466, 0.2436, 0.2648, 0.2449]])
        #
        # Find index of largest probability for each
        # of the batch_size prediction probs along each
        # row to get:
        #
        #  first to tensor([2,2]) then to [2,2]
        
        pred_tensor = torch.argmax(pred_probs, dim=1)
        self.preds = pred_tensor.tolist()

    #------------------------------------
    # compute_confusion_matrix
    #-------------------
    
    def compute_confusion_matrix(self, truth_labels, predicted_class_ids):
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

# *************** REMOVE
#     #------------------------------------
#     # num_correct 
#     #-------------------
# 
#     @property
#     def num_correct(self):
#         try:
#             return self._num_correct
#         except AttributeError:
#             self._num_correct = torch.sum(torch.diagonal(self.conf_matrix))
#             return self._num_correct
# 
#     #------------------------------------
#     # num_wrong
#     #-------------------
# 
#     @property
#     def num_wrong(self):
#         try:
#             return self._num_wrong
#         except AttributeError:
#             self._num_wrong = self.num_samples - self.num_correct
#             return self._num_wrong
# 
#     #------------------------------------
#     # precision 
#     #-------------------
# 
#     @property
#     def precision(self):
#         return self.precision_weighted
# 
#     #------------------------------------
#     # recall 
#     #-------------------
# 
#     @property
#     def recall(self):
#         return self.recall_weighted
# 
#     #------------------------------------
#     # within_class_recalls 
#     #-------------------
# 
#     @property
#     def within_class_recalls(self):
#         '''
#         A tensor with a recall for each
#         of the target classes. Length of the
#         tensor is therefore the number of 
#         the classes: 
#         '''
#         try:
#             return self._within_class_recalls
#         except AttributeError:
#             # The average == None causes prediction
#             # to be returned for all classes:
#             self._within_class_precisions = metrics.recall_score(self.truth_labels,
#                                                                  self.predicted_class_ids,
#                                                                  average=None
#                                                                  )
#             return self._within_class_recalls
#             
#     #------------------------------------
#     # within_class_precisions 
#     #-------------------
#     
#     @property
#     def within_class_precisions(self):
#         '''
#         A tensor with a precision for each
#         of the target classes. Length of the
#         tensor is therefore the number of 
#         the classes: 
#         '''
#         try:
#             return self._within_class_precisions
#         except AttributeError:
#             # The average == None causes prediction
#             # to be returned for all classes:
#             self._within_class_precisions = metrics.precision_score(self.truth_labels,
#                                                                     self.predicted_class_ids,
#                                                                     average=None
#                                                                     )
#             return self._within_class_precisions
# 
#     #------------------------------------
#     # accuracy 
#     #-------------------
# 
#     @property
#     def accuracy(self):
#         try:
#             return self._accuracy
#         except AttributeError:            
#             self._accuracy = metrics.accuracy_score(self.truth_labels, self.predicted_class_ids)
#             return self._accuracy
# 
#     #------------------------------------
#     # num_samples
#     #-------------------
# 
#     @property
#     def num_samples(self):
#         try:
#             return self._num_samples
#         except AttributeError:
#             self._num_samples = int(torch.sum(self.conf_matrix))
#             return self.num_samples

    #------------------------------------
    # __repr__ 
    #-------------------
    
    def __repr__(self):
        (cm_num_rows, cm_num_cols) = self.conf_matrix.size()
        cm_dim = f"{cm_num_rows} x {cm_num_cols}"
        learning_phase = str(self.phase)
        human_readable = (f"<ResultTally epoch {self.epoch} " +
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
        return f"<TrainResult object at {hex(id(self))}>"

    #------------------------------------
    # __eq__ 
    #-------------------
     
    def __eq__(self, other):
        '''
        Return True if given TrainResult instance
        is equal to self in all property values
        :param other: instance to compare to
        :type other: TrainResult
        :return: True for equality
        :rtype: bool
        '''
        if not isinstance(other, ResultTally):
            return False

        # Compare corresponding property values
        # between self and other. For Tensor quantities,
        # ensure that all corresponding elements of the 
        # tensors are equal:

        prop_dict_self   = self.__dict__
        prop_dict_other  = other.__dict__
        prop_names = prop_dict_self.keys()
        
        for prop_name in prop_names:
            val_self  = prop_dict_self[prop_name]
            val_other = prop_dict_other[prop_name]
            if type(val_self) != type(val_other):
                return False
            if type(val_self) == torch.Tensor:
                if (val_self != val_other).any():
                    return False
                continue
            else:
                #********
                try:
                    if val_self != val_other:
                        return False
                except Exception as e:
                    print(e)
                #********
        return True

