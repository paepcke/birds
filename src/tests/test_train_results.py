'''
Created on Dec 23, 2020

@author: paepcke
'''
import unittest

import torch
from sklearn.metrics import confusion_matrix

from birds_train_parallel import TrainResultCollection, TrainResult
from birds_train_parallel import LearningPhase

#********TEST_ALL = True
TEST_ALL = False

class Test(unittest.TestCase):

    #------------------------------------
    # setUp 
    #-------------------

    def setUp(self):
        self.tally_collection = TrainResultCollection()
        self.num_classes = 4
        
        self.single_pred  = torch.tensor([[ 0.1,  0.1,  0.5,  0.2]])
        
        # Label leading to batch correctly 
        # predicted: target class 2
        self.single_label_matching  = torch.tensor([2])
        
        # Label leading to batch badly predicted:
        # target class 3:
        self.single_label_non_match = torch.tensor([3])
        
        self.batch_pred  = torch.tensor([[ 0.1,  0.1,  0.5,  0.2],
                                         [ 0.6,  0.3,  0.5,  0.1]
                                         ])

        # Labels leading to both batches correctly 
        # predicted: target class 2 for first row,
        # class 0 for second row:
        
        self.batch_label_matching  = torch.tensor([2,0])
        
        # Labels that lead to first batch correct, 
        # second not:
        
        self.batch_label_non_match = torch.tensor([2,1]) 

    #------------------------------------
    # tearDown 
    #-------------------

    def tearDown(self):
        pass

    #------------------------------------
    # test_basics_single_split 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_basics_single_split(self):
        self.epoch = 1
        tally = self.tally_result(
                            0, # Split number
                            self.single_label_matching,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch(), 1)
        self.assertEqual(tally.split_num(), 0)
        self.assertEqual(tally.num_samples(), 1)
        self.assertEqual(tally.num_correct(), 1)
        self.assertEqual(tally.num_wrong(), 0)

        tally = self.tally_result(
                            0, # Split number
                            self.single_label_non_match,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch(), 1)
        self.assertEqual(tally.split_num(), 0)
        self.assertEqual(tally.num_samples(), 1)
        self.assertEqual(tally.num_correct(), 0)
        self.assertEqual(tally.num_wrong(), 1)

    #------------------------------------
    # test_basics_two_splits
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_basics_two_splits(self):
        self.epoch = 1
        tally = self.tally_result(
                            0, # Split number
                            self.batch_label_matching,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch(), 1)
        self.assertEqual(tally.split_num(), 0)
        self.assertEqual(tally.num_samples(), 2)
        self.assertEqual(tally.num_correct(), 2)
        self.assertEqual(tally.num_wrong(), 0)

        tally = self.tally_result(
                            0, # Split number
                            self.batch_label_non_match,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch(), 1)
        self.assertEqual(tally.split_num(), 0)
        self.assertEqual(tally.num_samples(), 2)
        self.assertEqual(tally.num_correct(), 1)
        self.assertEqual(tally.num_wrong(), 1)

    #------------------------------------
    # test_accuracy
    #-------------------

    #********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_accuracy(self):
        self.epoch = 1
        # Single split, correct prediction
        tally = self.tally_result(
                            0, # Split number
                            self.single_label_matching,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy(), 1)

        # Single split, incorrect prediction
        tally = self.tally_result(
                            0, # Split number
                            self.single_label_non_match,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy(), 0)

        # Two splits, correct predictions
        tally = self.tally_result(
                            0, # Split number
                            self.batch_label_matching,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy(), 1)

        # Two splits, incorrect predictions
        tally = self.tally_result(
                            0, # Split number
                            self.batch_label_non_match,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy(), 0.5)

# ---------------- Utils ------------

    #------------------------------------
    # tally_result
    #-------------------

    def tally_result(self,
                     split_num, 
                     labels_tns, 
                     pred_prob_tns,
                     learning_phase,
                     ):
        '''
        Copy of BirdTrainer's tally_result for
        testing the tallying facility:
        '''
        # Predictions are for one batch. Example for
        # batch_size 2 and 4 target classes:
        #  
        #    torch.tensor([[1.0, -2.0,  3.4,  4.2],
        #                  [4.1,  3.0, -2.3, -1.8]
        #                  ])
        # get:
        #     torch.return_types.max(
        #     values=tensor([4.2, 4.1]),
        #     indices=tensor([3, 0]))
        #
        # The indices are the class predictions:
        
        max_logits_rowise = torch.max(pred_prob_tns, dim=1)
        pred_class_ids = max_logits_rowise.indices
        
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
        
        conf_matrix = torch.tensor(confusion_matrix(labels_tns,       # Truth
                                                    pred_class_ids,   # Prediction
                                                    labels=list(range(self.num_classes)) # Class labels
                                                    ))

        tally = TrainResult(split_num, self.epoch, learning_phase, conf_matrix)
        self.tally_collection.add(tally)
        return tally

# ----------------- Main --------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()