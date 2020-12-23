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
        
        # Larger batch:
        self.ten_results = torch.tensor(           # Label
            [[0.5922, 0.6546, 0.7172, 0.0139],     #   2
        	 [0.9124, 0.9047, 0.6819, 0.9329],     #   3
        	 [0.2345, 0.1733, 0.5420, 0.4659],     #   2
        	 [0.5954, 0.8958, 0.2294, 0.5529],     #   1
        	 [0.3861, 0.2918, 0.0972, 0.0548],     #   0
        	 [0.4647, 0.7002, 0.9632, 0.1320],     #   2
        	 [0.5064, 0.3124, 0.6235, 0.0118],     #   2
        	 [0.3487, 0.6241, 0.8620, 0.4953],     #   2
        	 [0.0386, 0.4663, 0.2362, 0.4898],     #   3
        	 [0.7019, 0.5001, 0.4052, 0.2223]]     #   0
            )
        self.ten_labels_perfect     = torch.tensor([2,3,2,1,0,2,2,2,3,0])
        self.ten_labels_first_wrong = torch.tensor([0,3,2,1,0,2,2,2,3,0])

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

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
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

    #------------------------------------
    # test_per_class_recall 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_per_class_recall(self):
        self.epoch = 1
        # Single split, correct predictions
        # for all 10 samples:
        tally = self.tally_result(
                            0, # Split number
                            self.ten_labels_perfect,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        
        recalls = tally.within_class_recalls()
        truth   = torch.tensor([1.,1.,1.,1.])
        self.assertTrue((recalls == truth).all())
        
        tally = self.tally_result(
                            0, # Split number
                            self.ten_labels_first_wrong,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        recalls = tally.within_class_recalls()
        truth   = torch.tensor([1.,1.,.8,1.])
        self.assertTrue((recalls == truth).all())

    #------------------------------------
    # test_per_class_precision 
    #-------------------
    
    #******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_per_class_precision(self):
        self.epoch = 1
        # Single split, correct predictions
        # for all 10 samples:
        tally = self.tally_result(
                            0, # Split number
                            self.ten_labels_perfect,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        
        precisions = tally.within_class_precisions()
        truth   = torch.tensor([1.,1.,1.,1.])
        self.assertTrue(torch.eq(precisions, truth).all())

        tally = self.tally_result(
                            0, # Split number
                            self.ten_labels_first_wrong,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        precisions = tally.within_class_precisions()
        # Truth is actually [0.6666..., 1.,1.,1.]. So:
        # subtract truth from precision element by element.
        # The first el will be very small, the others will
        # be zero. Sum the elements to get one small float.
        # Assert that the float is within 2 decimal places
        # of zero:
        truth      = torch.tensor([0.67,1.,1.,1.])
        deviation_from_truth = float(torch.sum(precisions - truth))
        self.assertAlmostEqual(deviation_from_truth, 0.0, places=2)

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
