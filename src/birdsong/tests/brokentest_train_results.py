'''
Created on Dec 23, 2020

@author: paepcke
'''
import os, sys
packet_root = os.path.abspath(__file__.split('/')[0])
sys.path.insert(0,packet_root)


import unittest
#from sklearn.metrics import confusion_matrix
import torch

from birdsong.birds_train_parallel import LearningPhase
from birdsong.result_tallying import ResultCollection, ResultTally


TEST_ALL = True
#TEST_ALL = False

class Test(unittest.TestCase):

    #------------------------------------
    # setUp 
    #-------------------

    def setUp(self):
        self.tally_collection = ResultCollection()
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
        tally = self.tally_result(
                            self.single_label_matching,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch, 1)
        # self.assertEqual(tally.num_samples, 1)
        # self.assertEqual(tally.num_correct, 1)
        # self.assertEqual(tally.num_wrong, 0)

        tally = self.tally_result(
                            self.single_label_non_match,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch, 1)
        # self.assertEqual(tally.num_samples, 1)
        # self.assertEqual(tally.num_correct, 0)
        # self.assertEqual(tally.num_wrong, 1)

    #------------------------------------
    # test_basics_two_splits
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_basics_two_splits(self):
        tally = self.tally_result(
                            self.batch_label_matching,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch, 1)
        # self.assertEqual(tally.num_samples, 2)
        # self.assertEqual(tally.num_correct, 2)
        # self.assertEqual(tally.num_wrong, 0)

        tally = self.tally_result(
                            self.batch_label_non_match,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch, 1)
        # self.assertEqual(tally.num_samples, 2)
        # self.assertEqual(tally.num_correct, 1)
        # self.assertEqual(tally.num_wrong, 1)

    #------------------------------------
    # test_accuracy
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_accuracy(self):
        # Single split, correct prediction
        tally = self.tally_result(
                            self.single_label_matching,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy, 1)

        # Single split, incorrect prediction
        tally = self.tally_result(
                            self.single_label_non_match,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy, 0)

        # Two splits, correct predictions
        tally = self.tally_result(
                            self.batch_label_matching,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy, 1)

        # Two splits, incorrect predictions
        tally = self.tally_result(
                            self.batch_label_non_match,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy, 0.5)

    #------------------------------------
    # test_result_collection_generator 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_result_collection_generator(self):
        '''
        Generator functionality of TrainCollection.
        Should deliver sequence of ResultTally instances.
        '''
        # Epoch 1, learning phase TRAINING
        _tally_ep1_lp_train1 = self.tally_result(
                                   self.ten_labels_perfect,
                                   self.ten_results,
                                   LearningPhase.TRAINING,
                                   epoch=1
                                   )
        # Epoch 2, learning phase TRAINING
        _tally_ep2_lp_train2 = self.tally_result(
                                  self.ten_labels_perfect,
                                  self.ten_results,
                                  LearningPhase.TRAINING,
                                  epoch=2
                                  )
        # Epoch 3, learning phase TRAINING
        _tally_ep3_lp_train3 = self.tally_result(
                              self.ten_labels_first_wrong,
                              self.ten_results,
                              LearningPhase.TRAINING,
                              epoch=3
                              )
        # Second Epoch 1 result:
        _tally_ep1_lp_test1 = self.tally_result(
                              self.ten_labels_first_wrong,
                              self.ten_results,
                              LearningPhase.TESTING,
                              epoch=1
                              )
        
        tallies_sorted = [_tally_ep1_lp_train1,
                          _tally_ep2_lp_train2,
                          _tally_ep3_lp_train3,
                          _tally_ep1_lp_test1
                          ]
        
        # All tallies, sorted by time:
        tallies = list(self.tally_collection.tallies())
        self.assertEqual(tallies, tallies_sorted)

        # All TRAINING tallies, sorted by time:
        tallies = list(self.tally_collection.tallies(
            learning_phase=LearningPhase.TRAINING))
        self.assertEqual(tallies, tallies_sorted[:3])
        
        # All TESTING tallies, sorted by time:
        tallies = list(self.tally_collection.tallies(
            learning_phase=LearningPhase.TESTING))
        self.assertTrue(tallies[0] == tallies_sorted[3])

        # All tallies, sorted by time, but only testing in epoch 2:
        tallies = list(self.tally_collection.tallies(
            epoch=2,
            learning_phase=LearningPhase.TESTING))

    #------------------------------------
    # test_collection_num_classes 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_collection_num_classes(self):
        '''
        Whether collections properly ask their
        first ResultTally instance for the number
        of classes
        '''
        
        # Nothing added to collection, num_classes
        # should be 0
        self.assertEqual(len(self.tally_collection), 0)
                         
        
        _tally1 = self.tally_result(
                       self.ten_labels_perfect,
                       self.ten_results,
                       LearningPhase.TRAINING,
                       epoch=1
                       )
        # Epoch 1, learning phase TRAINING
        _tally2 = self.tally_result(
                       self.ten_labels_perfect,
                       self.ten_results,
                       LearningPhase.TRAINING,
                       epoch=1
                       )
        self.tally_collection.add(_tally1, 1)
        self.tally_collection.add(_tally2, 1)
        # Because results are equal should still only
        # have one result in collection:
        self.assertEqual(len(self.tally_collection), 1)

    #------------------------------------
    # test_copy 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_copy(self):
        tally1 = self.tally_result(
                       self.ten_labels_perfect,
                       self.ten_results,
                       LearningPhase.TRAINING,
                       epoch=1
                       )
        new_col = ResultCollection.create_from(self.tally_collection)
        
        # Contents of new collection should be same:
        self.assertEqual(len(new_col), 1)
        new_tally = list(new_col.tallies())[0]
        self.assertTrue(new_tally == tally1)
        
        for tally_old, tally_new in zip(self.tally_collection.tallies(),
                                        new_col.tallies()):
            self.assertTrue(tally_old == tally_new)

    # ****** Needs thinking and debugging in result_tallying
#     #------------------------------------
#     # test_within_class_recall_aggregation 
#     #-------------------
#     
#     #****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
#     def test_within_class_recall_aggregation(self):
#         tally1 = self.tally_result(
#                               0, # Split number
#                               self.single_label_matching,
#                               self.single_pred,
#                               LearningPhase.TRAINING
#                               )
#         tally2 = self.tally_result(
#                               0, # Split number
#                               self.single_label_non_match,
#                               self.single_pred,
#                               LearningPhase.TRAINING
#                               )
#         # Because only one class represented,
#         # the others will be nan:
#         within_class_recalls1 = tally1.within_class_recalls()
#         within_class_recalls2 = tally2.within_class_recalls()
# 
#         agg_within_class_recall = (within_class_recalls1 + within_class_recalls2) / 2.0
#         for idx in range(len(agg_within_class_recall)):
#             if idx in [0,1,3]:
#                 self.assertTrue(torch.isnan(agg_within_class_recall[idx]))
#             else:
#                 self.assertEqual(agg_within_class_recall[idx], 0.5)
# 
#         # Larger batch:
#  
#         tally1 = self.tally_result(
#                             0, # Split number
#                             self.ten_labels_perfect,
#                             self.ten_results,
#                             LearningPhase.TRAINING
#                             )
# 
#         tally2 = self.tally_result(
#                             0, # Split number
#                             self.ten_labels_first_wrong,
#                             self.ten_results,
#                             LearningPhase.TRAINING
#                             )
#         
#         recalls1 = tally1.within_class_recalls()
#         recalls2 = tally2.within_class_recalls()
#         
#         mean_within_class_recall = self.tally_collection.mean_within_class_recall()
#         print('foo')

# ---------------- Utils ------------

    #------------------------------------
    # tally_result
    #-------------------

    def tally_result(self,
                     labels_tns, 
                     pred_prob_tns,
                     learning_phase,
                     epoch=1
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

        # Use a random loss value:
        loss = torch.tensor(0.14)
        batch_size = 2
        tally = ResultTally(epoch, 
                            learning_phase, 
                            pred_prob_tns,
                            labels_tns,
                            loss, 
                            self.num_classes,
                            batch_size)
                            
        self.tally_collection.add(tally)
        return tally

# ----------------- Main --------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
