'''
Created on Dec 23, 2020

@author: paepcke
'''
import os,sys

import statistics
import unittest

from sklearn import metrics
#from sklearn.metrics import confusion_matrix
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from birds_train_parallel import LearningPhase
from birds_train_parallel import TrainResultCollection, TrainResult


TEST_ALL = True
#TEST_ALL = False

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
        tally = self.tally_result(
                            0, # Split number
                            self.single_label_matching,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch, 1)
        self.assertEqual(tally.split_num, 0)
        self.assertEqual(tally.num_samples, 1)
        self.assertEqual(tally.num_correct, 1)
        self.assertEqual(tally.num_wrong, 0)

        tally = self.tally_result(
                            0, # Split number
                            self.single_label_non_match,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch, 1)
        self.assertEqual(tally.split_num, 0)
        self.assertEqual(tally.num_samples, 1)
        self.assertEqual(tally.num_correct, 0)
        self.assertEqual(tally.num_wrong, 1)

    #------------------------------------
    # test_basics_two_splits
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_basics_two_splits(self):
        tally = self.tally_result(
                            0, # Split number
                            self.batch_label_matching,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch, 1)
        self.assertEqual(tally.split_num, 0)
        self.assertEqual(tally.num_samples, 2)
        self.assertEqual(tally.num_correct, 2)
        self.assertEqual(tally.num_wrong, 0)

        tally = self.tally_result(
                            0, # Split number
                            self.batch_label_non_match,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.epoch, 1)
        self.assertEqual(tally.split_num, 0)
        self.assertEqual(tally.num_samples, 2)
        self.assertEqual(tally.num_correct, 1)
        self.assertEqual(tally.num_wrong, 1)

    #------------------------------------
    # test_accuracy
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_accuracy(self):
        # Single split, correct prediction
        tally = self.tally_result(
                            0, # Split number
                            self.single_label_matching,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy, 1)

        # Single split, incorrect prediction
        tally = self.tally_result(
                            0, # Split number
                            self.single_label_non_match,
                            self.single_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy, 0)

        # Two splits, correct predictions
        tally = self.tally_result(
                            0, # Split number
                            self.batch_label_matching,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy, 1)

        # Two splits, incorrect predictions
        tally = self.tally_result(
                            0, # Split number
                            self.batch_label_non_match,
                            self.batch_pred,
                            LearningPhase.TRAINING
                            )
        self.assertEqual(tally.accuracy, 0.5)

    #------------------------------------
    # test_per_class_recall 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_per_class_recall(self):
        # Single split, correct predictions
        # for all 10 samples:
        tally = self.tally_result(
                            0, # Split number
                            self.ten_labels_perfect,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        
        recalls = tally.within_class_recalls
        truth   = torch.tensor([1.,1.,1.,1.])
        self.assertTrue((recalls == truth).all())
        
        tally = self.tally_result(
                            0, # Split number
                            self.ten_labels_first_wrong,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        recalls = tally.within_class_recalls
        truth   = torch.tensor([1.,1.,.8,1.])
        self.assertTrue((recalls == truth).all())

    #------------------------------------
    # test_per_class_precision 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_per_class_precision(self):
        # Single split, correct predictions
        # for all 10 samples:
        tally = self.tally_result(
                            0, # Split number
                            self.ten_labels_perfect,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        
        precisions = tally.within_class_precisions
        truth   = torch.tensor([1.,1.,1.,1.])
        self.assertTrue(torch.eq(precisions, truth).all())

        tally = self.tally_result(
                            0, # Split number
                            self.ten_labels_first_wrong,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        precisions = tally.within_class_precisions
        # Truth is actually [0.6666..., 1.,1.,1.]. So:
        # subtract truth from precision element by element.
        # The first el will be very small, the others will
        # be zero. Sum the elements to get one small float.
        # Assert that the float is within 2 decimal places
        # of zero:
        truth      = torch.tensor([0.67,1.,1.,1.])
        deviation_from_truth = float(torch.sum(precisions - truth))
        self.assertAlmostEqual(deviation_from_truth, 0.0, places=2)

    #------------------------------------
    # test_recall 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_recall(self):

        # Single split, correct predictions
        # for all 10 samples:
        _tally1 = self.tally_result(
                            0, # Split number
                            self.ten_labels_perfect,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        
        #prec_tally1 = tally.within_class_precisions()
        #truth   = torch.tensor([1.,1.,1.,1.])
        #self.assertTrue(torch.eq(precisions, truth).all())

        tally2 = self.tally_result(
                            0, # Split number
                            self.ten_labels_first_wrong,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        recall_tally1_2 = tally2.within_class_recalls
        self.assertTrue(recall_tally1_2.eq(torch.tensor([1.0000, 1.0000, 0.8000, 1.0000])).all())
        
        self.assertEqual(tally2.recall,
                         metrics.recall_score(self.ten_labels_first_wrong,
                                      torch.argmax(self.ten_results, dim=1),
                                      average='weighted'))


    #------------------------------------
    # test_precision 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_precision(self):

        # Single split, correct predictions
        # for all 10 samples:
        _tally = self.tally_result(
                            0, # Split number
                            self.ten_labels_perfect,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        
        #prec_tally1 = tally.within_class_precisions()
        #truth   = torch.tensor([1.,1.,1.,1.])
        #self.assertTrue(torch.eq(precisions, truth).all())

        tally = self.tally_result(
                            0, # Split number
                            self.ten_labels_first_wrong,
                            self.ten_results,
                            LearningPhase.TRAINING
                            )
        prec_tally1_2 = tally.within_class_precisions
        prec_rounded = torch.round(prec_tally1_2 * 10**2) / (10**2)
        self.assertTrue(prec_rounded.eq(torch.tensor([0.6700, 1.0000, 1.0000, 1.0000])).all())
        
        mean_precision = tally.precision
        predictions = torch.argmax(self.ten_results, dim=1)
        self.assertEqual(mean_precision,
                         metrics.precision_score(self.ten_labels_first_wrong,
                                                 predictions,
                                                 average='weighted'
                                                 ))

    #------------------------------------
    # test_accuracy_aggregation 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_accuracy_aggregation(self):
        
        tally1 = self.tally_result(
                              0, # Split number
                              self.single_label_matching,
                              self.single_pred,
                              LearningPhase.TRAINING
                              )
        tally2 = self.tally_result(
                              1, # Split number
                              self.single_label_non_match,
                              self.single_pred,
                              LearningPhase.TRAINING
                              )
        acc1 = tally1.accuracy
        acc2 = tally2.accuracy
        
        mean_acc = (acc1+acc2)/2
        
        mean_accuracy = self.tally_collection.mean_accuracy()
        self.assertEqual(mean_accuracy, mean_acc.item()) # 0.5

        tally1 = self.tally_result(
                              0, # Split number
                              self.ten_labels_perfect,
                              self.ten_results,
                              LearningPhase.TRAINING
                              )

    #------------------------------------
    # test_within_epoch_accuracy 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_within_epoch_accuracy(self):
        
        # Epoch 1 result
        _tally1 = self.tally_result(
                              0, # Split number
                              self.ten_labels_perfect,
                              self.ten_results,
                              LearningPhase.TRAINING,
                              epoch=1
                              )
        # Epoch 2 result:
        _tally2 = self.tally_result(
                              0, # Split number
                              self.ten_labels_first_wrong,
                              self.ten_results,
                              LearningPhase.TESTING,
                              epoch=2
                              )
        # Second Epoch 2 result:
        _tally3 = self.tally_result(
                              0, # Split number
                              self.ten_labels_first_wrong,
                              self.ten_results,
                              LearningPhase.TESTING,
                              epoch=2
                              )

        # Accuracy over epoch 1, Training phase (1 tally)
        epoch1_mean_acc = self.tally_collection.mean_accuracy(epoch=1,
                                                              learning_phase=LearningPhase.TRAINING
                                                              )
        self.assertEqual(epoch1_mean_acc, 1.0)
        
        # Accuracy over epoch 2, Testing phase (2 tallies)
        epoch2_mean_acc = self.tally_collection.mean_accuracy(epoch=2,
                                                              learning_phase=LearningPhase.TESTING
                                                              )
        self.assertEqual(epoch2_mean_acc, 0.9)

        true_mean = (0.9 + 0.9) / 2
        epoch2_3_mean_acc = self.tally_collection.mean_accuracy(epoch=2,
                                                                learning_phase=LearningPhase.TESTING
                                                                )
        self.assertEqual(epoch2_3_mean_acc, true_mean)

    #------------------------------------
    # test_loss_accumulation 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_loss_accumulation(self):
        
        loss1 = torch.tensor(10.0)
        # Add to epoch 1:
        self.tally_collection.add_loss(1, loss1)
        # Over all epochs:
        self.assertEqual(self.tally_collection.cumulative_loss(),
                         loss1
                         )
        
        # Just over epoch 1:
        self.assertEqual(self.tally_collection.cumulative_loss(epoch=1),
                         loss1
                         )

        # Add another loss to epoch one:
        loss2 = torch.tensor(20.0)
        self.tally_collection.add_loss(1, loss2)
        
        # Over all epochs:
        self.assertEqual(self.tally_collection.cumulative_loss(),
                         loss1 + loss2
                         )
        
        # Just epoch1:
        self.assertEqual(self.tally_collection.cumulative_loss(epoch=1),
                         loss1 + loss2
                         )

        loss3 = torch.tensor(30)

        # Add another epoch to track:
        self.tally_collection.add_loss(2, loss3)

        # Just epoch1:
        self.assertEqual(self.tally_collection.cumulative_loss(epoch=1),
                         loss1 + loss2
                         )

        # Just epoch2:
        self.assertEqual(self.tally_collection.cumulative_loss(epoch=2),
                         loss3
                         )

    #------------------------------------
    # test_collection_wide_precision 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_collection_wide_precision(self):

        # Epoch 1 result
        _tally1 = self.tally_result(
                              0, # Split number
                              self.ten_labels_perfect,
                              self.ten_results,
                              LearningPhase.TRAINING,
                              epoch=1
                              )
        # Epoch 2 result:
        _tally2 = self.tally_result(
                              0, # Split number
                              self.ten_labels_first_wrong,
                              self.ten_results,
                              LearningPhase.TESTING,
                              epoch=2
                              )
        # Second Epoch 2 result:
        _tally3 = self.tally_result(
                              0, # Split number
                              self.ten_labels_first_wrong,
                              self.ten_results,
                              LearningPhase.TESTING,
                              epoch=2
                              )



        # Epoch with 1 result in Training phase
        our_res = self.tally_collection.mean_weighted_precision(epoch=1,
                                                                learning_phase=LearningPhase.TRAINING
                                                                ) 
        self.assertEqual(our_res, _tally1.precision_weighted)


        # Epoch with 2 results in Testing phase
        true_mean_weighted_precision = statistics.mean([
                                                        _tally2.precision, 
                                                        _tally3.precision])
        our_res = self.tally_collection.mean_weighted_precision(epoch=2,
                                                                learning_phase=LearningPhase.TESTING
                                                                ) 
        self.assertEqual(round(our_res,2), round(true_mean_weighted_precision,2))

    #------------------------------------
    # test_collection_wide_recall 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_collection_wide_recall(self):

        # Epoch 1 result
        _tally1 = self.tally_result(
                              0, # Split number
                              self.ten_labels_perfect,
                              self.ten_results,
                              LearningPhase.TRAINING,
                              epoch=1
                              )
        # Epoch 2 result:
        _tally2 = self.tally_result(
                              0, # Split number
                              self.ten_labels_first_wrong,
                              self.ten_results,
                              LearningPhase.TESTING,
                              epoch=2
                              )
        # Second Epoch 2 result:
        _tally3 = self.tally_result(
                              0, # Split number
                              self.ten_labels_first_wrong,
                              self.ten_results,
                              LearningPhase.TESTING,
                              epoch=2
                              )



        # Epoch with 1 result in Training phase
        our_res = self.tally_collection.mean_weighted_recall(epoch=1,
                                                             learning_phase=LearningPhase.TRAINING
                                                             ) 
        self.assertEqual(our_res, _tally1.recall_weighted)


        # Epoch with 2 results in Testing phase
        true_mean_weighted_recall = statistics.mean([_tally2.recall, 
                                                     _tally3.recall])
        our_res = self.tally_collection.mean_weighted_recall(epoch=2,
                                                             learning_phase=LearningPhase.TESTING
                                                             ) 
        self.assertEqual(round(our_res,2), round(true_mean_weighted_recall,2))

    #------------------------------------
    # test_result_collection_generator 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_result_collection_generator(self):
        '''
        Generator functionality of TrainCollection.
        Should deliver sequence of TrainResult instances.
        '''
        # Epoch 1, learning phase TRAINING
        _tally_ep1_lp_train1 = self.tally_result(
                                   0, # Split number
                                   self.ten_labels_perfect,
                                   self.ten_results,
                                   LearningPhase.TRAINING,
                                   epoch=1
                                   )
        # Epoch 1, learning phase TRAINING
        _tally_ep1_lp_train2 = self.tally_result(
                                  1, # Split number
                                  self.ten_labels_perfect,
                                  self.ten_results,
                                  LearningPhase.TRAINING,
                                  epoch=1
                                  )
        # Epoch 2, learning phase TRAINING
        _tally_ep2_lp_train1 = self.tally_result(
                              0, # Split number
                              self.ten_labels_first_wrong,
                              self.ten_results,
                              LearningPhase.TRAINING,
                              epoch=2
                              )
        # Second Epoch 2 result:
        _tally_ep2_lp_test1 = self.tally_result(
                              0, # Split number
                              self.ten_labels_first_wrong,
                              self.ten_results,
                              LearningPhase.TESTING,
                              epoch=2
                              )
        
        tallies_sorted = [_tally_ep1_lp_train1,
                          _tally_ep1_lp_train2,
                          _tally_ep2_lp_train1,
                          _tally_ep2_lp_test1
                          ]
        
        # All tallies, sorted by time:
        tallies = list(self.tally_collection.tallies())
        self.assertEqual(tallies, tallies_sorted)

        # All tallies, sorted by time, but only epoch 1:
        tallies = list(self.tally_collection.tallies(epoch=1))
        self.assertEqual(tallies, tallies_sorted[:2])
        
        # All tallies, sorted by time, but only training:
        tallies = list(self.tally_collection.tallies(learning_phase=LearningPhase.TRAINING))
        self.assertEqual(tallies, tallies_sorted[:3])

        # All tallies, sorted by time, but only testing in epoch 2:
        tallies = list(self.tally_collection.tallies(epoch=2,
                                                     learning_phase=LearningPhase.TESTING))
        self.assertEqual(tallies, [_tally_ep2_lp_test1])

    #------------------------------------
    # test_collection_num_classes 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_collection_num_classes(self):
        '''
        Whether collections properly ask their
        first TrainResult instance for the number
        of classes
        '''
        
        # Nothing added to collection, num_classes
        # should be 0
        self.assertEqual(self.tally_collection.num_classes, 0)
                         
        
        _tally1 = self.tally_result(
                       0, # Split number
                       self.ten_labels_perfect,
                       self.ten_results,
                       LearningPhase.TRAINING,
                       epoch=1
                       )
        # Epoch 1, learning phase TRAINING
        _tally2 = self.tally_result(
                       1, # Split number
                       self.ten_labels_perfect,
                       self.ten_results,
                       LearningPhase.TRAINING,
                       epoch=1
                       )
        
        self.assertEqual(self.tally_collection.num_classes,
                         self.num_classes
                         )

    #------------------------------------
    # test_copy 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_copy(self):
        tally1 = self.tally_result(
                       0, # Split number
                       self.ten_labels_perfect,
                       self.ten_results,
                       LearningPhase.TRAINING,
                       epoch=1
                       )
        new_col = TrainResultCollection.create_from(self.tally_collection)
        
        # Contents of new collection should be same:
        self.assertEqual(len(new_col), 1)
        new_tally = list(new_col.tallies())[0]
        
        self.assertEqual(new_tally.split_num, tally1.split_num)
        self.assertEqual(new_tally.epoch, tally1.epoch)
        
        # But the contained tallies must not be equal: 
        self.assertNotEqual(new_tally, tally1)


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
                     split_num, 
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
        loss = 0.14
        tally = TrainResult(split_num, 
                            epoch, 
                            learning_phase, 
                            loss, 
                            pred_class_ids,
                            labels_tns,
                            self.num_classes,
                            badly_predicted_labels=None)
                            
        self.tally_collection.add(tally)
        return tally

        


# ----------------- Main --------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
