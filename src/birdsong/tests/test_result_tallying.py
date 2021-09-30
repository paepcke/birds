'''
Created on Sep 29, 2021

@author: paepcke
'''
import os
import unittest

import torch

from birdsong.result_tallying import ResultTally
from birdsong.utils.learning_phase import LearningPhase


TEST_ALL = True
#TEST_ALL = False


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        
    def setUp(self):
        pass


    def tearDown(self):
        pass

# --------------- Tests ---------------

    #------------------------------------
    # test_set_initial_preds
    #-------------------

    def test_set_initial_preds(self):
        
        # Single class, batch size one:
        outputs = torch.tensor([[-5.356]])
        loss    = torch.tensor([0.564])
        
        res_tally = ResultTally(
            0,      # Step
            LearningPhase.TESTING,
            outputs,
            [1],    # label
            loss,
            ['VASEG'], # class_names
            1,      # batch_size
            testing=False
            )
        
        res_tally._set_initial_preds(outputs)
        # Probabilities will be like tensor([[0.0047]])
        # Turn into [0.0047]. The initial value after
        # turning into a list is [[0.00469757616519928]]
        # Round to make comparison predictable:
        res = round(res_tally.probs.tolist()[0][0], 4)
        self.assertEqual(res, 0.0047)
        self.assertListEqual(res_tally.preds, [[0]])
        
        # Single class batch size 2:
        outputs = torch.tensor([[-5.356], [1.5]])
        loss    = torch.tensor([0.564, 0.4])
        
        res_tally._set_initial_preds(outputs)
        
        # Probs will be:
        #   [[0.00469757616519928], [0.8175744414329529]]
        res = [round(prob_list[0], 4) for prob_list in res_tally.probs.tolist()]
        self.assertListEqual(res, [0.0047, 0.8176])
        self.assertListEqual(res_tally.preds, [[0],[1]])
        
        # Single class batch size 2, threshold 0.001:
        res_tally._set_initial_preds(outputs, decision_threshold=0.001)
        
        # Probs will be:
        #   [[0.00469757616519928], [0.8175744414329529]]
        res = [round(prob_list[0], 4) for prob_list in res_tally.probs.tolist()]
        self.assertListEqual(res, [0.0047, 0.8176])
        self.assertListEqual(res_tally.preds, [[1],[1]])
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()