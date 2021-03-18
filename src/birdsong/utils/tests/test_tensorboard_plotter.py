'''
Created on Jan 25, 2021

@author: paepcke
'''

from copy import copy
import os
import random
import shutil
import unittest
import time

from sklearn.metrics import confusion_matrix 
from torch import Size
import torch
from torch.utils.tensorboard import SummaryWriter

from birdsong.result_tallying import ResultTally
from birdsong.rooted_image_dataset import SingleRootImageDataset
from birdsong.utils.learning_phase import LearningPhase
from birdsong.utils.tensorboard_manager import TensorBoardManager
from birdsong.utils.tensorboard_plotter import TensorBoardPlotter


TEST_ALL = True
#TEST_ALL = False

class TestTensorBoardPlotter(unittest.TestCase):


    test_tensorboard_port = 6009

    @classmethod
    def setUpClass(cls):
        
        cls.curr_dir = os.path.dirname(__file__)
        cls.tb_summary_dir = os.path.join(cls.curr_dir, 
                                          'tensorboard_summaries')
                                          
        try:
            shutil.rmtree(cls.tb_summary_dir)
        except FileNotFoundError:
            # Dir not existing is fine:
            pass
        
        # Create dir for tensorboard summaries
        os.makedirs(cls.tb_summary_dir)
        
        cls.sample_spectrogram_path = os.path.join(
            cls.curr_dir,
            '../../tests/data/birds/DYSMEN_S/SONG_Dysithamnusmentalis5114773.png'
            )

        cls.data_root = os.path.join(cls.curr_dir, '../../tests/data/cars')
        
        cls.tb_manager = TensorBoardManager(
            cls.tb_summary_dir,
            port=cls.test_tensorboard_port, 
            new_tensorboard_server=True)

    def setUp(self):
        self.writer = SummaryWriter(self.tb_summary_dir)

    def tearDown(self):
        pass
    
    @classmethod
    def tearDownClass(cls):
        try:
            cls.tb_manager.close()
        except Exception as e:
            print(f"Could not close tensorboard manager: {repr(e)}")

#*****************
#     @classmethod
#     # Remove the tensorboard summaries
#     # that were created during the tests:
#     def tearDownClass(cls):
#         shutil.rmtree(cls.tb_summary_dir)
#*****************
    
    #------------------------------------
    # test_conf_mat_to_tensorboard 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_conf_mat_to_tensorboard (self):
        
        # Add 3 random confusion matrices
        # to tensorboard. Then pause for user
        # input. User can check whether a slider
        # appeared in tensorboard to run through
        # the matrices:
        
        for epoch in range(4):
            
            print(f"Conf matrix to tensorboard: epoch {epoch}")
            # Get a 10x10 confusion matrix and ten
            # fake class names 

            class_names, conf_matrix = self.create_conf_matrix()
            
            plotter = TensorBoardPlotter()
            plotter.conf_matrix_to_tensorboard(self.writer,
                                               conf_matrix,
                                               class_names,
                                               epoch=epoch
                                               )
        self.await_user_ack(f"Reload in browser, then should see confusion matrix\n" +\
                            "Hit key when inspected:")

#         self.await_user_ack(f"Wrote conf matrix for tensorboard to\n" +\
#                             f"{self.tb_summary_dir}.\n" +\
#                             "Hit key when inspected:")
        
    #------------------------------------
    # test_write_img_grid 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_write_img_grid(self):
        '''
        Test creating a grid of train images
        for tensorboard. Method signature is:
        	writer, 
        	num_imgs=4, 
        	img_dirs=None, 
        	img_paths=None,
        	img_height=43,    # px
        	img_width=171,    # px
        	unittesting=False):

        '''
        
        # Test enough imgs in each dir
        # for the desired number of imgs
        # to display. Write summaries to a tmp
        # file underneath this script's dir, which 
        # will be removed later.
        plotter = TensorBoardPlotter()
        grid = plotter.write_img_grid(self.writer,
                                      self.data_root, 
                                      num_imgs=4,
                                      unittesting=False)

        self.assertEqual(grid.shape, torch.Size([3, 220, 1650]))

        self.await_user_ack(f"Reload in browser, then: 12 images of items in Train Input Examples\n" +\
                            "Hit key when inspected:"
                            )

        # Do it again, this time for real.
        # To test:
        #    o Comment out the tearDownClass() method
        #      above to avoid removing the tensorboard
        #      log file at the end of all tests.
        #    o Start a tensorflow service locally, on laptop:
        #       tensorboard tensorboard --logdir <dir-of-this-test-script/tensorboard_summaries>
        #    o In browser:
        #        localhost:6006
        #
        # If testing on a remote server:
        #    o On server in an unused window,
        #      start tensorboard service at remote
        #      server:
        #
        #       tensorboard --logdir <dir-of-this-test-script/tensorboard_summaries>
        #
        #    o On laptop in an unused window
        #
        #       ssh -N -L localhost:60006:localhost:6006 your_uid@server.your.domain
        #
        #      This will hang, and allow a browser 
        #      on the local laptop to reach the tensorflow
        #      server at the remote site.
        #
        #    o On laptop: localhost:60006
        # 
#         grid = plotter.write_img_grid(self.writer,
#                                       self.data_root,
#                                       num_imgs=4,
#                                       img_height=128,
#                                       img_width=512,
#                                       unittesting=False)

    #------------------------------------
    # test_more_img_requests_than_available 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_more_img_requests_than_available(self):
        
        # Test enough imgs in each dir
        # for the desired number of imgs
        # to display. Write summaries to a tmp
        # file underneath this script's dir, which 
        # will be removed later.
        plotter = TensorBoardPlotter()
        grid = plotter.write_img_grid(self.writer,
                                      self.data_root, 
                                      num_imgs=40,
                                      unittesting=False)
        self.assertEqual(grid.shape, Size([3,430,3290]))

    #------------------------------------
    # test_class_support_to_tensorboard 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_class_support_to_tensorboard(self):
        
        plotter = TensorBoardPlotter()
        dataset = SingleRootImageDataset(self.data_root)
        
        support_dict = plotter.class_support_to_tensorboard(dataset,
                                                            self.writer)
        self.assertDictEqual(support_dict,
                             {'audi' : 6,
                              'bmw'  : 6
                              })
                             
        self.await_user_ack(f"Should see both bmw/audo bars to 6.\n" +\
                            "Hit key when inspected:")

#         self.await_user_ack("Wrote class support histogram to \n"+\
#                             f"{self.tb_summary_dir}.\n" +\
#                             f"Should see both bmw/audo bars to 6.\n" +\
#                             "Hit key when inspected:")


    #------------------------------------
    # test_print_onto_image 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_print_onto_image(self):
        
        plotter = TensorBoardPlotter()
        img_tns = plotter.print_onto_image(self.sample_spectrogram_path,
                                           "Hello World"
                                           )
        self.assertEqual(img_tns.shape, torch.Size([1, 4, 128, 512]))

    #------------------------------------
    # test_make_f1_train_val_table 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_make_f1_train_val_table(self):
        '''
        Return a github flavored table:
           |phase|ep0  |ep1 |ep2 |
           |-----|-----|----|----|
           |train| f1_0|f1_1|f1_2|
           |  val| f1_0|f1_1|f1_2|        
        '''

        # Get one train and one val tally
        # with f1_macro/micro/weighted initialized
        # to 0.1, 0.2, 0.3, 0.4, 0.5, and 0.6:
        tally_ep0_train, tally_ep0_val = self.make_tallies() 
        
        # Get another train tally for Epoch 1:
        tally_ep1_train = self.clone_tally(tally_ep0_train,
                                           1.1, 1.2, 1.3
                                           )
        # Same with a validation tally:
        
        # Epoch 1 val:
        tally_ep1_val = self.clone_tally(tally_ep0_val,
                                         1.4, 1.5, 1.6
                                         )
        
        # Table with one tally each for
        # train and val for one epoch:
        test_tallies = [tally_ep0_train, tally_ep0_val] 
        tbl = TensorBoardPlotter.make_f1_train_val_table(test_tallies)
        correct = '|            |f1-macro ep0|\n|------------|------------|\n|  training  |     0.1    |\n|------------|------------|\n| validation |     0.4    |\n|------------|------------|\n'

        self.assertEqual(tbl, correct)
        
        # Add just one tally for epoch 1:
        test_tallies.append(tally_ep1_train)
        # Now have only 1 val and 2 train tallies.
        # Method should complain:
        try:
            tbl = TensorBoardPlotter.make_f1_train_val_table(test_tallies)
            self.fail("Method make_f1_train_val_table() should raise unequal train/val tallies error")
        except ValueError:
            # Succsess: method raised error
            pass
        
        # Add second tally for epoch 1
        test_tallies.append(tally_ep1_val)
        tbl = TensorBoardPlotter.make_f1_train_val_table(test_tallies)
        correct = '|            |f1-macro ep0|f1-macro ep1|\n|------------|------------|------------|\n|  training  |     0.1    |     1.1    |\n|------------|------------|------------|\n| validation |     0.4    |     1.4    |\n|------------|------------|------------|\n'
        self.assertEqual(tbl, correct)
        #print(tbl)

    #------------------------------------
    # test_make_all_classes_f1_table 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_make_all_classes_f1_table(self):

        # Get tallies with f1_all_classes value
        # being for three classes: 
        tally_train, tally_val = self.make_tallies()
        latest_result = {'train' : tally_train,
                         'val'   : tally_val
                         }
        class_names   = ['c1', 'c2', 'c3']
        
        tbl = TensorBoardPlotter.make_all_classes_f1_table(latest_result, 
                                                           class_names)
        
        correct = '|                      |weighted mean f1 train| weighted mean f1 val |\n|----------------------|----------------------|----------------------|\n|          c1          |          0.7         |          1.7         |\n|----------------------|----------------------|----------------------|\n|          c2          |          0.8         |          1.8         |\n|----------------------|----------------------|----------------------|\n|          c3          |          0.9         |          1.9         |\n|----------------------|----------------------|----------------------|\n'
        self.assertEqual(tbl, correct)
        
        # Now try having only one row:
        tally_train.f1_all_classes = [0.1]
        tally_val.f1_all_classes = [0.2]
        class_names = ['Just one class']
        
        tbl = TensorBoardPlotter.make_all_classes_f1_table(latest_result, 
                                                           class_names)
        
        correct = '|                      |weighted mean f1 train| weighted mean f1 val |\n|----------------------|----------------------|----------------------|\n|    Just one class    |          0.1         |          0.2         |\n|----------------------|----------------------|----------------------|\n'
        self.assertEqual(tbl, correct)
        
        # Finally: Empty list or None for 
        # one of the tallies: should fail:
        tally_val.f1_all_classes = []
        try:
            TensorBoardPlotter.make_all_classes_f1_table(latest_result, 
                                                            class_names)
            self.fail("Validation tally's f1_all_classes is empty, and should have raised exception")
        except ValueError:
            pass

# ------------------ Utils ---------------

    #------------------------------------
    # make_tallies
    #-------------------
    
    def make_tallies(self):
        '''
        Return two tallies, one 
        train, one val. Client can 
        use clone_tally() to replicate
        and then modify the f1 values
        '''
        preds = torch.tensor([[0.1,0.2,0.3],
                              [0.4,0.5,0.6]
                              ])
        tally_train = ResultTally(
            0,  # epoch
            LearningPhase.TRAINING,
            preds,                     # outputs
            torch.tensor([1,2,3,4]),   # labels,
            torch.tensor(0.5),         # loss,
            3,                         # num_classes
            64,                        # batch_size
            testing=True
            )                

        # Set f1 values, then use 
        # this tally as a blueprint
        # to clone others:
        # Epoch 0 train
        tally_train.f1_macro = 0.1
        tally_train.f1_micro = 0.2
        tally_train.f1_weighted = 0.3
        
        # Assume three classes, and 
        # create three f1 values, one
        # for each class:
        tally_train.f1_all_classes = [0.7, 0.8, 0.9]

        # Epoch 0 val
        tally_val = copy(tally_train)
        tally_val.f1_macro = 0.4
        tally_val.f1_micro = 0.5
        tally_val.f1_weighted = 0.6
        tally_val.phase = LearningPhase.VALIDATING
        tally_val.f1_all_classes = [1.7, 1.8, 1.9]
        
        return(tally_train, tally_val)

    #------------------------------------
    # clone_tally
    #-------------------
    
    def clone_tally(self, 
                    tally,
                    f1_macro, 
                    f1_micro, 
                    f1_weighted,
                    phase=None):

        tally_clone = copy(tally)
        
        tally_clone.f1_macro = f1_macro
        tally_clone.f1_micro = f1_micro
        tally_clone.f1_weighted = f1_weighted
        
        if phase is not None:
            tally_clone.phase = phase
            
        return tally_clone

    #------------------------------------
    # gen_pred_truth 
    #-------------------

    def gen_pred_truth(self, num_classes, perc_correct):
        '''
        Given a number of classes, and a percentage 
        of hypothetically correct predictions, generate
        two sequences[int] of length num_classes: First,
        a sequence that represents true target class labels. 
        Second, a sequence that matches the truth labels
        perc_correct number of times.

        @param num_classes: number of hypothetical 
            classes
        @type num_classes: int
        @param perc_correct: percentage of hypothetial
            class predictions that are to match the truth
        @type perc_correct: float
        @return: a dict: {'truth': [int], 'pred' : [int]}
        '''
        
        truth       = torch.randperm(num_classes)
        pred        = torch.randperm(num_classes)
        
        # Number of predictions we need to 
        # modify so they predict correctly:
        
        num_correct = round(num_classes * perc_correct + 0.5)
        
        # Pick num_correct random indices
        # into the truth, and make the corresponding
        # pred match:
        
        for _i in range(num_correct):
            truth_pos = random.randint(0,num_classes-1)
            pred[truth_pos] = truth[truth_pos]
            
        return {'truth' : truth, 'pred' : pred}

    #------------------------------------
    # create_conf_matrix
    #-------------------
    
    def create_conf_matrix(self, 
                               num_classes=10,
                               perc_correct=0.25):
        '''
        Creates and returns random confusion
        matrix, and corresponding fake class
        names.
        
        @returns tuple of class name list, and the
           confusion matrix
        @rtype ([str], np.ndarray)
        '''
        
        # Get dict {'truth' : <tensor>, 'pred' : <tensor>}
        # with ten classes, and 25% correct answers
        
        pred_outcome = self.gen_pred_truth(num_classes,
                                           perc_correct)
        # Rows will be Truths, cols will be Preds:
        conf_matrix = confusion_matrix(pred_outcome['truth'],
                                       pred_outcome['pred'],
                                       )
        # Generate fantasy class names:
        class_names = [f'class_num-{idx}' 
                       for idx 
                       in range(num_classes)]
        
        return class_names, conf_matrix
        
    #------------------------------------
    # await_user_ack 
    #-------------------
    
    def await_user_ack(self, msg=None):
        '''
        Print given msg or the default msg,
        then block till user hits any key.
        
        Used when images are sent to tensorboard,
        and user needs to check them. The end
        of the test suite will remove the tensorflow
        event files; so then it would be too late. 
        
        @param msg: prompt to print
        @type msg: {str | None}
        '''
        if msg is None:
            msg = "Wrote a record to Tensorboard; hit \n" +\
                  "ENTER when manually checked. Record \n" +\
                  "will then be deleted."
        print(msg)
        input("Waiting...")

# ------------------ Main ---------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()