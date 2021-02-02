'''
Created on Jan 25, 2021

@author: paepcke
'''

import os
import random
import shutil
import unittest

from sklearn.metrics import confusion_matrix 
from torch import Size
import torch
from torch.utils.tensorboard import SummaryWriter

from birdsong.utils.tensorboard_plotter import TensorBoardPlotter


TEST_ALL = True
#***** TEST_ALL = False

class TestTensorBoardPlotter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.tb_summary_dir = os.path.join(cls.curr_dir, 
                                          'tensorboard_summaries')
                                          
        
        if not os.path.exists(cls.tb_summary_dir):
            os.makedirs(cls.tb_summary_dir)
            
        cls.sample_spectrogram_path = os.path.join(
            cls.curr_dir,
            '../../tests/data/birds/DYSMEN_S/SONG_Dysithamnusmentalis5114773.png'
            )

    def setUp(self):
        self.writer = SummaryWriter(self.tb_summary_dir)

    def tearDown(self):
        pass

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
            cm_ax = plotter.fig_from_conf_matrix(conf_matrix,
                                                 class_names,
                                                 title='Confusion Matrix')
    
            plotter.plot_fig_to_tensorboard(self.writer, 
                                            cm_ax.figure,
                                            epoch)
        self.await_user_ack(f"Wrote conf matrix for tensorboard to\n" +\
                            f"{self.tb_summary_dir}.\n" +\
                            "Hit key when inspected:")
        
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
        # This unittest is under the utils dir.
        # The test img tree is under the main
        # test dir:
        
        data_root = os.path.join(self.curr_dir, '../../tests/data/cars')
        
        # Test enough imgs in each dir
        # for the desired number of imgs
        # to display. Write summaries to a tmp
        # file underneath this script's dir, which 
        # will be removed later.
        plotter = TensorBoardPlotter(self.tb_summary_dir)
        grid = plotter.write_img_grid(self.writer,
                                      data_root, 
                                      num_imgs=4,
                                      unittesting=False)

        self.assertEqual(grid.shape, torch.Size([3, 220, 1650]))

        self.await_user_ack(f"Wrote tensorboard sample imgs to\n" +\
                            f"{self.tb_summary_dir}.\n" +\
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
#                                       data_root,
#                                       num_imgs=4,
#                                       img_height=128,
#                                       img_width=512,
#                                       unittesting=False)

    #------------------------------------
    # test_more_img_requests_than_available 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_more_img_requests_than_available(self):
        
        data_root = os.path.join(self.curr_dir, '../../tests/data/cars')
        
        # Test enough imgs in each dir
        # for the desired number of imgs
        # to display. Write summaries to a tmp
        # file underneath this script's dir, which 
        # will be removed later.
        plotter = TensorBoardPlotter(self.tb_summary_dir)
        grid = plotter.write_img_grid(self.writer,
                                      data_root, 
                                      num_imgs=40,
                                      unittesting=False)
        self.assertEqual(grid.shape, Size([3,430,3290]))

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


# ------------------ Utils ---------------

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