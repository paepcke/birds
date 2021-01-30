'''
Created on Jan 25, 2021

@author: paepcke
'''

import os
import shutil
import unittest

from sklearn.metrics import confusion_matrix 
from torch import Size
import torch
from torch.utils.tensorboard import SummaryWriter

from birdsong.utils.tensorboard_plotter import TensorBoardPlotter


#******TEST_ALL = True
TEST_ALL = False

class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.tb_summary_dir = os.path.join(cls.curr_dir, 
                                          'tensorboard_summaries')
                                          
        
        if not os.path.exists(cls.tb_summary_dir):
            os.makedirs(cls.tb_summary_dir)
        

    def setUp(self):
        self.writer = SummaryWriter(self.tb_summary_dir)

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tb_summary_dir)
    
    #------------------------------------
    # test_ 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_cm_creation(self):
        
        # Get dict {'truth' : <tensor>, 'pred' : <tensor>}
        # with ten classes, and 25% correct answers
        
        num_classes = 10
        pred_outcome = self.classes10_corr0_25 = self.gen_pred_truth(num_classes,
                                                                     0.25)
        conf_matrix = confusion_matrix(pred_outcome['truth'],
                                       pred_outcome['pred'],
                                       )
        class_names = [f'class_num-{idx}' for idx in range(num_classes)]
        
        plotter = TensorBoardPlotter()
        cm_ax = plotter.fig_from_conf_matrix(self, 
                                             conf_matrix,
                                             class_names,
                                             title='Confusion Matrix')
        
        
        cm_img = plotter.plot_confusion_matrix(
                pred_outcome['truth'],
                pred_outcome['pred'],
                class_names,
                title="Test Figure",
                tensor_name="TestFig/image",
                normalize=False
                )

        #img_d_summary_writer.add_summary(img_d_summary, current_step)
        self.writer.add_image('conf_matrix', 
                              cm_img,
                              global_step=1
                              )
        print('foo') 
        
    #------------------------------------
    # test_write_img_grid 
    #-------------------
    
    #****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
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
                                      unittesting=True)
        # self.assertEqual(grid.shape, Size([3, 175, 92]))
        self.assertEqual(grid.shape, torch.Size([3, 47, 694]))
        
        # Do it again, this time for real:
        grid = plotter.write_img_grid(self.writer,
                                      data_root,
                                      num_imgs=4,
                                      img_height=128,
                                      img_width=512,
                                      unittesting=False)
        print(grid)
        


# ------------------ Utils ---------------

    def gen_pred_truth(self, num_classes, perc_correct):
        
        truth       = (torch.rand(num_classes)*10).int()
        pred        = (torch.rand(num_classes)*10).int()
        
        num_correct = round(num_classes * perc_correct + 0.5)
        self.true_pos_idxs = torch.randperm(num_correct)
        
        for true_pos_idx in self.true_pos_idxs:
            pred[true_pos_idx] = truth[true_pos_idx]
        return {'truth' : truth, 'pred' : pred}
        
        
    
# ------------------ Main ---------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()