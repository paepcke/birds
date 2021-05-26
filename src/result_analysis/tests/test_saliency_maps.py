'''
Created on May 24, 2021

@author: paepcke
'''
import os
import unittest

from birdsong.utils.utilities import FileUtils
from data_augmentation.utils import Utils
from result_analysis.saliency_maps import SaliencyMapper


#*******TEST_ALL = True
TEST_ALL = False

class SaliencyMapTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.model_path = os.path.join(
            cls.cur_dir, 
            '../../birdsong/tests/models/mod_2021-05-04T13_02_14_net_resnet18_pre_True_frz_0_lr_0.01_opt_SGD_bs_128_ks_7_folds_10_gray_True_classes_34_ep9.pth'
            )
        cls.snips_dir = os.path.join(
            cls.cur_dir,
            '../../birdsong/utils/tests/data/fld_snippets'
            )
        cls.example_img_path = Utils.listdir_abs(cls.snips_dir)[0]
        
    def setUp(self):
        self.mapper = SaliencyMapper(self.model_path, self.snips_dir, unittesting=True)

    def tearDown(self):
        pass

# ----------------------- Test Routines ---------

    #------------------------------------
    # test_materialize_model 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_materialize_model(self):
        
        self.mapper.materialize_model(self.model_path)
        # Test some random fact about the model
        # to ensure loaded successfully:
        
        num_modules = len(list(self.mapper.model.modules()))
        self.assertEqual(num_modules, 68)


    #------------------------------------
    # test_load_image
    #-------------------
    
    #*******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_load_image(self):
        
        img, metadata = self.mapper.load_img(self.example_img_path)
        print('foo')
        
    


# ---------------------- Main -------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()