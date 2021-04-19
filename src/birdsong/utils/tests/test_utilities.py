'''
Created on Mar 11, 2021

@author: paepcke
'''
import os
import unittest

from birdsong.utils.utilities import FileUtils
from birdsong.utils.neural_net_config import NeuralNetConfig


TEST_ALL = True
#TEST_ALL = False

class TestUtilities(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.fname = 'run_2021-03-11T10_59_02_net_resnet18_pre_True_lr_0.01_opt_SGD_bs_64_ks_7_folds_0_classes_10.csv'
        cls.csv_data_path = os.path.join(os.path.dirname(__file__),
                                         f"data/sample_csv_file/{cls.fname}"
                                     )
        cls.config_path = os.path.join(cls.curr_dir,
                                       'utils_test_config.cfg'
                                       )
        
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass


# --------------- Tests ---------------

    #------------------------------------
    # test_construct_filename 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_construct_filename(self):
        
        conf = NeuralNetConfig(self.config_path)
        training_section = conf.Training
        prop_dict = FileUtils.make_run_props_dict(training_section)
        
        fname = FileUtils.construct_filename(prop_dict, 
                                             prefix='model', 
                                             suffix='.pth', 
                                             incl_date=False
                                             )
        expected = 'model_net_resnet18_pre_na_frz_na_lr_0.01_opt_na_bs_2_ks_7_folds_3_gray_False_classes_na.pth' 

        self.assertEqual(fname, expected)

    #------------------------------------
    # test_make_run_props_dict 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_make_run_props_dict(self):
        
        conf = NeuralNetConfig(self.config_path)
        training_section = conf.Training
        
        # Create the expected result for ground truth:
        expected_dict = {}
        for short_name, long_name in FileUtils.fname_short_2_long.items():
            try:
                val = training_section[long_name]
                expected_dict[short_name] = val
            except KeyError:
                # Config file happens not to 
                # have an entry for the long_name:
                expected_dict[short_name] = 'na'
                continue
        
        prop_dict = FileUtils.make_run_props_dict(training_section)
        self.assertDictEqual(prop_dict, expected_dict)

    #------------------------------------
    # test_parse_filename
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_parse_filename(self):
        
        prop_dict = FileUtils.parse_filename(self.csv_data_path)
        
        self.assertEqual(prop_dict['timestamp'], '2021-03-11T10_59_02')
        self.assertEqual(prop_dict['net_name'], 'resnet18')
        self.assertEqual(prop_dict['pretrained'], True)
        self.assertEqual(prop_dict['lr'], 0.01)
        self.assertEqual(prop_dict['opt_name'], 'SGD')
        self.assertEqual(prop_dict['batch_size'], 64)
        self.assertEqual(prop_dict['kernel_size'], 7)
        self.assertEqual(prop_dict['num_folds'], 0)
        self.assertEqual(prop_dict['num_classes'], 10)

    #------------------------------------
    # test_load_preds_and_labels 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_load_preds_and_labels(self):
        
        tally_coll = FileUtils.load_preds_and_labels(self.csv_data_path)
        
        # Expect four tallies from the two
        # rows in the csv file: each row has
        # a train and a val:
        
        self.assertEqual(len(tally_coll), 4)
        tally = tally_coll[0]
        self.assertEqual(tally.batch_size, 64)
        self.assertEqual(str(tally.phase), 'LearningPhase.TRAINING')

    #------------------------------------
    # test_ellipsed_file_path 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_ellipsed_file_path(self):
        
        # File name too long even without
        # leading dirs:
        self.assertEqual(FileUtils.ellipsed_file_path('/home/yatagait/birds/src/birdsong/recordings/CALL_XC482431-R024 white ruffed manakin.mp3'),
                         '/home/...CALL_XC482431-R024 white ruffed manakin.mp3'
                         )
        # Same without leading slash
        self.assertEqual(FileUtils.ellipsed_file_path('home/yatagait/birds/src/birdsong/recordings/CALL_XC482431-R024 white ruffed manakin.mp3'),
                         'home/...CALL_XC482431-R024 white ruffed manakin.mp3'
                         )
        self.assertEqual(FileUtils.ellipsed_file_path('foobar'),
                         'foobar'
                         )
        self.assertEqual(FileUtils.ellipsed_file_path('foobar/fum'),
                         'foobar/fum'
                         )
        # Uneven num of dirs:
        self.assertEqual(FileUtils.ellipsed_file_path('foobar/bluebell/grayhound/'),
                         'foobar/.../grayhound'
                         )
        
        # Even num of dirs
        self.assertEqual(FileUtils.ellipsed_file_path('blue/foobar/bluebell/grayhound/'),
                         'blue/.../bluebell/grayhound'
                         )
        # Length just acceptable
        self.assertEqual(FileUtils.ellipsed_file_path('blue/foobar/grayhound/bar'),
                         'blue/foobar/grayhound/bar'
                         )
        # Length one over acceptable
        self.assertEqual(FileUtils.ellipsed_file_path('Bblue/foobar/grayhound/bar'),
                         'Bblue/.../grayhound/bar'
                         )
        # Absolute path:
        self.assertEqual(FileUtils.ellipsed_file_path('/Bblue/foobar/grayhound/bar'),
                         '/Bblue/.../grayhound/bar'
                         )



# ---------------- Main ---------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
