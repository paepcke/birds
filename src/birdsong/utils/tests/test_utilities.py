'''
Created on Mar 11, 2021

@author: paepcke
'''
import os
import unittest

from birdsong.utils.utilities import FileUtils


TEST_ALL = True
#TEST_ALL = False

class TestUtilities(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fname = 'run_2021-03-11T10_59_02_net_resnet18_pretrain_0_lr_0.01_opt_SGD_bs_64_ks_7_folds_0_classes_10.csv'
        cls.csv_data_path = os.path.join(os.path.dirname(__file__),
                                         f"data/sample_csv_file/{cls.fname}"
                                     )
        
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass


# --------------- Tests ---------------

    #------------------------------------
    # test_parse_filename
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_parse_filename(self):
        
        prop_dict = FileUtils.parse_filename(self.csv_data_path)
        
        self.assertEqual(prop_dict['timestamp'], '2021-03-11T10_59_02')
        self.assertEqual(prop_dict['net_name'], 'resnet18')
        self.assertEqual(prop_dict['pretrain'], 0)
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