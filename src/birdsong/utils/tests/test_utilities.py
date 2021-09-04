'''
Created on Mar 11, 2021

@author: paepcke
'''
import os
from pathlib import Path
import tempfile
import unittest

from birdsong.utils.neural_net_config import NeuralNetConfig
from birdsong.utils.utilities import FileUtils


#*********TEST_ALL = True
TEST_ALL = False

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
        
        props = {'net_name': 'resnet18',
                 'min_epochs': '3',
                 'max_epochs': '6',
                 'batch_size': '2',
                 'num_folds': '3',
                 'seed': '42',
                 'kernel_size': '7',
                 'sample_width': '400',
                 'sample_height': '400',
                 'lr': '0.01',
                 'to_grayscale': 'False'
        }

        fname = FileUtils.construct_filename(props, 
                                             prefix='model', 
                                             suffix='.pth', 
                                             incl_date=False
                                             )
        expected = 'model_net_resnet18_bs_2_folds_3_ks_7_lr_0.01_gray_False.pth'

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
    # test_find_files_by_type
    #-------------------

    #*******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_find_files_by_type(self):
        
        expected_dict = {}
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='find_files_test_') as tmp_dir_nm:
            # Create test files and dirs:
            
            tmp_root_path = Path(tmp_dir_nm)
            
            # Top level singleton file:
            src = tmp_root_path.joinpath('tmp_top1.wav')
            # Create the file
            src.touch()
            
            found_files = FileUtils.find_files_by_type(src, FileUtils.AUDIO_EXTENSIONS)
            expected_dict[None] = [str(src)]
            self.assertDictEqual(found_files, expected_dict)

            # Add second top level file:
            src1 = tmp_root_path.joinpath('tmp_top2.wav')
            # Create the file
            src1.touch()
            
            found_files = FileUtils.find_files_by_type([src, src1],
                                                        FileUtils.AUDIO_EXTENSIONS)
            expected_dict[None] = [str(src), str(src1)]
            self.assertDictEqual(found_files, expected_dict)
            
            # Add a subdir with a file:
            level1_dir = tmp_root_path.joinpath('Level1')
            level1_dir.mkdir()
            src2 = level1_dir.joinpath('level1_tmp1.MP3')
            src2.touch()
            
            found_files = FileUtils.find_files_by_type([src, level1_dir, src1],
                                                        FileUtils.AUDIO_EXTENSIONS)

            expected_dict[str(level1_dir)] = ['level1_tmp1.MP3']
            self.assertDictEqual(found_files, expected_dict)

            # Add sibling to level1 subdir with 2 files:
            level2_dir = tmp_root_path.joinpath('Level2')
            level2_dir.mkdir()
            src3 = level2_dir.joinpath('level2_tmp1.MP3')
            src4 = level2_dir.joinpath('level2_tmp2.WAV')
            src3.touch()
            src4.touch()
            
            found_files = FileUtils.find_files_by_type([src, level1_dir, src1, level2_dir],
                                                        FileUtils.AUDIO_EXTENSIONS)

            expected_dict[str(level2_dir)] = ['level2_tmp1.MP3', 'level2_tmp2.WAV']
            self.assertDictEqual(found_files, expected_dict)
    
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
        self.assertEqual(str(tally.phase), 'TRAINING')

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
