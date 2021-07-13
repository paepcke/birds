'''
Created on Apr 25, 2021

@author: paepcke
'''
import os
import unittest

import pandas as pd

from data_augmentation.utils import AugmentationGoals
from data_augmentation.utils import Utils, Interval

TEST_ALL = True
#TEST_ALL = False

class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.data_dir = os.path.join(cls.cur_dir, 'data')
        cls.spectros_dir = os.path.join(cls.cur_dir, 'spectro_data')
        # A small Raven selection table with its
        # row ordering intentionally scrambled: 
        cls.raven_sel_tbl_path = os.path.join(cls.data_dir, 'raven_sel_tbl_unsorted.csv')

    def setUp(self):
        pass


    def tearDown(self):
        pass

# ----------------------- Tests ---------------


    #------------------------------------
    # test_orig_file_name 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_orig_file_name(self):
        
        # Identity:
        aug_nm = "foo.wav"
        orig = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, aug_nm)
        
        aug_nm = "Amaziliadecora1061880-volume-10.wav"
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, 'Amaziliadecora1061880.wav')
        
        aug_nm = 'Amaziliadecora1061883-rain_bgd0ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, 'Amaziliadecora1061883.wav')
        
        aug_nm = 'Amaziliadecora1061886-shift4600ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, 'Amaziliadecora1061886.wav')
        
        # With directory relative:
        aug_nm = 'foo/bar/Amaziliadecora1061886-shift4600ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, 'foo/bar/Amaziliadecora1061886.wav')
        
        # With directory absolute:
        aug_nm = '/foo/bar/Amaziliadecora1061886-shift4600ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, '/foo/bar/Amaziliadecora1061886.wav')


    #------------------------------------
    # test_listdir_abs 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_listdir_abs(self):
        
        # Get the built-in directory listing
        # with just the file names:
        nearly_truth = os.listdir(self.cur_dir)
        
        abs_paths = Utils.listdir_abs(self.cur_dir)
        self.assertEquals(len(nearly_truth), len(abs_paths))
        
        # Check existence of first file or dir:
        self.assertTrue(os.path.exists(abs_paths[0]))

    #------------------------------------
    # test_find_in_tree_gen
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_find_in_tree_gen(self):
        
        res = list(Utils.find_in_tree_gen(
            self.spectros_dir, 
            pattern='*.png'
            ))
        expected = [f"{self.spectros_dir}/AMADEC/Amaziliadecora1061880.png",
                    f"{self.spectros_dir}/FORANA/SONG_XC609364-41759.png",
                    f"{self.spectros_dir}/FORANA/SONG_XC253440-FORANA04.png",
                    f"{self.spectros_dir}/FORANA/SONG_XC520628-passarochao.png",
                    f"{self.spectros_dir}/FORANA/SONG_XC360575-BFAN.png",
                    f"{self.spectros_dir}/FORANA/SONG_XC171241-Formicarius_analis.png"
                    ]
        self.assertSetEqual(set(res), set(expected))

    #------------------------------------
    # test_sample_compositions_by_species 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_sample_compositions_by_species(self):
        
        dist_df = Utils.sample_compositions_by_species(self.spectros_dir)
        #truth =  pd.DataFrame.from_dict({'AMADEC' : 1, 'FORANA' : 5}, orient='index', columns=['num_samples'])
        self.assertListEqual(list(dist_df.columns), ['num_samples'])
        self.assertEqual(int(dist_df.loc['AMADEC']), 1)
        self.assertEqual(int(dist_df.loc['FORANA']), 5)
        
    #------------------------------------
    # test_compute_num_augs_per_species 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_compute_num_augs_per_species(self):
        
        aug_volumes = AugmentationGoals.MAX
        sample_distrib_df = Utils.sample_compositions_by_species(self.spectros_dir)
        augs_to_do = Utils.compute_num_augs_per_species(aug_volumes, sample_distrib_df)
        self.assertEqual(augs_to_do['AMADEC'], 4)
        self.assertEqual(augs_to_do['FORANA'], 0)

    #------------------------------------
    # test_binary_in_interval_search 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_binary_in_interval_search(self):
        
        intervals = [Interval(1,3), Interval(4,5), Interval(6,7)]
    
        res = Utils.binary_in_interval_search(intervals, 0, 'low_val', 'high_val')
        assert(res == -1)
    
        res = Utils.binary_in_interval_search(intervals, 1, 'low_val', 'high_val')
        assert(res == 0)
    
        res = Utils.binary_in_interval_search(intervals, 2, 'low_val', 'high_val')
        assert(res == 0)
    
        res = Utils.binary_in_interval_search(intervals, 3, 'low_val', 'high_val')
        assert(res == -1)
    
        res = Utils.binary_in_interval_search(intervals, 4, 'low_val', 'high_val')
        assert(res == 1)
    
        res = Utils.binary_in_interval_search(intervals, 5, 'low_val', 'high_val')
        assert(res == -1)
    
        res = Utils.binary_in_interval_search(intervals, 8, 'low_val', 'high_val')
        assert(res == -1)
        

    #------------------------------------
    # test_read_raven_selection_table 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_read_raven_selection_table(self):
        
        dict_list = Utils.read_raven_selection_table(self.raven_sel_tbl_path)
        
        desired = \
                [{'Selection': '1',
                'View': 'Spectrogram 1',
                'Channel': '1',
                'Begin Time (s)': 0.0,
                'End Time (s)': 6.23740263,
                'Low Freq (Hz)': 2088.175,
                'High Freq (Hz)': 8538.314,
                'species': 'VASE',
                'type': 'song',
                'number': '1',
                'mix': ['RBPS','BGTA','RCWP'],
                'time_interval': {'low_val': 0.0,'high_val': 6.23740263},
                'freq_interval': {'low_val': 2088.175,'high_val': 8538.314}
                },
            
                {'Selection': '18',
                'View': 'Spectrogram 1',
                'Channel': '1',
                'Begin Time (s)': 1.024500915,
                'End Time (s)': 3.294473531,
                'Low Freq (Hz)': 2161.467,
                'High Freq (Hz)': 4492.937,
                'species': 'HOWP',
                'type': 'call-1',
                'number': '1',
                'mix': ['RBPS','VASE'],
                'time_interval': {'low_val': 1.024500915,'high_val': 3.294473531},
                'freq_interval': {'low_val': 2161.467,'high_val': 4492.937}
                },
            
                {'Selection': '2',
                'View': 'Spectrogram 1',
                'Channel': '1',
                'Begin Time (s)': 2.390074726,
                'End Time (s)': 3.231216904,
                'Low Freq (Hz)': 1564.1,
                'High Freq (Hz)': 3519.1,
                'species': 'RBPS',
                'type': 'song',
                'number': '1',
                'mix': [],
                'time_interval': {'low_val': 2.390074726,'high_val': 3.231216904},
                'freq_interval': {'low_val': 1564.1,'high_val': 3519.1}
                },
            
                {'Selection': '19',
                'View': 'Spectrogram 1',
                'Channel': '1',
                'Begin Time (s)': 5.926034705,
                'End Time (s)': 7.964992409,
                'Low Freq (Hz)': 3944.33,
                'High Freq (Hz)': 9791.219,
                'species': 'BGTA',
                'type': 'call',
                'number': '1',
                'mix': ['HOWP'],
                'time_interval': {'low_val': 5.926034705, 'high_val': 7.964992409},
                'freq_interval': {'low_val': 3944.33,'high_val': 9791.219}
                }]
        self.assertEqual(dict_list, desired)

    #------------------------------------
    # test_pad_series 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_pad_series(self):
        s = pd.Series([1,2,3])
        
        # Series already (more than) the proper length:
        new_s = Utils.pad_series(s, 'left', 1)
        self.assertTrue((s == new_s).all())
        
        # Series already (exactly) the proper length:
        new_s = Utils.pad_series(s, 'left', 3)
        self.assertTrue((s == new_s).all())        
        
        # Pad one on the left:
        new_s = Utils.pad_series(s, 'left', 4)
        self.assertTrue((new_s == pd.Series([1,1,2,3])).all())
        
        # Pad one on the right:
        new_s = Utils.pad_series(s, 'right', 4)
        self.assertTrue((new_s == pd.Series([1,2,3,3])).all())
        


# ---------------- Main --------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()