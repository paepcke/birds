'''
Created on Apr 25, 2021

@author: paepcke
'''
import datetime
import os
import unittest

from birdsong.utils.species_name_converter import ConversionError
from data_augmentation.utils import AugmentationGoals
from data_augmentation.utils import Utils, Interval
import pandas as pd


#******TEST_ALL = True
TEST_ALL = False

class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.data_dir = os.path.join(cls.cur_dir, 'data')
        cls.spectros_dir = os.path.join(cls.cur_dir, 'spectro_data')
        # A small Raven selection table with its
        # row ordering intentionally scrambled: 
        cls.raven_sel_tbl_path = os.path.join(cls.data_dir, 
                                              'raven_sel_tbl_unsorted.csv')
        cls.raven_sel_tbl_missing_type_path = os.path.join(cls.data_dir, 
                                                           'raven_sel_tbl_missing_type_info.csv'
                                                           )
        cls.raven_sel_tbl_bad_col_header_spellings_path = os.path.join(cls.data_dir, 
                                                           'raven_sel_tbl_misspelled_col_names.csv'
                                                           )


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
        self.assertEqual(orig, aug_nm)
        
        aug_nm = "Amaziliadecora1061880-volume-10.wav"
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEqual(orig, 'Amaziliadecora1061880.wav')
        
        aug_nm = 'Amaziliadecora1061883-rain_bgd0ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEqual(orig, 'Amaziliadecora1061883.wav')
        
        aug_nm = 'Amaziliadecora1061886-shift4600ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEqual(orig, 'Amaziliadecora1061886.wav')
        
        # With directory relative:
        aug_nm = 'foo/bar/Amaziliadecora1061886-shift4600ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEqual(orig, 'foo/bar/Amaziliadecora1061886.wav')
        
        # With directory absolute:
        aug_nm = '/foo/bar/Amaziliadecora1061886-shift4600ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEqual(orig, '/foo/bar/Amaziliadecora1061886.wav')


    #------------------------------------
    # test_listdir_abs 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_listdir_abs(self):
        
        # Get the built-in directory listing
        # with just the file names:
        nearly_truth = os.listdir(self.cur_dir)
        
        abs_paths = Utils.listdir_abs(self.cur_dir)
        self.assertEqual(len(nearly_truth), len(abs_paths))
        
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
                'species': 'VASEG',
                'type': 'song',
                'number': '1',
                'mix': ['RBPSG','BGTAG','WTROS'],
                'time_interval': {'low_val': 0.0,'high_val': 6.23740263, 'step' : 1},
                'freq_interval': {'low_val': 2088.175,'high_val': 8538.314, 'step' : 1}
                },
            
                {'Selection': '18',
                'View': 'Spectrogram 1',
                'Channel': '1',
                'Begin Time (s)': 1.024500915,
                'End Time (s)': 3.294473531,
                'Low Freq (Hz)': 2161.467,
                'High Freq (Hz)': 4492.937,
                'species': 'HOWPG',
                'type': 'call-1',
                'number': '1',
                'mix': ['RBPSG','VASEG'],
                'time_interval': {'low_val': 1.024500915,'high_val': 3.294473531, 'step' : 1},
                'freq_interval': {'low_val': 2161.467,'high_val': 4492.937, 'step' : 1}
                },
            
                {'Selection': '2',
                'View': 'Spectrogram 1',
                'Channel': '1',
                'Begin Time (s)': 2.390074726,
                'End Time (s)': 3.231216904,
                'Low Freq (Hz)': 1564.1,
                'High Freq (Hz)': 3519.1,
                'species': 'LEGRG',
                'type': '',
                'number': '1',
                'mix': [],
                'time_interval': {'low_val': 2.390074726,'high_val': 3.231216904, 'step' : 1},
                'freq_interval': {'low_val': 1564.1,'high_val': 3519.1, 'step' : 1}
                },
            
                {'Selection': '19',
                'View': 'Spectrogram 1',
                'Channel': '1',
                'Begin Time (s)': 5.926034705,
                'End Time (s)': 7.964992409,
                'Low Freq (Hz)': 3944.33,
                'High Freq (Hz)': 9791.219,
                'species': 'BTSAC',
                'type': 'call',
                'number': '1',
                'mix': ['HOWPG'],
                'time_interval': {'low_val': 5.926034705, 'high_val': 7.964992409, 'step' : 1},
                'freq_interval': {'low_val': 3944.33,'high_val': 9791.219, 'step' : 1}
                }]
        self.assertEqual(dict_list, desired)
        
        # Case when song/call split is required, but
        # selection table's Type column is empty:
        
        with self.assertRaises(ConversionError): 
            Utils.read_raven_selection_table(self.raven_sel_tbl_missing_type_path)
            
        # Misspelled column headers:
        
        dict_list = Utils.read_raven_selection_table(self.raven_sel_tbl_bad_col_header_spellings_path)
        desired = [
            {'Selection': '3',
            'View': 'Spectrogram 1',
            'Channel': '1',
            'Begin Time (s)': 1.234,
            'End Time (s)': 5.64,
            'Low Freq (Hz)': 1564.1,
            'High Freq (Hz)': 3519.1,
            'species': 'OBNTC',
            'type': 'CALL',
            'number': '2',
            'mix': ['SHWCG'],
            'time_interval': {'low_val': 1.234,'high_val': 5.64, 'step' : 1},
            'freq_interval': {'low_val': 1564.1,'high_val': 3519.1, 'step' : 1}
            },
            {'Selection': '2',
            'View': 'Spectrogram 1',
            'Channel': '1',
            'Begin Time (s)': 2.390074726,
            'End Time (s)': 3.231216904,
            'Low Freq (Hz)': 1564.1,
            'High Freq (Hz)': 3519.1,
            'species': 'OBNTC',
            'type': 'CALL',
            'number': '2',
            'mix': ['GAGAG'],
            'time_interval': {'low_val': 2.390074726,'high_val': 3.231216904, 'step' : 1},
            'freq_interval': {'low_val': 1564.1,'high_val': 3519.1, 'step' : 1}
            }
            ]
                
        self.assertDictEqual(dict_list[0], desired[0])
        self.assertDictEqual(dict_list[1], desired[1])

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
        
    #------------------------------------
    # test_time_delta_str
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_time_delta_str(self):
        
        res = Utils.time_delta_str(10)
        self.assertEqual(res, '10.0 seconds')

        res = Utils.time_delta_str(100)
        self.assertEqual(res, '1.0 minute, 40.0 seconds')

        res = Utils.time_delta_str(datetime.timedelta(seconds=100))
        self.assertEqual(res, '1.0 minute, 40.0 seconds')
        
        res = Utils.time_delta_str(datetime.timedelta(seconds=3610))
        self.assertEqual(res, '1.0 hour, 10.0 seconds')


    #------------------------------------
    # test_assertDataframesEqual
    #-------------------
    
    #********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_assertDataframesEqual(self):
        
        df1 = pd.DataFrame([[1,2,3], [4,5,6]])
        df2 = pd.DataFrame([[1,2,3], [4,5,6]])
        
        Utils.assertDataframesEqual(df1,df2)
        
        df1.columns = ['foo', 'bar', 'fum']
        
        # Different column names:
        with self.assertRaises(AssertionError):
            Utils.assertDataframesEqual(df1,df2)

        # Restore column names
        df1.columns = [0,1,2]
        # DFs are equal again:
        Utils.assertDataframesEqual(df1,df2)
        
        # Change one number:
        df1.iloc[0,0] = 0
        with self.assertRaises(AssertionError):
            Utils.assertDataframesEqual(df1,df2)
        
        # Back to equal:
        df1.iloc[0,0] = 1
        Utils.assertDataframesEqual(df1,df2)

        # Different index (row lables:
        df1.index = ['blue', 'green']
        with self.assertRaises(AssertionError):
            Utils.assertDataframesEqual(df1,df2)
        
        # Back to equal:
        df1.index = [0,1]
        Utils.assertDataframesEqual(df1,df2)

    #------------------------------------
    # test_assertSeriesEqual
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_assertSeriesEqual(self):
        
        ser_base = pd.Series([1.1234, 2.6789], name='test')
        ser = pd.Series([1.1234, 2.6789], name='test')
        Utils.assertSeriesEqual(ser, ser_base)

        # Test diff in name only:
        ser = pd.Series([1.1234, 2.6789], name='foo')
        with self.assertRaises(AssertionError):
            Utils.assertSeriesEqual(ser_base, ser)
            
        # Test equal to given decimals
        ser = pd.Series([1.1234, 2.7289], name='test')
        # Should be OK if decimals specified to
        # one place, but fail is required equality
        # to 2nd place:
        Utils.assertSeriesEqual(ser, ser_base, decimals=1)
        with self.assertRaises(AssertionError):
            Utils.assertSeriesEqual(ser_base, ser, decimals=2)

        # Test with non-numeric index:
        
        ser_base_str_idx = pd.Series({'foo' : 1.1234, 'bar' : 2.6789}, name='test')
        ser = ser_base_str_idx
        Utils.assertSeriesEqual(ser, ser_base_str_idx)

    #------------------------------------
    # test_intervals
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_intervals(self):
        iv = Interval(0,5)
        self.assertEqual(iv['low_val'], 0)
        self.assertEqual(iv['high_val'], 5)
        self.assertEqual(iv['step'], 1)
        
        # Iteration:
        self.assertListEqual(list(iv.values()), [0,1,2,3,4])
        
        # Indexing as into list:
        self.assertEqual(iv[0], 0)
        self.assertEqual(iv[-1], 4)
        
        # Floats
        iv = Interval(0,2,0.1)
        self.assertEqual(iv[1], 0.1)
        self.assertAlmostEqual(iv[10], 1, delta=0.01)
        self.assertAlmostEqual(iv[19], 1.90, delta=0.01)
        with self.assertRaises(IndexError):
            self.assertEqual(iv[20])
        
        # Overlaps:
        
        iv1 = Interval(0, 10, 1)
        iv2 = Interval(-5, -4, 1)
        self.assertFalse(iv1.overlaps(iv2))
        iv2 = Interval(-5, 10, 1)
        self.assertTrue(iv1.overlaps(iv2))
        iv2 = Interval(10, 15, 1)
        self.assertFalse(iv1.overlaps(iv2))
        iv2 = Interval(4, 5, 1)
        self.assertTrue(iv1.overlaps(iv2))
        iv2 = Interval(-4, 15, 1)
        self.assertTrue(iv1.overlaps(iv2))
        
        # Contains (a given number)
        iv1 = Interval(0,10,1)
        self.assertTrue(iv1.contains(0))
        self.assertTrue(iv1.contains(9))
        self.assertFalse(iv1.contains(10))
        self.assertFalse(iv1.contains(-1))
        
        # Overlap percentage
        
        # other is same as self:
        iv2 = Interval(0,10,1)
        self.assertEqual(iv1.percent_overlap(iv2), 100)
        # other abuts to the right
        iv2 = Interval(10,20,1)
        self.assertEqual(iv1.percent_overlap(iv2), 0)
        # other abuts to the left:
        iv2 = Interval(-5,0,1)
        self.assertEqual(iv1.percent_overlap(iv2), 0)
        # other starts within by 1, and reaches behond:
        iv2 = Interval(9,20,1)
        self.assertEqual(iv1.percent_overlap(iv2), 10)
        # other starts below self, and reaches to middle
        # of self
        iv2 = Interval(-1,5,1)
        self.assertEqual(iv1.percent_overlap(iv2), 50)
        # other fully containse self:
        iv2 = Interval(-10, 10, 1)
        self.assertEqual(iv1.percent_overlap(iv2), 50)
    
        # Classmethod binary_search_contains:
        intervals = [Interval(10, 20),Interval(21, 30), Interval(31, 40)]
        # Test values below or above all intervals:
        int_idx = Interval.binary_search_contains(intervals, 9)
        self.assertEqual(int_idx, -1)
        int_idx = Interval.binary_search_contains(intervals, 40)
        self.assertEqual(int_idx, -1)
        
        # Value is in first interval
        int_idx = Interval.binary_search_contains(intervals, 10)
        self.assertEqual(int_idx, 0)
        int_idx = Interval.binary_search_contains(intervals, 19)
        self.assertEqual(int_idx, 0)
        
        # Value is in last interval
        int_idx = Interval.binary_search_contains(intervals, 31)
        self.assertEqual(int_idx, 2)
        int_idx = Interval.binary_search_contains(intervals, 39)
        self.assertEqual(int_idx, 2)
        
        # Value is in middle interval
        int_idx = Interval.binary_search_contains(intervals, 21)
        self.assertEqual(int_idx, 1)
        int_idx = Interval.binary_search_contains(intervals, 29)
        self.assertEqual(int_idx, 1)
        
        # Classmethod binary_search_overlaps
        
        tst_interval = Interval(0,9)
        self.assertEqual(Interval.binary_search_overlap(intervals, tst_interval), -1)
        tst_interval = Interval(40,50)
        self.assertEqual(Interval.binary_search_overlap(intervals, tst_interval), -1)
        
        # Overlap with first interval in list:
        tst_interval = Interval(0,11)
        self.assertEqual(Interval.binary_search_overlap(intervals, tst_interval), 0)
        
        # Overlap with middle interval in list:
        tst_interval = Interval(29,60)
        self.assertEqual(Interval.binary_search_overlap(intervals, tst_interval), 1)
         
        # Overlap with last interval in list:
        tst_interval = Interval(30,39)
        self.assertEqual(Interval.binary_search_overlap(intervals, tst_interval), 2)

    #------------------------------------
    # test_df_extract_rect
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_df_extract_rect(self):
        
        # Make a df:
        # 
        #           9.0   10.1  12.5  20.5
        #     4000     1     2     3     4
        #     3000     5     6     7     8
        #     2000     9    10    11    12
        #     1000    13    14    15    16

        spectro = pd.DataFrame([[1,2,3,4],[5,6,7,8],[9,10,11,12], [13,14,15,16]], 
                               index=[4000,3000,2000,1000], 
                               columns=[9.0,10.1,12.5,20.5])
        
        # Given values exist in index and cols:
        rect = Utils.df_extract_rect(spectro, 
                                     yx=(3000,10.1), 
                                     height=1000, 
                                     width=3)
        expected = pd.DataFrame([[2,3], [6,7]],
                                index=[4000,3000],
                                columns=[10.1, 12.5]
                                )
        Utils.assertDataframesEqual(rect, expected)

        # Given values doe not exist in index/cols:
        rect = Utils.df_extract_rect(spectro, 
                                     yx=(2500,10), 
                                     height=1000, 
                                     width=3)
        expected = pd.DataFrame([[6, 7]], index=[3000], columns=[10.1, 12.5])
        Utils.assertDataframesEqual(rect, expected)
        
        # Beyond range of index/cols:
        rect = Utils.df_extract_rect(spectro, 
                                     yx=(4500,40), 
                                     height=1000, 
                                     width=3)
        
        expected = pd.DataFrame([])
        Utils.assertDataframesEqual(rect, expected)
        
        # Right on the border:
        rect = Utils.df_extract_rect(spectro, 
                                     yx=(1000,20.5), 
                                     height=1000, 
                                     width=3)
        
        expected = pd.DataFrame([[12], [16]], index=[2000,1000], columns=[20.5])
        Utils.assertDataframesEqual(rect, expected)

    #------------------------------------
    # test_df_eq
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_df_eq(self):
        
        # Index numeric:
        df1 = pd.DataFrame([[1,2,3],[4,5,6]], index=[0.1234, 0.2345])
        df2 = df1.copy()
        self.assertTrue(Utils.df_eq(df1, df2))
        
        # Index numeric, but diff in last dig, yet decimal saves it:
        df2.index = pd.Index([0.1234, 0.2347])
        self.assertFalse(Utils.df_eq(df1, df2))
        self.assertTrue(Utils.df_eq(df1, df2, decimals=2))

        # Cols numeric:
        df1 = pd.DataFrame([[1,2,3],[4,5,6]], columns=[0.1234, 0.2345, 0.7890])
        df2 = df1.copy()
        df2.columns = pd.Index([0.1234, 0.2347, 0.7890])
        self.assertFalse(Utils.df_eq(df1, df2))
        self.assertTrue(Utils.df_eq(df1, df2, decimals=2))
        
        # Number part, index, and col different, but saved by decimal:
        df1 = pd.DataFrame([[1.1234,2.1234,3.1234],[4,5,6]], 
                           index=[0.1234, 0.2345],
                           columns=[0.1234, 0.2345, 0.7890])
        df2 = df1.copy()
        self.assertTrue(Utils.df_eq(df1, df2))
        df2.iloc[0,1] = 2.1236
        df2.index = pd.Index([0.1234, 0.2347])
        df2.columns = pd.Index([0.1234, 0.2345, 0.7899])
        self.assertFalse(Utils.df_eq(df1, df2))
        self.assertTrue(Utils.df_eq(df1, df2, decimals=2))
        
        #*******TEST Non-numeric index/columns
        #*******TEST series_eq
        # Adjust assertDfEqual/assertSerEqual. 


# ---------------- Main --------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()