'''
Created on Aug 27, 2021

@author: paepcke
'''
import os
from pathlib import Path
import statistics
import unittest

from result_analysis.utils.frequency_bands_from_selection_tables import FrequencyBandExtractor


TEST_ALL = True
#TEST_ALL = False


class TestFrequencyBandsFromSelectionTables(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        
        cls.sel_tbl_path_lots = os.path.join(cls.cur_dir, "selection_tables/tst_sel_tbl_lots.txt")
        cls.sel_tbl_path_one = os.path.join(cls.cur_dir, "selection_tables/tst_sel_tbl_one.txt")

        cls.all_tbls_root_dir = str(Path(cls.sel_tbl_path_lots).parent)
        os.makedirs(cls.all_tbls_root_dir, exist_ok=True)
         
        cls.content_lots, cls.content_one = cls.read_tst_sel_tbls(cls.sel_tbl_path_lots, 
                                                               cls.sel_tbl_path_one)

    def setUp(self):
        pass

    def tearDown(self):
        pass

# --------------------- Tests --------------

    #------------------------------------
    # test_one_table_import 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_one_table_import(self):
        
        extractor = FrequencyBandExtractor(self.sel_tbl_path_one)
        # List of all rows in the (only) selection table:
        dict_list = extractor.sel_dict_list
        
        expected_dict  = {
            'Selection': '12', 
            'View': 
            'Spectrogram 1', 
            'Channel': '1', 
            'Begin Time (s)': 28.728433252, 
            'End Time (s)': 29.279807974, 
            'Low Freq (Hz)': 170.49, 
            'High Freq (Hz)': 767.204, 
            'type': 'SONG', 
            'mix': [], 
            'number': '1', 
            'species': 'WTDOG', 
            'time_interval': {'low_val': 28.728433252, 'high_val': 29.279807974}, 
            'freq_interval': {'low_val': 170.49, 'high_val': 767.204}
            }

        self.assertEqual(len(dict_list), 1)
        self.assertDictEqual(dict_list[0], expected_dict)
        
    #------------------------------------
    # test_two_table_import
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_two_table_import(self):
        
        # The parent dir of sel_tbl_path_one  and sel_tbl_path_lots
        # holds both of those tables. So: list dicts comprising 
        # all rows of the combined two test selection tables:

        extractor = FrequencyBandExtractor(os.path.dirname(self.sel_tbl_path_one))
        dict_list = extractor.sel_dict_list
        
        # Number of dicts should be the content
        # of both tables, minus any empty lines
        # and minus the two headers:
        
        num_non_empty_lines = sum([1 
                                   for dummy in self.content_one + self.content_lots
                                   if not dummy.isspace()
                                   ])

        self.assertEqual(len(dict_list), num_non_empty_lines - 2)

    #------------------------------------
    # test_min_max_freqs 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_min_max_freqs(self):
        extractor = FrequencyBandExtractor(self.sel_tbl_path_one)

        WTDOG_freq_info = extractor['WTDOG']
        
        self.assertEqual(WTDOG_freq_info.min_frequency(), 170.490)
        self.assertEqual(WTDOG_freq_info.max_frequency(), 767.204)

        extractor = FrequencyBandExtractor(self.sel_tbl_path_lots)

        WTDOG_freq_info = extractor['WTDOG']
        self.assertEqual(WTDOG_freq_info.min_frequency(), 160.7)
        self.assertEqual(WTDOG_freq_info.max_frequency(), 807.1)
        
        BLPAG_freq_info = extractor['BLPAG']
        self.assertEqual(BLPAG_freq_info.min_frequency(), 1865.436)
        self.assertEqual(BLPAG_freq_info.max_frequency(), 4900.250)

    #------------------------------------
    # test_center_freq
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_center_freq(self):
        extractor = FrequencyBandExtractor(self.sel_tbl_path_one)

        # Center frequency of single occurrence
        # of a species:

        finfo = extractor['WTDOG']
        freq_low  = finfo.min_frequency()
        freq_high = finfo.max_frequency()
        
        center = (freq_high - freq_low) / 2
        self.assertEqual(finfo.center_frequency(), center)
        
        extractor = FrequencyBandExtractor(self.all_tbls_root_dir)
        finfo = extractor['WTDOG']
        # Compute center freques of the three 
        # occurring WTDOG selections (1 in the single-row
        # tbl, and 2 in the other tbl):
        
        freq_centers = []
        freq_centers.append((767.204 - 170.490)/2.)
        freq_centers.append((807.100 - 160.700)/2.)
        freq_centers.append((610.700 - 225.000)/2.)
        
        mean_center_freq = sum(freq_centers) / 3
        
        self.assertEqual(finfo.center_frequency(), mean_center_freq)
        

    #------------------------------------
    # test_stdev_freq_bands
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_stdev_freq_bands(self):
        
        extractor = FrequencyBandExtractor(self.all_tbls_root_dir)
        finfo = extractor['BLPAG']
        freq_centers = []
        freq_centers.append((4900.250 - 2171.702) / 2.0)
        freq_centers.append((4900.250 - 2378.600) / 2.0)
        freq_centers.append((4844.565 - 1976.805) / 2.0)
        freq_centers.append((4788.881 - 1865.436) / 2.0)

        # Get the population standard deviation
        # as opposed to stdev(), which would be the
        # sample popoulation:
        expected_stdev = statistics.pstdev(freq_centers)
        computed_stdev = finfo.stdev_freq_bands()
        
        self.assertEqual(computed_stdev, expected_stdev)


# --------------------- Utilities ----------------

    #------------------------------------
    # read_tst_sel_tbls 
    #-------------------
    
    @classmethod
    def read_tst_sel_tbls(cls, sel_tbl_path_lots, sel_tbl_path_one):
        '''
        Read two example selection tables
        that contain various pitfalls. The table stored
        in sel_tbl_path_one will have a single line. 
        The other will have several.
        
        Return a tuple with the content of each. Each 
        content is a multi-line string. 

        :param sel_tbl_path_lots: where to put sel tbl that
            has more than one entry
        :type sel_tbl_path_lots: src
        :param sel_tbl_path_one: where to put sel tbl that
            only has one entry
        :type sel_tbl_path_one: str
        :return 2-tuple, each element being the content of
            one table
        :rtype [str, str]
        '''
        with open(sel_tbl_path_lots, 'r') as fd:
            tbl_content_lots = fd.readlines()

        with open(sel_tbl_path_one, 'r') as fd:
            tbl_content_one = fd.readlines()

        return (tbl_content_lots, tbl_content_one)
    
# --------------- Main --------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
