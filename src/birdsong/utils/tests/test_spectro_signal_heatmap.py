'''
Created on Jun 9, 2021

@author: paepcke
'''
import os
import unittest

from birdsong.utils.spectro_signal_heatmap import SpectrogramHeatmapper


TEST_ALL = True
#TEST_ALL = False


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir  = os.path.dirname(__file__)
        cls.tst_spectro_path = os.path.join(cls.cur_dir,
                                            'data/spectro_heatmapping/spectro_heatmap_tst.png')

    def setUp(self):
        self.heatmapper = SpectrogramHeatmapper(self.tst_spectro_path)

    def tearDown(self):
        pass

# ---------------- Tests --------------

    #------------------------------------
    # test_freq_analysis 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_freq_analysis(self):
        
        ##### analyze the frequencies
        

# --------------- Main --------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    