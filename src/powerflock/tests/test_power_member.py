'''
Created on Nov 9, 2021

@author: paepcke
'''
import os
import unittest

from powerflock.power_member import PowerMember

#*******TEST_ALL = True
TEST_ALL = False

class TestPowerMember(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        
        cls.rec_XC = os.path.join(cls.cur_dir,
                                  'signal_processing_sounds/XenoCanto/CMTOG/SONG_XC274155-429_Chestnut-mandibled_Toucan_2_song.mp3')
        cls.rec_fld = os.path.join(cls.cur_dir,
                                  'signal_processing_sounds/Field/CMTOG/DS_AM03_20190713_055956.wav')

        
        cls.sel_tbl_XC = os.path.join(cls.cur_dir, 
                                      'selection_tables/XenoCanto/sel_tbl_Kelley_SONG_XC274155-429_CMTO_KEL.txt')
        cls.sel_tbl_fld = os.path.join(cls.cur_dir, 
                                       'selection_tables/JZ_DS_AM03_20190713_055956.Table.1.selections.txt')
        
    def setUp(self):
        pass


    def tearDown(self):
        pass

# ----------------- Tests -------------

    #------------------------------------
    # test_calibrate_probabilities
    #-------------------
    
    # @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    # def test_calibrate_probabilities(self):
    #
    #     pm = PowerMember(
    #         species_name='CMTOG', 
    #         spectral_template_info=zip([self.rec_XC], [self.sel_tbl_XC]),
    #         apply_bandpass=True
    #         )
    #
    #     pm.calibrate_probabilities(self.rec_XC, self.sel_tbl_XC, apply_bandpass=True, plot_pr=True)
    #
    #     print('foo')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()