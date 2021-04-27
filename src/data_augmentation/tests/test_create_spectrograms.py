'''
Created on Apr 26, 2021

@author: paepcke
'''
import os
import tempfile
import unittest

from data_augmentation.create_spectrograms import Spectrogrammer
from data_augmentation.utils import Utils


TEST_ALL = True
#TEST_ALL = False


class TestSpectrogramCreator(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)

    def setUp(self):
        pass


    def tearDown(self):
        pass

# -------------------------- Tests -------------

    #------------------------------------
    # test_create_spectro 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_create_spectro_one_dir(self):

        # A directory that itself contains audio files:
        src_dir = os.path.join(self.cur_dir, 'sound_data/DYSMEN_S/')
        self.run_spectrogrammer(src_dir)

    #------------------------------------
    # test_create_spectro_multiple_dirs 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_create_spectro_one_dirs(self):
        # Set of directories that contain the sound files:
        src_dir = os.path.join(self.cur_dir, 'sound_data')
        self.run_spectrogrammer(src_dir)
    
# -------------------- Utilities ---------------

    def run_spectrogrammer(self, src_dir):
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            dirs_filled = Spectrogrammer.create_spectrograms(src_dir, 
                                                             dst_dir, 
                                                             num=2)

            # Check that each spectro is of
            # reasonable size:
            for species_dst_dir in dirs_filled:
                for spec_file in Utils.listdir_abs(species_dst_dir):
                    self.assertTrue(os.stat(spec_file).st_size > 5000)



# ------------------ Main -------------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()