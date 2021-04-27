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
    def test_create_spectro(self):
        src_dir = os.path.join(self.cur_dir, 'sound_data/DYSMEN_S/')
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            Spectrogrammer.create_spectrogram(src_dir, dst_dir, num=2)

            # Check that each spectro is of
            # reasonable size:
            for spec_file in Utils.listdir_abs(dst_dir):
                self.assertTrue(os.stat(spec_file).st_size > 5000)
         


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()