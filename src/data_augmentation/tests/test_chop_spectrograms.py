'''
Created on Apr 29, 2021

@author: paepcke
'''
import os, sys
import tempfile
import unittest

from data_augmentation.chop_spectrograms import SpectrogramChopper
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import WhenAlreadyDone


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


TEST_ALL = True
#TEST_ALL = False

class TestChopSpectrograms(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestChopSpectrograms, cls).setUpClass()
        cls.cur_dir = os.path.dirname(__file__)
        cls.spectro_file = os.path.join(cls.cur_dir, 
                                        'spectro_data/AMADEC/Amaziliadecora1061880.png'
                                        )
        

    def setUp(self):
        pass


    def tearDown(self):
        pass

# -------------- Test Methods ---------------

    #------------------------------------
    # test_chop_one_spectrogram_file
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_chop_one_spectrogram_file(self):
        
        with tempfile.TemporaryDirectory(dir='/tmp',
                                         prefix='chopping', 
                                         suffix='.png') as dir_nm:
            chopper = SpectrogramChopper(
                os.path.dirname(self.spectro_file),
                dir_nm,
                overwrite_policy=WhenAlreadyDone.OVERWRITE
                )
            chopper.chop_one_spectro_file(self.spectro_file,
                                          dir_nm
                                          )
            snippet_names = os.listdir(dir_nm)
            print(snippet_names)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()