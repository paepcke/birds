'''
Created on Apr 29, 2021

@author: paepcke
'''
import os, sys
from pathlib import Path
import tempfile
import unittest

from data_augmentation.chop_spectrograms import SpectrogramChopper
from data_augmentation.list_png_metadata import PNGMetadataLister
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import WhenAlreadyDone, Utils


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


TEST_ALL = True
#TEST_ALL = False

class TestChopSpectrograms(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestChopSpectrograms, cls).setUpClass()
        
        cls.skip_size = 2 # sec
        
        cls.cur_dir  = os.path.dirname(__file__)
        cls.src_root = os.path.join(cls.cur_dir, 
                                        'spectro_data_long')
        cls.spectro_file = os.path.join(cls.src_root, 'DOVE/dove_long.png')
                                        
        _spectro, metadata = SoundProcessor.load_spectrogram(cls.spectro_file)
        try:
            cls.duration      = float(metadata['duration'])
        except KeyError:
            raise AssertionError(f"Spectrogram test file {os.path.basename(cls.spectro_file)} has no duration metadata")
        
        cls.default_win_len = 5 # seconds
        

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
                                         ) as dir_nm:
            chopper = SpectrogramChopper(
                self.src_root,
                dir_nm,
                overwrite_policy=WhenAlreadyDone.OVERWRITE
                )
            species = Path(self.spectro_file).parent.stem
            outdir  = os.path.join(dir_nm, species) 
            chopper.chop_one_spectro_file(self.spectro_file,
                                          outdir,
                                          skip_size=self.skip_size
                                          )
            snippet_names = os.listdir(outdir)
            num_expected_snippets = 1 + int(self.duration // self.skip_size)
            
            self.assertEqual(len(snippet_names), num_expected_snippets)
            
            # Check embedded metadata of one snippet:
            
            _spectro, metadata = SoundProcessor.load_spectrogram(Utils.listdir_abs(outdir)[0])
            self.assertEqual(int(metadata['duration']), self.default_win_len)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()