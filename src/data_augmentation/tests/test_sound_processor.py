'''
Created on Apr 29, 2021

@author: paepcke
'''
import os
import tempfile
import unittest

from data_augmentation.sound_processor import SoundProcessor


TEST_ALL = True
#TEST_ALL = False

# NOTE: SoundProcessor is also exercised in 
#       other unittests, such as create_spectrograms()

class TestSoundProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # -------------------- Tests ---------------

    #------------------------------------
    # test_create_one_spectro 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_create_one_spectro(self):
        
        # Test creating one spectrogram from an audio
        # file. Ensure that sample rate, duration, and 
        # species are included int the destination spectrogram
        # .png file:
        audio_path = os.path.join(self.cur_dir, 'audio_aug_tst_data/DYSMEN_S/dys1.mp3')
        (aud, sr) = SoundProcessor.load_audio(audio_path)
        with tempfile.NamedTemporaryFile(suffix='.png', prefix='spectro', dir='/tmp', delete=True) as fd:
            SoundProcessor.create_spectrogram(aud, 
                                              sr,
                                              fd.name, 
                                              info={'species' : 'DYSMEN_C'})
            _spectro, info = SoundProcessor.load_spectrogram(fd.name)
            truth = {'sr': '22050', 'duration': '10.8', 'species': 'DYSMEN_C'}
            self.assertDictEqual(info, truth)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()