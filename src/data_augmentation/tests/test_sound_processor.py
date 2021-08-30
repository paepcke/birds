'''
Created on Apr 29, 2021

@author: paepcke
'''
import datetime
import os
import tempfile
import unittest

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Interval


TEST_ALL = True
#TEST_ALL = False

# NOTE: SoundProcessor is also exercised in 
#       other unittests, such as create_spectrogram()

class TestSoundProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.snd_file_data = os.path.join(cls.cur_dir, "data/mp3_and_wav_data/")
        
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

    #------------------------------------
    # test_energy_highlights 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_energy_highlights(self):
        audio_path = os.path.join(self.cur_dir, 'audio_aug_tst_data/DYSMEN_S/dys1.mp3')
        (aud, sr) = SoundProcessor.load_audio(audio_path)
        high_intensity_intervals = SoundProcessor.energy_highlights(aud, sr)
        
        # Should have 2 freq 'band bundles" of high intensity
        # area:
        self.assertEqual(len(high_intensity_intervals), 2)
        
        # And the frequency bounds:
        truth = [Interval(0.0, 22.533203125), Interval(1679.58984375, 1971.2880859375)]
        self.assertEqual(high_intensity_intervals, truth)
        
    #------------------------------------
    # test_soundfile_metadata
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_soundfile_metadata(self):
        
        # mp3 from the Web:
        bells_mp3 = os.path.join(self.snd_file_data, 'church_bells.mp3')
        sound_md = SoundProcessor.soundfile_metadata(bells_mp3)
        expected = {'duration' : datetime.timedelta(seconds=54.19),
                    'sample_rate' : 48000
                    }
        self.assertDictEqual(sound_md, expected)
        
        # wav converted on the Web from a Web mp3
        bells_wav = os.path.join(self.snd_file_data, 'church_bells.wav')
        sound_md = SoundProcessor.soundfile_metadata(bells_wav)
        expected = {'duration' : datetime.timedelta(seconds=54.15),
                    'sample_rate' : 48000
                    }
        self.assertDictEqual(sound_md, expected)
         
        # A Xeno Canto recording:
        xc_recording_mp3 = os.path.join(self.snd_file_data, 'mp3_file_xeno_canto.mp3')
        sound_md = SoundProcessor.soundfile_metadata(xc_recording_mp3)
        expected = {'duration' : datetime.timedelta(seconds=18.88),
                    'sample_rate' : 44100
                    }
        self.assertDictEqual(sound_md, expected)
        
        # Random music:
        music_mp3 = os.path.join(self.snd_file_data, 'music.mp3')
        sound_md = SoundProcessor.soundfile_metadata(music_mp3)
        expected = {'duration' : datetime.timedelta(minutes=3, seconds=34.6),
                    'sample_rate' : 44100
                    }
        self.assertDictEqual(sound_md, expected)
        

# ------------- Main ------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()