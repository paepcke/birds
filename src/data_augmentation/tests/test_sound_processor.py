'''
Created on Apr 29, 2021

@author: paepcke
'''
import datetime
import gzip
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import unittest

from birdsong.utils.utilities import FileUtils
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Interval
import pandas as pd


TEST_ALL = True
#TEST_ALL = False

# NOTE: SoundProcessor is also exercised in 
#       other unittests, such as create_spectrogram()

class TestSoundProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.snd_file_data = os.path.join(cls.cur_dir, "data/mp3_and_wav_data/")
        
        # Recordings for testing recording_lengths_by_species():
        cls.more_recordings_dir = os.path.join(cls.cur_dir, "sound_data")
        
    def setUp(self):
        inventory_dir = FileUtils.make_manifest_dir_name(self.more_recordings_dir)
        shutil.rmtree(inventory_dir, ignore_errors=True)

    def tearDown(self):
        inventory_dir = FileUtils.make_manifest_dir_name(self.more_recordings_dir)
        shutil.rmtree(inventory_dir, ignore_errors=True)

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
        
    #------------------------------------
    # test_find_recording_lengths
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_find_recording_lengths(self):
        
        duration_distrib_df = SoundProcessor.find_recording_lengths(self.snd_file_data)

        expected = pd.DataFrame([[ 54.15, '0:00:54.150000'],
                                 [  18.88,'0:00:18.880000'],
                                 [ 214.60,'0:03:34.600000'],
                                 [  54.19,'0:00:54.190000']
                                 ],
                                index=['church_bells.wav',
                                       'mp3_file_xeno_canto.mp3',
                                       'music.mp3',
                                       'church_bells.mp3'
                                       ],
                                columns=['recording_length_secs',
                                         'recording_length_hhs_mins_secs'
                                         ]
                                )
        self.assertTrue(all((expected == duration_distrib_df)))

        total_dur = SoundProcessor.find_total_recording_length(self.snd_file_data)
        self.assertEqual(total_dur, sum([54.15,18.88,214.60,54.19]))
        
        # Test getting duration of just a single file:
        one_snd_file = os.path.join(self.snd_file_data, 'church_bells.mp3')
        duration_distrib_df = SoundProcessor.find_recording_lengths(one_snd_file)
        
        expected = pd.DataFrame([[  54,'0:00:54']],
                                index=['church_bells.mp3'],
                                columns=['recording_length_secs',
                                         'recording_length_hhs_mins_secs'
                                         ]
                                )
        
        self.assertTrue(all((expected == duration_distrib_df)))
        
    #------------------------------------
    # test_recording_lengths_by_species_no_inventory
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_recording_lengths_by_species_no_inventory(self):
        
        dur_distrib_by_species = SoundProcessor.recording_lengths_by_species(self.more_recordings_dir)

        expected = pd.DataFrame([[213.89, '0:03:33.890000'],
                                 [145.59, '0:02:27.590000']
                                 ],
                                 index=['DYSMEN_S', 'HENLES_S'],
                                 columns=['total_recording_length (secs)', 'duration (hrs:mins:secs)']
                                 )
        self.assertTrue(all(dur_distrib_by_species == expected))
        
        inventory_dir = FileUtils.make_manifest_dir_name(self.more_recordings_dir)
        # Should not exist:
        self.assertFalse(os.path.exists(inventory_dir))

    #------------------------------------
    # test_recording_lengths_by_species_inventory_creation
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_recording_lengths_by_species_inventory_creation(self):
        
        inventory_dir = FileUtils.make_manifest_dir_name(self.more_recordings_dir)

        # Find total recordings lengths for each species,
        # but also save each file's duration in 
        #   Audio_Manifest_<self.more_recordings_dir-name>.
        # The 'True' makes the method choose that default location
        # for placing the detailed durations:
        dur_distrib_by_species = SoundProcessor.recording_lengths_by_species(
            self.more_recordings_dir,
            save_recording_durations=True
            )

        expected = pd.DataFrame([[213.89, '0:03:33.890000'],
                                 [147.59, '0:02:27.590000']
                                 ],
                                 index=['DYSMEN_S', 'HENLES_S'],
                                 columns=['total_recording_length (secs)', 'duration (hrs:mins:secs)']
                                 )
        self.assertTrue(all(dur_distrib_by_species == expected))
        
        # Should exist:
        self.assertTrue(os.path.exists(inventory_dir))
        
        # Should have two compressed json files:
        expected = set(['DYSMEN_S_manifest.json.gz', 'HENLES_S_manifest.json.gz'])
        self.assertSetEqual(set(os.listdir(inventory_dir)), expected)
        
        # Read the DYSMEN gz file:
        dysmen_manifest_path = os.path.join(inventory_dir, 'DYSMEN_S_manifest.json.gz')
        rec_lengths_dict = json.loads(SoundProcessor.read_gzipped_file(dysmen_manifest_path))
        
        expected = {
            'recording_length_secs':
        	    {'SONG_Dysithamnusmentalis_xc513.mp3': 21.96,
        	     'SONG_Dysithamnusmentalis_xc518466.mp3': 46.79,
        	     'SONG_Dysithamnusmentalis_xc531750.mp3': 23.17,
        	     'SONG_Dysithamnusmentalis_xc50519.mp3': 10.81,
        	     'SONG_Dysithamnusmentalis_xc511477.mp3': 58.86,
        	     'SONG_Dysithamnusmentalis_xc548015.mp3': 52.3
                 },
        	'recording_length_hhs_mins_secs':
        	    {'SONG_Dysithamnusmentalis_xc513.mp3': '0:00:21.960000',
        	     'SONG_Dysithamnusmentalis_xc518466.mp3': '0:00:46.790000',
        	     'SONG_Dysithamnusmentalis_xc531750.mp3': '0:00:23.170000',
        	     'SONG_Dysithamnusmentalis_xc50519.mp3': '0:00:10.810000',
        	     'SONG_Dysithamnusmentalis_xc511477.mp3': '0:00:58.860000',
        	     'SONG_Dysithamnusmentalis_xc548015.mp3': '0:00:52.300000'
        	     }
             }
        
        self.assertDictEqual(rec_lengths_dict, expected)

    #------------------------------------
    # test_add_background
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_add_background(self):
        
        main_species_path    = os.path.join(self.more_recordings_dir,
                                            'DYSMEN_S/SONG_Dysithamnusmentalis_xc50519.mp3')
        overlay_species_path = os.path.join(self.more_recordings_dir,
                                            'HENLES_S/SONG_Henicorhinaleucosticta_xc259378.mp3')
        
        main_dur    = SoundProcessor.find_total_recording_length(main_species_path)
        overlay_dur = SoundProcessor.find_total_recording_length(overlay_species_path)

        # Make randomness reproducible for testing:
        random.seed(42)
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='sound_overlay_test') as out_dir:
            (new_fname, noise_used) = SoundProcessor.add_background(main_species_path,
                                                                    overlay_species_path, 
                                                                    out_dir
                                                                    )
            
            self.assertEqual(noise_used, overlay_species_path)
            
            after_main_dur    = SoundProcessor.find_total_recording_length(main_species_path)
            after_overlay_dur = SoundProcessor.find_total_recording_length(overlay_species_path)
            
            # Original main and overlay files should
            # be unaffected:
            self.assertEqual(after_main_dur, main_dur)
            self.assertEqual(after_overlay_dur, overlay_dur)
            
            mix_dur = SoundProcessor.find_total_recording_length(new_fname)
            self.assertEqual(int(mix_dur), int(main_dur))
            
            (new_fname, noise_used) = SoundProcessor.add_background(main_species_path,
                                                                    os.path.join(self.more_recordings_dir,
                                                                                 'HENLES_S'),
                                                                    out_dir
                                                                    )
            # Since we set the random seed, the
            # noise file picked is predictable, b/c
            # a fixed number of calls to random.randint()
            # are involved in add_background():
            self.assertEqual(Path(noise_used).name, 'SONG_Henicorhinaleucosticta_xc259384.mp3') 

    #------------------------------------
    # test_gzip_file
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_gzip_file(self):
        
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='gzip_test') as dir_nm:
            content = ("This is a test,\n"
                       "So give it a rest."
                       )
            fname = os.path.join(dir_nm, 'gzip_tst.txt')
            with open(fname, 'w') as fd:
                fd.write(content)
            
            # Read it back in the clear:
            with open(fname, 'r') as fd:
                recovered_content = fd.read()
                self.assertEqual(recovered_content, content)
                
            # Compress the file:
            gz_content_fname = SoundProcessor.gzip_file(fname, delete=False)
            
            # Should have two files now:
            self.assertEqual(len(os.listdir(dir_nm)), 2)
            
            # Original should be untouched:
            # Read it back in the clear:
            with open(fname, 'r') as fd:
                recovered_content = fd.read()
                self.assertEqual(recovered_content, content)

            # Read the compressed version:
            with gzip.open(gz_content_fname, 'rb') as fd:
                recovered_content = fd.read().decode('utf8')
                self.assertEqual(recovered_content, content)

            # Same reading-back should work with
            # SoundProcessor.read_gzipped_file:
            recovered_content = SoundProcessor.read_gzipped_file(gz_content_fname)
            self.assertEqual(recovered_content, content)

    #------------------------------------
    # test_gzip_file_no_deletion
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_gzip_file_no_deletion(self):
        
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='gzip_test') as dir_nm:
            content = ("This is a test,\n"
                       "So give it a rest."
                       )
            fname = os.path.join(dir_nm, 'gzip_tst.txt')
            with open(fname, 'w') as fd:
                fd.write(content)
                
            # Compress the file:
            gz_content_fname = SoundProcessor.gzip_file(fname, 
                                                        delete=True)
            
            # Should have only the gzipped file now:
            self.assertEqual(os.listdir(dir_nm),
                             [Path(gz_content_fname).name]
                             )

# ------------- Main ------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()