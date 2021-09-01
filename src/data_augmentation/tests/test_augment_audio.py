'''
Created on Apr 21, 2021

@author: paepcke
'''
import os
from pathlib import Path
import statistics
import tempfile
import unittest
import warnings

import librosa

from data_augmentation.augment_audio import AudioAugmenter
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import AugmentationGoals, Utils
import pandas as pd
import shutil


TEST_ALL = True
#TEST_ALL = False

class AudioAugmentationTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.species1_dir = os.path.join(cls.curr_dir, 'audio_aug_tst_data/DYSMEN_S')
        cls.species2_dir = os.path.join(cls.curr_dir, 'audio_aug_tst_data/HENLES_S')
        one_fname        = os.listdir(cls.species1_dir)[0]
        cls.one_aud_file = os.path.join(cls.species1_dir, one_fname)
        cls.noise_path   = os.path.join(cls.curr_dir, '../lib')
        
        cls.aug_tst_data = os.path.join(cls.curr_dir, 'data/augmentation_tst_data/')
        cls.aug_tst_out_dir = os.path.join(cls.curr_dir, 'data/audio_augmentations/')
        
        cls.wtros_total = 29.88 + 14.55 + 45.53
        cls.yceug_total = 22.81 + 22.54
        cls.legrg_total =  7.46 + 5.40
        
        cls.totals = {'WTROS' : cls.wtros_total,
                      'YCEUG' : cls.yceug_total,
                      'LEGRG' : cls.legrg_total
                      }
        cls.median_duration = statistics.median(list(cls.totals.values()))


        # Hide the UserWarning: PySoundFile failed. Trying audioread instead.
        warnings.filterwarnings(action="ignore",
                                message="PySoundFile failed. Trying audioread instead.",
                                category=UserWarning, 
                                module='', 
                                lineno=0)

    def setUp(self):
        # Remove previously created augmentations:
        shutil.rmtree(self.aug_tst_out_dir, ignore_errors=True)
        
        species_root = Path(self.species2_dir).parent.stem
        self.full_species_root = os.path.abspath(species_root)

    def tearDown(self):
        shutil.rmtree(self.aug_tst_out_dir, ignore_errors=True)

    #------------------------------------
    # test_add_noise 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_add_noise(self):
        with tempfile.TemporaryDirectory(prefix='aud_tests', dir='/tmp') as tmpdir_nm:

            out_file = SoundProcessor.add_background(self.one_aud_file, self.noise_path, tmpdir_nm)
            # Can't do more than try to load the new
            # file and check its length against manually
            # examined truth:
            self.assertEqualDurSR(out_file, self.one_aud_file)
        
    #------------------------------------
    # test_change_sample_volume 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_change_sample_volume(self):

        with tempfile.TemporaryDirectory(prefix='aud_tests', dir='/tmp') as tmpdir_nm:

            out_file = SoundProcessor.change_sample_volume(self.one_aud_file, tmpdir_nm)
            # Can't do more than try to load the new
            # file and check its length against manually
            # examined truth:
            self.assertEqualDurSR(out_file, self.one_aud_file)

    #------------------------------------
    # test_change_sample_volume 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_time_shift(self):

        with tempfile.TemporaryDirectory(prefix='aud_tests', dir='/tmp') as tmpdir_nm:

            out_file = SoundProcessor.time_shift(self.one_aud_file, tmpdir_nm)
            # Can't do more than try to load the new
            # file and check its length against manually
            # examined truth:
            self.assertEqualDurSR(out_file, self.one_aud_file)


    #------------------------------------
    # test_compute_num_augs_per_species 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_compute_num_augs_per_species(self):
        
        # Get
        #           num_samples
        #      foo           10
        #      bar           25
        #      fum           50
        aug_goals  = AugmentationGoals.MEDIAN
        
        population = pd.DataFrame.from_dict({'foo' : 10, 'bar' : 25, 'fum' : 50}, 
                                            orient='index', 
                                            columns=['num_samples']
                                            )

        Utils.compute_num_augs_per_species(aug_goals, population)
        num_samples = population.loc[:,'num_samples']
        med = num_samples.median()
        print(f"Median: {med}")
        
        # species foo must receive med-10  = 15           augmentations
        #         bar              med-25  =  0           augmentations
        #         fum              med-50  = -25 --> 0    augmentations

        truth = {'foo' : 15, 'bar' : 0, 'fum' : 0}
        res   = Utils.compute_num_augs_per_species(aug_goals, population)
        self.assertDictEqual(truth, res)
        
    #------------------------------------
    # test_required_species_seconds
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_required_species_seconds(self):
        
        augmenter = AudioAugmenter(self.aug_tst_data, unittesting=True)
        
        # Get dict mapping SpeciesRecordingAsset
        # instances to number of seconds of augmentation
        # needed. Each SpeciesRecordingAsset provides:
        #    o available_seconds
        #    o species
        #
        # The test set has the following play durations:
        #    wtros_bird1.mp3 : 29.88
        #    wtros_bird2.mp3 : 14.55
        #    wtros_bird3.mp3 : 45.53
        #   
        #    yceug_bird1.mp3 : 22.81
        #    yceug_bird2.mp3 : 22.54
        #
        #    legrg_bird1.mp3 :  7.46
        #    legrg_bird2.mp3 :  5.40

        max_durations    = max(self.wtros_total, self.yceug_total, self.legrg_total)
        median_durations = statistics.median([self.wtros_total, self.yceug_total, self.legrg_total])
        tenth_durations  = max_durations / 10.

        # Test goal MAX:
        species_assets = augmenter._required_species_seconds(self.aug_tst_data,
                                                             AugmentationGoals.MAX
                                                             )
        
        for asset, needed_seconds in species_assets.items():
            species      = asset.species
            true_total   = self.totals[species]
            true_needed  = round(max_durations - true_total, 2)
            self.assertEqual(round(needed_seconds, 2), true_needed)

        # Test goal MEDIAN:
        species_assets = augmenter._required_species_seconds(self.aug_tst_data,
                                                             AugmentationGoals.MEDIAN
                                                             )
        
        for asset, needed_seconds in species_assets.items():
            species      = asset.species
            true_total   = self.totals[species]
            true_needed  = round(median_durations - true_total, 2)
            self.assertEqual(round(needed_seconds, 2), true_needed)

        # Test goal TENTH:
        species_assets = augmenter._required_species_seconds(self.aug_tst_data,
                                                             AugmentationGoals.TENTH
                                                             )
        
        for asset, needed_seconds in species_assets.items():
            species      = asset.species
            true_total   = self.totals[species]
            true_needed  = round(tenth_durations - true_total, 2)
            self.assertEqual(round(needed_seconds, 2), true_needed)

        # Test goal fixed number of seconds: 100:
        species_assets = augmenter._required_species_seconds(self.aug_tst_data,
                                                             AugmentationGoals.NUMBER,
                                                             absolute_seconds=100
                                                             )
        
        for asset, needed_seconds in species_assets.items():
            species      = asset.species
            true_total   = self.totals[species]
            true_needed  = round(100 - true_total, 2)
            self.assertEqual(round(needed_seconds, 2), true_needed)


    #------------------------------------
    # test_specify_augmentation_tasks
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_specify_augmentation_tasks(self):
        
        augmenter = AudioAugmenter(self.aug_tst_data, unittesting=True)
        
        # Test goal MAX:
        
        # Get asset --> secs_needed dict:
        species_assets = augmenter._required_species_seconds(self.aug_tst_data,
                                                             AugmentationGoals.MAX
                                                             )
        task_list = augmenter.specify_augmentation_tasks(self.aug_tst_data,
                                                         AugmentationGoals.MAX
                                                         )
        # Add up the seconds that would be augmented
        # by each task, and check that they are roughly
        # equal to the required additional seconds for each
        # species:
        species_gained_secs_tally = {}
        for task in task_list:
            species = task.species
            try:
                species_gained_secs_tally[species] += task.duration_added
            except KeyError:
                # First encounter with this species:
                species_gained_secs_tally[species] = task.duration_added

        # Check gained against needed:
        for asset, secs_needed in species_assets.items():
            species = asset.species
            self.assertGreaterEqual(species_gained_secs_tally[species],
                                    secs_needed
                                    )

    #------------------------------------
    # test_augmentation
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_augmentation(self):
        
        _augmenter = AudioAugmenter(self.aug_tst_data,
                                   num_workers=1,
                                   aug_goal=AugmentationGoals.MEDIAN
                                   )
        
        
        # Only LEGRG was less than the 
        # median of available recordings across
        # species (~45 sec). So only LEGRG should 
        # have been augmented:
        
        self.assertEqual(len(os.listdir(self.aug_tst_out_dir)), 1)

        # Get sum of new LEGRG recording durations:
        new_legrg_durations = SoundProcessor.find_total_recording_length(os.path.join(self.aug_tst_out_dir, 'LEGRG'))
        # New plus already existing durations should be 
        # about at the median:
        self.assertGreaterEqual(new_legrg_durations + self.totals['LEGRG'], self.median_duration)
        
        # Clear out the augmentations for the
        # next test:
        shutil.rmtree(self.aug_tst_out_dir, ignore_errors=True)
        
        # Next: Try MAX as a goal: i.e. lift LEGRG and YECUG to
        # about 88:  
        
        _augmenter = AudioAugmenter(self.aug_tst_data,
                                   num_workers=1,
                                   aug_goal=AugmentationGoals.MAX
                                   )
        new_legrg_durations = SoundProcessor.find_total_recording_length(os.path.join(self.aug_tst_out_dir, 'LEGRG'))
        new_yceug_durations = SoundProcessor.find_total_recording_length(os.path.join(self.aug_tst_out_dir, 'YCEUG'))

        self.assertGreaterEqual(new_yceug_durations + self.totals['YCEUG'], 88)
        print('foo')

    #------------------------------------
    # test_species_filter
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_species_filter(self):
        
        # Test whether we can limit augs to 
        # a given list of species, and ignore
        # others that are present under the species
        # root dir:
        
        # Make it a single species, the one with the
        # most recordings:
        _augmenter = AudioAugmenter(self.aug_tst_data,
                                   num_workers=1,
                                   aug_goal=AugmentationGoals.MAX,
                                   species_filter=['WTROS']
                                   )
        # Should have done nothing:
        with self.assertRaises(FileNotFoundError):
            os.listdir(self.aug_tst_out_dir)
            
        # Now filter for two species out of the three:
        _augmenter = AudioAugmenter(self.aug_tst_data,
                                   num_workers=1,
                                   aug_goal=AugmentationGoals.MAX,
                                   species_filter=['WTROS', 'LEGRG']
                                   )
        new_legrg_durations = SoundProcessor.find_total_recording_length(os.path.join(self.aug_tst_out_dir, 
                                                                                      'LEGRG'))
        
        self.assertEqual(new_legrg_durations, 72)
        self.assertListEqual(os.listdir(self.aug_tst_out_dir), ['LEGRG'])

# ------------------------- Utilities -----------------------

    #------------------------------------
    # cmp_duration_and_sample_rates 
    #-------------------
    
    def assertEqualDurSR(self, fname1, fname2):
        '''
        Tests whether the audio duration and
        sampling rates of two audio files are
        equal.
        
        :param fname1: audio file 1 path
        :type fname1: str
        :param fname2: audio file 2 path
        :type fname2: str
        :raises AssertionError
        '''
        
        sound1, sr1 = librosa.load(fname1)
        sound2, sr2 = librosa.load(fname2)
        
        self.assertEqual(sr1, sr2)

        dur1 = librosa.get_duration(sound1)
        dur2 = librosa.get_duration(sound2)
        
        self.assertEqual(dur1, dur2)


# ---------------- Main ---------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()