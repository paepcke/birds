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

from data_augmentation.augment_audio import AudAugMethod, AudioAugmenter
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import AugmentationGoals, WhenAlreadyDone, Utils
import pandas as pd


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
        
        # Hide the UserWarning: PySoundFile failed. Trying audioread instead.
        warnings.filterwarnings(action="ignore",
                                message="PySoundFile failed. Trying audioread instead.",
                                category=UserWarning, 
                                module='', 
                                lineno=0)

    def setUp(self):
        species_root = Path(self.species2_dir).parent.stem
        self.full_species_root = os.path.abspath(species_root)
        self.aud_augmenter_median = AudioAugmenter (
                self.full_species_root,
                plot=False,
                overwrite_policy=WhenAlreadyDone.OVERWRITE,
                aug_goals=AugmentationGoals.MEDIAN,
                random_augs = False,
                multiple_augs = False)

        self.aud_augmenter_max = AudioAugmenter (
                self.full_species_root,
                plot=False,
                overwrite_policy=WhenAlreadyDone.OVERWRITE,
                aug_goals=AugmentationGoals.MAX,
                random_augs = False,
                multiple_augs = False)

        self.aud_augmenter_tenth = AudioAugmenter (
                self.full_species_root,
                plot=False,
                overwrite_policy=WhenAlreadyDone.OVERWRITE,
                aug_goals=AugmentationGoals.TENTH,
                random_augs = False,
                multiple_augs = False)

    def tearDown(self):
        pass

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
    # test_create_new_sample
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_create_new_sample(self):

        with tempfile.TemporaryDirectory(prefix='aud_tests', dir='/tmp') as tmpdir_nm:

            for method in AudAugMethod:
                out_file = self.aud_augmenter_median.create_new_sample(self.one_aud_file, 
                                                                tmpdir_nm,
                                                                method)
                # Can't do more than try to load the new
                # file and check its length against manually
                # examined truth:
                self.assertEqualDurSR(out_file, self.one_aud_file)


    #------------------------------------
    # test_generate_all_augmentations_median 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_generate_all_augmentations_median(self):
        self.aud_augmenter_median.generate_all_augmentations()
        dirs_with_augs = os.path.join(self.curr_dir,
                                      'Augmented_samples_-0.33n-0.33ts-0.33w-exc/DYSMEN_S'
                                      )
        # Should have two new audio files:
        #   o DYSMEN_S had 2 to start with
        #   o HENLES_S had 6 to start with
        #   o We asked for median goal, which is 4
        #   o So DYSMEN_S should have received 2 new augs
        new_files = os.listdir(dirs_with_augs)
        self.assertEqual(len(new_files), 2)

    #------------------------------------
    # test_generate_all_augmentations_median_species_filter 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_generate_all_augmentations_median_species_ilter (self):
        
        augmenter_median = AudioAugmenter (
                self.full_species_root,
                plot=False,
                overwrite_policy=WhenAlreadyDone.OVERWRITE,
                aug_goals=AugmentationGoals.MEDIAN,
                random_augs = False,
                multiple_augs = False,
                species_filter = {'DYSMEN_S' : 4}
                )
        
        augmenter_median.generate_all_augmentations()

        # Only augs for DYSMEN_S should have been created,
        # not for HENLES_S:
        
        dir_with_augs = os.path.join(self.curr_dir,
                                      'Augmented_samples_-0.33n-0.33ts-0.33w-exc/'
                                      )
        self.assertEqual(len(os.listdir(dir_with_augs)), 1)
                         
        dysmen_aug_dir = os.path.join(dir_with_augs, 'DYSMEN_S')
        new_files = os.listdir(dysmen_aug_dir)
        self.assertEqual(len(new_files), 4)

    #------------------------------------
    # test_generate_all_augmentations_max 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_generate_all_augmentations_max(self):
        self.aud_augmenter_max.generate_all_augmentations()
        dirs_with_augs = os.path.join(self.curr_dir,
                                      'Augmented_samples_-0.33n-0.33ts-0.33w-exc/DYSMEN_S'
                                      )
        # Should have two new audio files:
        #   o DYSMEN_S had 2 to start with
        #   o HENLES_S had 6 to start with
        #   o We asked for median goal, which is 4
        #   o So DYSMEN_S should have received 2 new augs
        new_files = os.listdir(dirs_with_augs)
        self.assertEqual(len(new_files), 4)

    #------------------------------------
    # test_generate_all_augmentations_tenth
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_generate_all_augmentations_tenth(self):
        # Even the minimally recorded DYSMEN_S has 
        # 2 recordings. 1/10 of the six HENLES_S species
        # is 1. So we expect the following to do nothing:
        self.aud_augmenter_tenth.generate_all_augmentations()
        dirs_with_augs = os.path.join(self.curr_dir,
                                      'Augmented_samples_-0.33n-0.33ts-0.33w-exc'
                                      )
        new_files = os.listdir(dirs_with_augs)
        self.assertEqual(len(new_files), 0)

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
    # test_required_species_minutes
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_required_species_minutes(self):
        
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

        wtros_total = 29.88 + 14.55 + 45.53
        yceug_total = 22.81 + 22.54
        legrg_total =  7.46 + 5.40
        
        totals = {'WTROS' : wtros_total,
                  'YCEUG' : yceug_total,
                  'LEGRG' : legrg_total
                  }
        
        max_durations    = max(wtros_total, yceug_total, legrg_total)
        median_durations = statistics.median([wtros_total, yceug_total, legrg_total])
        tenth_durations  = max_durations / 10.

        # Test goal MAX:
        species_assets = augmenter._required_species_seconds(self.aug_tst_data,
                                                             AugmentationGoals.MAX
                                                             )
        
        for asset, needed_seconds in species_assets.items():
            species      = asset.species
            true_total   = totals[species]
            true_needed  = round(max_durations - true_total, 2)
            self.assertEqual(round(needed_seconds, 2), true_needed)

        # Test goal MEDIAN:
        species_assets = augmenter._required_species_seconds(self.aug_tst_data,
                                                             AugmentationGoals.MEDIAN
                                                             )
        
        for asset, needed_seconds in species_assets.items():
            species      = asset.species
            true_total   = totals[species]
            true_needed  = round(median_durations - true_total, 2)
            self.assertEqual(round(needed_seconds, 2), true_needed)

        # Test goal TENTH:
        species_assets = augmenter._required_species_seconds(self.aug_tst_data,
                                                             AugmentationGoals.TENTH
                                                             )
        
        for asset, needed_seconds in species_assets.items():
            species      = asset.species
            true_total   = totals[species]
            true_needed  = round(tenth_durations - true_total, 2)
            self.assertEqual(round(needed_seconds, 2), true_needed)

        # Test goal fixed number of seconds: 100:
        species_assets = augmenter._required_species_seconds(self.aug_tst_data,
                                                             AugmentationGoals.NUMBER,
                                                             absolute_seconds=100
                                                             )
        
        for asset, needed_seconds in species_assets.items():
            species      = asset.species
            true_total   = totals[species]
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