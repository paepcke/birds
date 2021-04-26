'''
Created on Apr 21, 2021

@author: paepcke
'''
import os
from pathlib import Path
import tempfile
import unittest
import warnings

import librosa
import pandas as pd

from data_augmentation.augment_audio import AudAugMethod, AudioAugmenter
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import AugmentationGoals, WhenAlreadyDone, Utils

#******TEST_ALL = True
TEST_ALL = False


class AudioAugmentationTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.species1_dir = os.path.join(cls.curr_dir, 'audio_aug_tst_data/DYSMEN_S')
        cls.species2_dir = os.path.join(cls.curr_dir, 'audio_aug_tst_data/HENLES_S')
        one_fname        = os.listdir(cls.species1_dir)[0]
        cls.one_aud_file = os.path.join(cls.species1_dir, one_fname)
        cls.noise_path   = os.path.join(cls.curr_dir, '../lib')
        
        # Hide the UserWarning: PySoundFile failed. Trying audioread instead.
        warnings.filterwarnings(action="ignore",
                                message="PySoundFile failed. Trying audioread instead.",
                                category=UserWarning, 
                                module='', 
                                lineno=0)

    def setUp(self):
        species_root = Path(self.species2_dir).parent.stem
        full_species_root = os.path.abspath(species_root)
        self.aud_augmenter_median = AudioAugmenter (
                full_species_root,
                plot=False,
                overwrite_policy=WhenAlreadyDone.OVERWRITE,
                aug_goals=AugmentationGoals.MEDIAN,
                random_augs = False,
                multiple_augs = False)

        self.aud_augmenter_max = AudioAugmenter (
                full_species_root,
                plot=False,
                overwrite_policy=WhenAlreadyDone.OVERWRITE,
                aug_goals=AugmentationGoals.MAX,
                random_augs = False,
                multiple_augs = False)

        self.aud_augmenter_tenth = AudioAugmenter (
                full_species_root,
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
    
    #*****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_compute_num_augs_per_species(self):
        
        # Get
        #           num_samples
        #      foo           10
        #      bar           25
        #      fum           50
        aug_goals  = AugmentationGoals.MEDIAN
        
        #********
        population = pd.DataFrame.from_dict({'AMADEC_C' : 20, 'ARRAUR_C' : 36}, 
                                            orient='index', 
                                            columns=['num_samples']
                                            )

         
        # population = pd.DataFrame.from_dict({'foo' : 10, 'bar' : 25, 'fum' : 50}, 
                                            # orient='index', 
                                            # columns=['num_samples']
                                            # )
        # #********

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



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()