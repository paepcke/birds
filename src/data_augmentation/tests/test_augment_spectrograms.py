'''
Created on Apr 27, 2021

@author: paepcke
'''
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from data_augmentation.augment_spectrograms import SpectrogramAugmenter
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import ImgAugMethod, Utils
from data_augmentation.utils import WhenAlreadyDone, AugmentationGoals


#*******TEST_ALL = True
TEST_ALL = False


class TestAugmentSpectrograms(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.species_dir = os.path.join(cls.curr_dir, 'spectro_data/AMADEC')
        one_fname        = os.listdir(cls.species_dir)[0]
        cls.one_spectro_file = os.path.join(cls.species_dir, one_fname)

    def setUp(self):
        # Dir containing the species subdirs
        # AMADEC and FONTANA:
        species_root = Path(self.species_dir).parent.stem
        full_species_root = os.path.abspath(species_root)
        self.spectro_augmenter_median = SpectrogramAugmenter (
                full_species_root,
                plot=False,
                overwrite_policy=WhenAlreadyDone.OVERWRITE,
                aug_goals=AugmentationGoals.MEDIAN
                )

        self.spectro_augmenter_max = SpectrogramAugmenter (
                full_species_root,
                plot=False,
                overwrite_policy=WhenAlreadyDone.OVERWRITE,
                aug_goals=AugmentationGoals.MAX
                )

        self.spectro_augmenter_tenth = SpectrogramAugmenter (
                full_species_root,
                plot=False,
                overwrite_policy=WhenAlreadyDone.OVERWRITE,
                aug_goals=AugmentationGoals.TENTH
                )

        self.full_species_root = full_species_root

    def tearDown(self):
        pass

# ---------------------- Tests ---------------------

    #------------------------------------
    # test_create_random_noise
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_create_random_noise(self):
        spectro = SoundProcessor.load_spectrogram(self.one_spectro_file)
        new_spectro, _fname = SoundProcessor.random_noise(spectro, 
                                                         noise_type='uniform')
        self.assertTrue((new_spectro > spectro).any())
        self.assertTrue((new_spectro <= 255).all())

    #------------------------------------
    # test_create_spectrogram
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_create_new_sample(self):
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            # Time masking:
            aug_spectro_path = self.spectro_augmenter_median.create_new_sample(
                    self.one_spectro_file,
                    dst_dir,
                    ImgAugMethod.TMASK
                    )
            # The two images shouldn't be the same:
            orig = SoundProcessor.load_spectrogram(self.one_spectro_file)
            aug  = SoundProcessor.load_spectrogram(aug_spectro_path)
            
            self.assertFalse((orig == aug).all())

            # Frequency masking:
            
            aug_spectro_path = self.spectro_augmenter_median.create_new_sample(
                    self.one_spectro_file,
                    dst_dir,
                    ImgAugMethod.FMASK
                    )
            # The two images shouldn't be the same:
            orig = SoundProcessor.load_spectrogram(self.one_spectro_file)
            aug  = SoundProcessor.load_spectrogram(aug_spectro_path)
            self.assertFalse((orig == aug).all())

    #------------------------------------
    # test_generate_all_augmentations_median
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_generate_all_augmentations_median(self):
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            
            out_dir = self.prep_aug_tmp_dirs(dst_dir)            
            # Tell the augmenter where the src and dest roots are:
            self.spectro_augmenter_median.input_dir_path = dst_dir
            self.spectro_augmenter_median.output_dir_path = out_dir
            
            # AMADEC has 1 spectro, FORANA has 5
            # Median is (5+1)/2 = 3. So AMADEC needs
            # (Median - existing) = 2 new spectros:
            num_augs_needed = 2
            
            self.spectro_augmenter_median.generate_all_augmentations()
            
            # Should have one directory in aug_spectros
            new_dirs  = Utils.listdir_abs(out_dir)
            self.assertTrue(len(new_dirs) == 1)
            # AMADEC subdir should have 2 new files
            new_files = os.listdir(new_dirs[0])  
            self.assertTrue(len(new_files), num_augs_needed)

    #------------------------------------
    # test_generate_all_augmentations_max
    #-------------------

    #********88@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_generate_all_augmentations_max(self):
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            
            out_dir = self.prep_aug_tmp_dirs(dst_dir)            
            # Tell the augmenter where the src and dest roots are:
            self.spectro_augmenter_max.input_dir_path = dst_dir
            self.spectro_augmenter_max.output_dir_path = out_dir
            
            # AMADEC has 1 spectro, FORANA has 5
            # MAX is 5, So AMADEC needs 4 augementation:
            
            num_augs_needed = 4
            
            self.spectro_augmenter_max.generate_all_augmentations()
            
            # Should have one directory in aug_spectros
            new_dirs  = Utils.listdir_abs(out_dir)
            self.assertTrue(len(new_dirs) == 1)
            # AMADEC subdir should have 2 new files
            new_files = os.listdir(new_dirs[0])  
            self.assertTrue(len(new_files), num_augs_needed)
            
# --------------------------- Utilities --------------

    #------------------------------------
    # prep_aug_tmp_dirs 
    #-------------------
    
    def prep_aug_tmp_dirs(self, dst_tmp_dir):
        '''
        Copies AMADEC single-spectrogram directory,
        and FORANA 5-spectrogram dir to the given tmp 
        dir. Creates dir 'aug_spectros' in that same
        tmp dir. Returns path to that aug_spectros
        dir.
         
        :param dst_tmp_dir: temporary directory
        :type dst_tmp_dir: src
        :return: output directory for future spectro augmentations
        :rtype: str
        '''

        # Do all testing in the tmp dir, where
        # all files/dirs will be deleted automatically:
        
        for species_dir in Utils.listdir_abs(self.full_species_root):
            species_name = Path(species_dir).stem
            dst_species_dir = os.path.join(dst_tmp_dir,species_name)
            shutil.copytree(species_dir, dst_species_dir)
            
        # Dir where augmentations are to be placed,
        # one subdir per species:
        out_dir = os.path.join(dst_tmp_dir, 'aug_spectros')
        os.mkdir(out_dir)
        return out_dir


# ---------------------- Main -----------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()