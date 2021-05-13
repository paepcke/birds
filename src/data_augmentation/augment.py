#!/usr/bin/env python
# coding: utf-8

import argparse
from enum import Enum
import os
import random
import sys

from logging_service import LoggingService

from data_augmentation.sound_processor import SoundProcessor
import data_augmentation.utils as utils
import numpy as np


# How many augmentations to create 
# for each species. Measured against
# the number of samples of the species 
# with the most number of available
# samples.
# Meaning: 
#    TENTH: all species will have at least a 				  
#           10th of samples in the most populous 
#           species 	  
#   MEDIAN: all species will have at least the median
#    	    number of samples in the species populations
#      MAX: all species will end up with the number of
#           species of the most populously represented species
#           in the training set: 

class AugmentationGoals(Enum):
    TENTH   = 0
    MEDIAN  = 1
    MAX     = 2

#---------------- Class Augmenter ---------------

class Augmenter:

    ADD_NOISE   = 1/3
    TIME_SHIFT  = 1/3
    WARP        = 1/3
    
    P_DIST    = [ADD_NOISE, TIME_SHIFT, WARP]
    AUDIO_AUG_NAMES = ["add_noise", "time_shift", "warp"]
    AUG_SPECTROGRAMS_DIR = "spectrograms_augmented/"
    AUG_WAV_DIR = "wav_augmented/"
    NOISE_PATH = "data_augmentation/lib/Noise_Recordings/"
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 input_dir_path,
                 plot=False,
                 overwrite_policy=False,
                 aug_goals=AugmentationGoals.MEDIAN,
                 random_augs = False,
                 multiple_augs = False,):

        '''
        
        :param input_dir_path: directory holding .wav files
        :type input_dir_path: str
        :param plot: whether or not to plot informative chars 
            along the way
        :type plot: bool
        :param overwrite_policy: if true, don't ask each time
            previously created work will be replaced
        :type overwrite_policy: bool 
        :param aug_goals: either an AugmentationGoals member,
               or a dict with a separate AugmentationGoals
               for each species: {species : AugmentationGoals}
               (See definition of AugmentationGoals)
        :type aug_goals: {AugmentationGoals | {str : AugmentationGoals}}
        :param random_augs: if this is true, will randomly choose augmentation 
            to use for each new sample
        :type random_augs: bool
        :param multiple_augs: if we want to allow multiple augmentations per sample 
            (ie time shift and warp)):
        :type multiple_augs: bool
        '''

        self.log = LoggingService()
    
        self.input_dir_path   = input_dir_path
        self.multiple_augs    = multiple_augs
        self.plot             = plot
        self.overwrite_freely = overwrite_policy

        # Get dataframe with row lables being the
        # species, and one col with number of samples
        # in the respective species:
        #       num_species
        # sp1       10
        # sp2       15
        #      ..

        self.sample_distrib_df = utils.sample_compositions_by_species(input_dir_path, 
                                                                      augmented=False)
        
        if plot:
            # Plot a distribution:
            self.sample_distrib_df.plot.bar()

        # Build a dict with number of augmentations to do
        # for each species:
        self.augs_to_do = self.compute_num_augs_per_species(aug_goals, self.sample_distrib_df)

        if random_augs:
            self.output_dir_path = f"{input_dir_path[:-1]}_augmented_samples_random"

        else:
            assert(self.ADD_NOISE + self.TIME_SHIFT + self.WARP == 1)
            self.output_dir_path = f"{input_dir_path[:-1]}_augmented_samples-{self.ADD_NOISE:.2f}n-{self.TIME_SHIFT:.2f}ts-{self.WARP:.2f}w"

        if self.multiple_augs:
            self.output_dir_path += "/"
        else:
            # Indicate that augmentations are mutually exclusive
            self.output_dir_path += "-exc/"  

        self.log.info(f"Results will be in {self.output_dir_path}")

        # Creates output file structure
        # Self.output_dir_path
        #       |-- AUG_WAV_DIR
        #       |         |----
        #       |         |----
        #       |-- AUG_SPECTROGRAMS_DIR
        #       |         |----
        #       |         |----
        utils.create_folder(self.output_dir_path, self.overwrite_freely)
        utils.create_folder(os.path.join(self.output_dir_path, self.AUG_WAV_DIR),
                            self.overwrite_freely)
        utils.create_folder(os.path.join(self.output_dir_path, self.AUG_SPECTROGRAMS_DIR),
                            self.overwrite_freely)

    #------------------------------------
    # compute_num_augs_per_species
    #-------------------

    def compute_num_augs_per_species(self, aug_volumes, sample_distrib_df):
        '''
        Return a dict mapping species name to 
        number of samples that should be available after
        augmentation. 
        
        The aug_volumes arg is either a dict mapping species name
        to an AugmentationGoals (TENTH, MEDIAN, MAX), or just
        an individual AugmentationGoals.

        The sample_distrib_df is a dataframe whose row labels are 
        species names and the single column's values are numbers
        of available samples for training/validation/test for the
        respective row's species.
        
        :param aug_volumes: how many augmentations for each species
        :type aug_volumes: {AugmentationGoals | {str : AugmentationGoals}}
        :param sample_distrib_df: distribution of initially available
            sample numbers for each species
        :type sample_distrib_df: pandas.DataFrame
        :return: dict mapping each species to the 
            number of samples that need to be created.
        :rtype: {str : int}
        '''
        
        # Get straight array of number of audio samples
        # for each species. Ex: array([6,4,5]) for
        # three species with 6,4, and 5 audio recordings,
        # respectively

        species_np = self.sample_distrib_df.values.flatten()
        index_max, index_min = (self.sample_distrib_df.idxmax().to_numpy()[0], 
                                self.sample_distrib_df.idxmin().to_numpy()[0])
        
        
        # Get number of recordings for
        # the species with the most number
        # of recordings: 
        max_num_samples = np.max(species_np)
        tenth_max_num_samples = max_num_samples//10 +1
        median_num_samples = np.median(species_np)
        
        volumes = {AugmentationGoals.TENTH  : tenth_max_num_samples,
                   AugmentationGoals.MEDIAN : median_num_samples,
                   AugmentationGoals.MAX    : max_num_samples
                   }
        
        aug_requirements = {}
        if type(aug_volumes) == AugmentationGoals:
            for species in sample_distrib_df.index:
                aug_requirements[species] = volumes[aug_volumes]
        else:
            # Have dict of species-name : AugmentationGoals:
            for species in sample_distrib_df.index:
                aug_requirement = aug_volumes[species]
                aug_requirements[species] = volumes[aug_requirement]
        
        self.log.info(f"Median: {median_num_samples},  Min: {np.min(species_np)} ({index_min}) ,  Max: {max_num_samples} ({index_max})")
        self.log.info(f"10% of max is {tenth_max_num_samples}")

        return aug_requirements 

    #------------------------------------
    # generate_all_augmentations
    #-------------------

    def generate_all_augmentations(self):
        '''
        Create new samples via augmentation or each species. 
        Augment the audio and/or spectrogram files to reach 
        the number of spectrograms indicated in the self.aug_requirements.
        
        Spectrograms are created on the way.

        Assumption: self.aug_requirements is a dict mapping 
        species-name : num_required_augmentations
        '''

        for species, rows in self.sample_distrib_df.iterrows():
            num_samples_orig = rows['num_samples']
            self.augment_one_species(species, 
                                      num_samples_orig, 
                                      self.augs_to_do[species]
                                      )
        
        #input(f"Finished for {species}")

        # Clean up directory clutter:
        search_root_dir = os.path.join(self.output_dir_path + self.AUG_SPECTROGRAMS_DIR) 
        os.system(f"find {search_root_dir} -name \".DS_Store\" -delete")
        
        spectro_dir = os.path.join(self.output_dir_path, self.AUG_SPECTROGRAMS_DIR)
        augmented_df = utils.sample_compositions_by_species(spectro_dir, augmented=True)
        augmented_df["total_samples"] = augmented_df.sum(axis=1)
        
        self.log.debug(f"augmented_df: {augmented_df}")

        if self.plot:
            augmented_df.plot.bar(y=["add_bg", "time_shift", "mask", "original"],stacked=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))

        self.log.info("Done")
    #------------------------------------
    # augment_one_species
    #-------------------

    def augment_one_species(self, species, num_samples_orig, threshold):

        species_wav_input_dir  = os.path.join(self.input_dir_path, species)
        species_wav_output_dir = os.path.join(self.output_dir_path, self.AUG_WAV_DIR, species)
        species_spectrogram_output_dir = os.path.join(self.output_dir_path, self.AUG_SPECTROGRAMS_DIR, species)
        
        # Make output folders under self.output_dir_path
        # Returns if either folder already exists
        
        if not (utils.create_folder(species_wav_output_dir, self.overwrite_freely) and 
                utils.create_folder(species_spectrogram_output_dir, self.overwrite_freely)):
            self.log.info(f"Skipping augmentations for {species}")
            return
    
        wav_files = os.listdir(species_wav_input_dir)
        
        # Create original spectrograms
        if num_samples_orig < threshold:
            samples_to_add = int(threshold - num_samples_orig)
            self.create_original_spectrograms(wav_files, 
                                              num_samples_orig, 
                                              species_wav_input_dir, 
                                              species_spectrogram_output_dir)
        else:
            samples_to_add = 0
            self.create_original_spectrograms(wav_files, 
                                              threshold,
                                              species_wav_input_dir, 
                                              species_spectrogram_output_dir)
        self.log.info(f"Num Original Samples for {species}: {len(wav_files)}. Creating {samples_to_add} more samples using augmentations.")
        
        # Cannot do augmentations for species with 0 samples
        if len(wav_files) == 0:
            self.log.info(f"Skipping for {species} since there are no original samples.")
            return
        
        # Create samples_to_add samples using augmentations
        for i in range(samples_to_add):
            # if samples to add more than original samples, 
            # we may have to augment original samples more than once:
            
            sample_name = wav_files[i%len(wav_files)]
            
            # The maximum number of augmentations is equal to 
            # using each available augmentation at most once per sample.
            max_num_augs = utils.count_max_augs(self.P_DIST)
            num_augs_per_sample = np.random.randint(1, max_num_augs+1) if self.multiple_augs else 1
            
            self.create_new_sample(sample_name, 
                                   (species_wav_input_dir, 
                                    species_wav_output_dir, 
                                    species_spectrogram_output_dir),
                                   num_augs=num_augs_per_sample
                                   )

    #------------------------------------
    # create_new_sample 
    #-------------------

    def create_new_sample(self, sample_name, paths, num_augs=1):

        (species_wav_input_dir, species_wav_output_dir, species_spectrogram_output_dir) = paths
        
        aug_choices = np.random.choice(self.AUDIO_AUG_NAMES, 
                                       size=num_augs, 
                                       p=self.P_DIST, 
                                       replace=False)
        # input(f"Aug choices: {aug_choices}")
        # Warping must be done after all the other augmentations take place, 
        # after spectrogram is created
        warp = False
        if "warp" in aug_choices:
            warp = True
            aug_choices = aug_choices.tolist()
            # print(f"Aug chioces as list: {aug_choices}")
            aug_choices.remove("warp")
            # print(f"Aug chioces after: {aug_choices}")
            
        for i in range(len(aug_choices)):
            # print(aug_choices)
            aug_name = aug_choices[i]
            if i != 0:  # if not first augmentation, then, source wav is in output wav directory
                species_wav_input_dir = species_wav_output_dir
            if aug_name == "add_noise":
                
                # Add_noise; which noise to add will be chosen at random
                updated_name = SoundProcessor.add_background(sample_name, 
                                              self.NOISE_PATH, 
                                              species_wav_input_dir, 
                                              species_wav_output_dir, 
                                              len_noise_to_add=5.0)
            elif aug_name == "time_shift":
                updated_name = SoundProcessor.time_shift(sample_name, species_wav_input_dir, species_wav_output_dir)
            sample_name = updated_name
            
        # create new spectrogram if augmented
        if len(aug_choices) != 0:
            sample_name = SoundProcessor.create_spectrograms(sample_name, 
                                             species_wav_output_dir, 
                                             species_spectrogram_output_dir, 
                                             n_mels=128)
            
        if warp:
            #warp
            # if len(aug_choices) +1 > 1:
            #     input(f"num_augs = {len(aug_choices) +1} for {sample_name}")
            sample_name = sample_name[:-len(".wav")] + ".png"
            # Above: if sample is unaugmented to this point, sample_name will be
            # *.wav. Since SoundProcessor.warp_spectrogram expects sample_name to be *.png, we 
            # replace extension. If augmented and sample_name is already *.png, 
            # there is no change.
            warped_name = SoundProcessor.warp_spectrogram(sample_name, 
                                           species_spectrogram_output_dir, 
                                           species_spectrogram_output_dir)
            # if warp is not the only augmentation, 
            # we do not want spectrogram before warp
            if len(aug_choices) != 0: 
                assert(warped_name != sample_name)
                fname = os.path.join(species_spectrogram_output_dir, sample_name)
                os.remove(fname)

    #------------------------------------
    # create_original_spectrograms 
    #-------------------

    def create_original_spectrograms(self, samples, n, species_wav_input_dir, species_spectrogram_output_dir):
        samples = random.sample(samples, int(n)) # choose n from all samples
        for sample_name in samples:
            SoundProcessor.create_spectrograms(sample_name, species_wav_input_dir, species_spectrogram_output_dir, n_mels=128)


# ------------------------ Main ------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Chop audio files into snippets, plus data augmentation."
                                     )
   
    parser.add_argument('-p', '--plot',
                        help='whether or not to plot species distributions; default: False',
                        action='store_true',
                        default=False
                        )
                        
    parser.add_argument('-y', '--overwrite_freely',
                        help='if set, overwrite existing out directories without asking; default: False',
                        action='store_true',
                        default=False
                        )

    parser.add_argument('input_dir_path',
                        help='path to .wav files',
                        default=None)

    args = parser.parse_args()

    if (not os.path.exists(args.input_dir_path)) or \
       (not os.path.isdir(args.input_dir_path)):
        print(f"Wav file directory {args.input_dir_path} does not exist")
        sys.exit(1)

    augmenter = Augmenter(args.input_dir_path,
                          plot=args.plot,
                          overwrite_freely=args.overwrite_freely
                          )

    augmenter.generate_all_augmentations()

#augment_one_species("TANGYR_S", 8, sample_threshold)

# Old:
#   INPUT_DIR_PATH = '../TAKAO_BIRD_WAV_feb20/'




