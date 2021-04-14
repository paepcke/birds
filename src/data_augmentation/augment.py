#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random
import sys

import numpy as np

from logging_service import LoggingService

import data_augmentation.utils as utils

#---------------- Class Augmenter ---------------

class Augmenter:

    ADD_NOISE   = 1/3
    TIME_SHIFT  = 1/3
    WARP        = 1/3
    
    P_DIST    = [ADD_NOISE, TIME_SHIFT, WARP]
    AUG_NAMES = ["add_noise", "time_shift", "warp"]
    AUG_SPECTROGRAMS_DIR = "spectrograms_augmented/"
    AUG_WAV_DIR = "wav_augmented/"
    NOISE_PATH = "data_augmentation/lib/Noise_Recordings/"
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 input_dir_path,
                 plot=False,
                 to_median=True,
                 random_augs = False,
                 multiple_augs = False,):

        '''
        
        @param input_dir_path:
        @type input_dir_path:
        @param plot:
        @type plot:
        @param to_median: True is to make all samples have the median number of samples. 
            False is to make all classes have at least 10% of max samples.
            Define the proportions using each augmentation. Proportion used will be 
            num_in_category / total_of_categories
        @type to_median: bool
        @param random_augs: if this is true, will randomly choose augmentation 
            to use for each new sample
        @type random_augs: bool
        @param multiple_augs: if we want to allow multiple augmentations per sample 
            (ie time shift and warp)):
        @type multiple_augs: bool
        '''

        self.log = LoggingService()
    
        self.input_dir_path = input_dir_path
        self.multiple_augs  = multiple_augs
        
        species_df = utils.sample_compositions_by_species(input_dir_path, augmented=False)
        
        if plot:
            species_df.plot.bar()

        species_np = species_df.values.flatten()
        index_max, index_min = species_df.idxmax().to_numpy()[0], species_df.idxmin().to_numpy()[0]
        
        val_max = np.max(species_np)
        max_tenth = val_max//10 +1
        val_median = np.median(species_np)
        
        self.log.info(f"Median: {val_median},  Min: {np.min(species_np)} ({index_min}) ,  Max: {val_max} ({index_max})")
        self.log.info(f"10% of max is {max_tenth}")

        if random_augs:
            self.output_dir_path = f"{input_dir_path[:-1]}_augmented_samples_random"

        else:
            assert(self.ADD_NOISE + self.TIME_SHIFT + self.WARP == 1)
            self.output_dir_path = f"{input_dir_path[:-1]}_augmented_samples-{self.ADD_NOISE:.2f}n-{self.IME_SHIFT:.2f}ts-{self.WARP:.2f}w"

        if self.multiple_augs:
            self.output_dir_path += "/"
        else:
            self.output_dir_path += "-exc/"  # indicate that augmentations are mutually exclusive

        self.sample_threshold = val_median if to_median else max_tenth

        # Creates output file structure
        # Self.output_dir_path
        #       |-- AUG_WAV_DIR
        #       |         |----
        #       |         |----
        #       |-- AUG_SPECTROGRAMS_DIR
        #       |         |----
        #       |         |----
        utils.create_folder(self.output_dir_path)
        utils.create_folder(os.path.join(self.output_dir_path, self.AUG_WAV_DIR))
        utils.create_folder(os.path.join(self.output_dir_path, self.AUG_SPECTROGRAMS_DIR))

    #------------------------------------
    # create_augmentations
    #-------------------

    def create_augmentations(self, species, num_samples_orig, threshold):

        species_wav_input_dir  = os.path.join(self.input_dir_path, species)
        species_wav_output_dir = os.path.join(self.output_dir_path, self.AUG_WAV_DIR, species)
        species_spectrogram_output_dir = os.path.join(self.output_dir_path, self.AUG_SPECTROGRAMS_DIR, species)
        
        # Make output folders under self.output_dir_path
        # Returns if either folder already exists
        
        if not (utils.create_folder(species_wav_output_dir) and 
                utils.create_folder(species_spectrogram_output_dir)):
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
            self.log.infor(f"Skipping for {species} since there are no original samples.")
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
                                   num_augs_per_sample=num_augs_per_sample
                                   )

    #------------------------------------
    # create_new_sample 
    #-------------------

    def create_new_sample(self, sample_name, paths, num_augs_per_sample=1):

        (species_wav_input_dir, species_wav_output_dir, species_spectrogram_output_dir) = paths
        
        aug_choices = np.random.choice(self.AUG_NAMES, 
                                       size=num_augs_per_sample, 
                                       p=self.P_DIST, 
                                       replace=False)
        # input(f"Aug chioces: {aug_choices}")
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
                #add_noise which noise to add will be chosen at random
                updated_name = aug.add_background(sample_name, NOISE_PATH, species_wav_input_dir, species_wav_output_dir, len_noise_to_add=5.0)
            elif aug_name == "time_shift":
                #time_shift
                updated_name = aug.time_shift(sample_name, species_wav_input_dir, species_wav_output_dir)
            sample_name = updated_name
            
        # create new spectrogram if augmented
        if len(aug_choices) != 0:
            sample_name = aug.create_spectrogram(sample_name, species_wav_output_dir, species_spectrogram_output_dir, n_mels=128)
            
        if warp:
            #warp
            # if len(aug_choices) +1 > 1:
            #     input(f"num_augs = {len(aug_choices) +1} for {sample_name}")
            sample_name = sample_name[:-len(".wav")] + ".png"
            # Above: if sample is unaugmented to this point, sample_name will be
            # *.wav. Since warp_spectrogram expects sample_name to be *.png, we 
            # replace extension. If augmented and sample_name is already *.png, 
            # there is no change.
            warped_name = aug.warp_spectrogram(sample_name, species_spectrogram_output_dir, species_spectrogram_output_dir)
            if len(aug_choices) != 0: # if warp is not the only augmentation, we do not want spectrogram before warp
                assert(warped_name != sample_name)
                os.remove(species_spectrogram_output_dir + sample_name)

#---------------- Class SpectrogramCreator ---------------

class SpectrogramCreator:

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self):
        pass

    #------------------------------------
    # create_original_spectrograms 
    #-------------------

    def create_original_spectrograms(self, samples, n, species_wav_input_dir, species_spectrogram_output_dir):
        samples = random.sample(samples, int(n)) # choose n from all samples
        for sample_name in samples:
            aug.create_spectrogram(sample_name, species_wav_input_dir, species_spectrogram_output_dir, n_mels=128)

#-------------------------
# In[127]:


for species, rows in species_df.iterrows():
    num_samples_orig = rows['num_samples']
    create_augmentations(species, num_samples_orig, sample_threshold)
#     input(f"Finished for {species}")


# In[130]:


os.system(f"find {self.output_dir_path + AUG_SPECTROGRAMS_DIR} -name \".DS_Store\" -delete")
augmented_df = utils.sample_compositions_by_species(self.output_dir_path + AUG_SPECTROGRAMS_DIR, augmented=True)
augmented_df["total_samples"] = augmented_df.sum(axis=1)
print(augmented_df)
augmented_df.plot.bar(y=["add_bg", "time_shift", "mask", "original"],stacked=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))

# ------------------------ Main ------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Chop audio files into snippets, plus data augmentation."
                                     )

    # parser.add_argument('-l', '--errLogFile',
                        # help='fully qualified log file name to which info and error messages \n' +\
                        # 'are directed. Default: stdout.',
                        # dest='errLogFile',
                        # default=None)
    # parser.add_argument('-d', '--dryRun',
                        # help='show what script would do if run normally; no actual downloads \nor other changes are performed.',
                        # action='store_true')
    # parser.add_argument('my_integers',
                        # type=int,
                        # nargs='+',
                        # help='Repeatable: integers. Will show as list in my_integers')
    
    parser.add_argument('-p', '--plot',
                        help='whether or not to plot species distributions; default: False',
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
                          plot=args.plot
                          )

    augmenter.create_augmentations(species, num_samples_orig, threshold)

#create_augmentations("TANGYR_S", 8, sample_threshold)

# Old:
#   INPUT_DIR_PATH = '../TAKAO_BIRD_WAV_feb20/'




