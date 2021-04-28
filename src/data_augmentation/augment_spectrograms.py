#!/usr/bin/env python
# coding: utf-8

# TODO:
#    - This is a rough copy of augment_audio, and
#      needs to be modified to be for spectrograms.

import argparse
import math
from pathlib import Path
import random
import sys, os
import warnings

from logging_service import LoggingService

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import AugmentationGoals, WhenAlreadyDone
from data_augmentation.utils import Utils, ImgAugMethod


#---------------- Class AudioAugmenter ---------------
class SpectrogramAugmenter:

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 input_dir_path,
                 plot=False,
                 overwrite_policy=False,
                 aug_goals=AugmentationGoals.MEDIAN
                ):

        '''
        
        :param input_dir_path: directory holding .png files
        :type input_dir_path: str
        :param plot: whether or not to plot informative charts 
            along the way
        :type plot: bool
        :param overwrite_policy: if true, don't ask each time
            previously created work will be replaced
        :type overwrite_policy: bool 
        :param aug_goals: either an AugmentationGoals member,
               or a dict with a separate AugmentationGoals
               for each species: {species : AugmentationGoals}
               (See definition of AugmentationGoals; TENTH/MAX/MEDIAN)
        :type aug_goals: {AugmentationGoals | {str : AugmentationGoals}}
        '''

        self.log = LoggingService()

        if not isinstance(overwrite_policy, WhenAlreadyDone):
            raise TypeError(f"Overwrite policy must be a member of WhenAlreadyDone, not {type(overwrite_policy)}") 

        if not os.path.isabs(input_dir_path):
            raise ValueError(f"Input path must be a full, absolute path; not {input_dir_path}")

        self.input_dir_path   = input_dir_path
        self.plot             = plot
        self.overwrite_policy = overwrite_policy
        
        self.species_names = Utils.find_species_names(self.input_dir_path)

        # If aug_goals is not a dict mapping
        # each species to an aug_goals, but just
        # a single AugmentationGoals, create
        # a dict from all bird species, mapping
        # each to that same value:
        
        if type(aug_goals) != dict:
            aug_goals = {species : aug_goals
                          for species in self.species_names
                          }

        # Get dataframe with row lables being the
        # species, and one col with number of samples
        # in the respective species:
        #       num_species
        # sp1       10
        # sp2       15
        #      ..

        self.sample_distrib_df = Utils.sample_compositions_by_species(input_dir_path, 
                                                                      augmented=False)
        
        if plot:
            # Plot a distribution:
            self.sample_distrib_df.plot.bar()

        # Build a dict with number of augmentations to do
        # for each species:
        self.augs_to_do = Utils.compute_num_augs_per_species(aug_goals, 
                                                             self.sample_distrib_df)
        
        # Get input dir path without trailing slash:
#****        canonical_in_path = str(Path(input_dir_path))
        # Create the descriptive name of an output directory 
        # for the augmented samples: 
        dir_nm = f"AugmentedSpectros"
        self.output_dir_path = os.path.join(Path(input_dir_path).parent, dir_nm)
        self.log.info(f"Results will be in {self.output_dir_path}")

        Utils.create_folder(self.output_dir_path, self.overwrite_policy)

    #------------------------------------
    # generate_all_augmentations
    #-------------------

    def generate_all_augmentations(self):
        '''
        Workhorse:
        Create new samples via augmentation for each spectrogram. 
        Augment the spectro files to reach the number of spectro files
        indicated in the self.aug_requirements.
        
        Assumption: self.aug_requirements is a dict mapping 
        species-name : num_required_augmentations
        
        Assumption: self.sample_distrib_df is a dataframe like
        
        	        num_species
        	  sp1       10
        	  sp2       15
        	       ...
        	  
        '''
        num_augmentations = 0
        
        for species, _rows in self.sample_distrib_df.iterrows():
            # For each spectrogram, create as many augmentations
            # as was computed earlier:
            num_needed_augs = self.augs_to_do[species]
            if num_needed_augs == 0:
                continue
            in_dir = os.path.join(self.input_dir_path, species)
            out_dir = os.path.join(self.output_dir_path, species)
            aug_paths = self.augment_one_species(in_dir,
                                                 out_dir,
                                                 num_needed_augs 
                                                 )
            num_augmentations += len(aug_paths)

        # Clean up directory clutter:
        search_root_dir = os.path.join(self.output_dir_path)
        os.system(f"find {search_root_dir} -name \".DS_Store\" -delete")
        
        self.log.info(f"Total of {num_augmentations} new spectrogam files")
        
        self.log.info("Done")
        
    #------------------------------------
    # augment_one_species
    #-------------------

    def augment_one_species(self, in_dir, out_dir, num_augs_to_do):
        '''
        Takes one species, and a number of audio
        augmentations to do. Generates the files,
        and returns a list of the newly created 
        files (full paths).
        
        The maximum number of augmentations created
        depends on the number of audio augmentation 
        methods available (currently 3), and the number
        of audio files available for the given species:
        
           num-available-audio-augs * num-of-audio-files
        
        If num_augs_to_do is higher than the above maximum,
        only that maximum is created. The rest will need to 
        be accomplished by spectrogram augmentation in a 
        different portion of the workflow.

        Augmentations are effectively done round robin across all of
        the species' audio files such that each file is
        augmented roughly the same number of times until
        num_augs_to_do is accomplished.

        :param in_dir: directory holding one species' audio files
        :type in_dir: str
        :param out_dir: destination for new audio files
        :type out_dir: src
        :param num_augs_to_do: number of augmentations
        :type num_augs_to_do: int
        :returns: list of newly created file paths
        :rtype: [src]
        '''
        
        # By convention, species name is the last part of the directory:
        species_name = Path(in_dir).stem
        
        # Create subfolder for the given species:
        if not Utils.create_folder(out_dir, self.overwrite_policy):
            self.log.info(f"Skipping augmentations for {species_name}")
            return
    
        in_spectro_files     = [os.path.join(in_dir, fname) 
                            for fname 
                            in os.listdir(in_dir)
                            ]
        num_aud_files = len(in_spectro_files)

        # Cannot do augmentations for species with 0 samples
        if num_aud_files == 0:
            self.log.info(f"Skipping for {species_name} since there are no original samples.")
            return
        
        # How many augs per file?
        num_augs_per_file = num_augs_to_do // num_aud_files
        # Num of augs left to do after we will have
        # applied an even number of augs to each sample:
        remainder = num_augs_to_do - num_augs_per_file * num_aud_files
        new_sample_paths = []

        failures = 0

        for sample_path in in_spectro_files:
            # Create num_augs_per_file samples with
            # different methods:
            
            # Pick spectro aug methods to apply (without replacement)
            # Note that if more augs are to be applied to each file
            # than methods are available, some methods will need
            # to be applied multiple times; no problem, as each
            # method includes randomness:
            method_list = list(ImgAugMethod)
            methods = random.sample(method_list, 
                                    min(num_augs_per_file, len(method_list))
                                    )
            
            # Now have something like:
            #     [noise, fmask], or all methods: [noise, fmask, tmask]
            
            if num_augs_per_file > len(method_list):
                # Repeat the methods as often as
                # needed:
                num_method_set_repeats = int(math.ceil(num_augs_per_file/len(methods)))
                # The slice to num_augs_per_file chops off
                # the possible excess from the array replication: 
                method_seq = (methods * num_method_set_repeats)[:num_augs_per_file]
                
                # Assuming num_augs_per_file is 7, we not have method_seq:
                #    [m1,m2,m3,m1,m2,m3,m1]
            else:
                method_seq = methods
                
            for method in method_seq:
                out_path = self.create_new_sample(sample_path, out_dir, method)
                if out_path is not None:
                    new_sample_paths.append(out_path)
                else:
                    failures += 1

        # Take care of the remainders: the number
        # of augmentations to be done that were not
        # covered in the above loop. We apply additional
        # methods to a few of the files:

        for i in range(remainder):
            sample_path = in_spectro_files[i]
            method = random.sample(list(ImgAugMethod), 1)
            out_path = self.create_new_sample(sample_path, out_dir, method)
            if out_path is not None:
                new_sample_paths.append(out_path)
            else:
                failures += 1

        self.log.info(f"Spectrogram aug report: {len(new_sample_paths)} new files; {failures} failures")
        return new_sample_paths

    #------------------------------------
    # create_new_sample 
    #-------------------

    def create_new_sample(self,
                          sample_path,
                          out_dir,
                          method
                          ):
        '''
        Given one spectrogram file, and an image augmentation
        method name, compute that augmentation, create a file name
        that gives insight into the aug applied, and write that
        new spectrogram file to out_dir.
        
        Currently available types of image augmentation technique:
        
            o adding random or uniform sounds
            o frequency masking
            o time masking

        Returns the full path of the newly created audio file:
        
        :param sample_path: absolute path to audio sample
        :type sample_path: str
        :param out_dir: destination of resulting new samples
        :type out_dir: src
        :param method: the audio augmentation method to apply
        :type method: ImgAugMethod
        :return: Newly created audio file (full path) or None,
            if a failure occurred.
        :rtype: {str | None|
        '''
        
        success = False
        spectro = SoundProcessor.load_spectrogram(sample_path) 
        if method == ImgAugMethod.NOISE:
            try:
                # Default is uniform noise:
                new_spectro, out_fname = SoundProcessor.random_noise(spectro)
                success = True
            except Exception as e:
                sample_fname = Path(sample_path).stem
                self.log.err(f"Failed to add noise to {sample_fname} ({repr(e)})")

        elif method == ImgAugMethod.FMASK:
            try:
                # Horizontal bands:
                new_spectro, out_fname = SoundProcessor.freq_mask(spectro, 
                                                                  max_height=15 # num freq bands
                                                                  ) 
                success = True
            except Exception as e:
                sample_fname = Path(sample_path).stem                
                self.log.err(f"Failed to time shift on {sample_fname} ({repr(e)})")
                
        elif method == ImgAugMethod.TMASK:
            try:
                # Vertical bands:
                new_spectro, out_fname = SoundProcessor.time_mask(spectro, 
                                                                  max_width=15 # num time ticks
                                                                  ) 
                success = True
            except Exception as e:
                sample_fname = Path(sample_path).stem                
                self.log.err(f"Failed to time shift on {sample_fname} ({repr(e)})")
            
        if success:
            sample_p = Path(sample_path)
            appended_fname = sample_p.stem + out_fname + sample_p.suffix
            out_path = os.path.join(out_dir, appended_fname)
            SoundProcessor.save_img_array(new_spectro, out_path)
        return out_path if success else None
    

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
                        
    parser.add_argument('-y', '--overwrite_policy',
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

    # Turn the overwrite_policy arg into the
    # proper enum mem
    if args.overwrite_policy:
        overwrite_policy = WhenAlreadyDone.OVERWRITE
    else:
        overwrite_policy = WhenAlreadyDone.ASK

    augmenter = SpectrogramAugmenter(args.input_dir_path,
                                     plot=args.plot,
                                     overwrite_policy=overwrite_policy
                                     )

    augmenter.generate_all_augmentations()
