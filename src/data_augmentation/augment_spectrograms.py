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

from logging_service import LoggingService

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import AugmentationGoals, WhenAlreadyDone
from data_augmentation.utils import Utils, ImgAugMethod


#---------------- Class SpectrogramAugmenter ---------------

class SpectrogramAugmenter:

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 species_root,
                 output_dir_path,
                 plot=False,
                 overwrite_policy=False,
                 aug_goal=AugmentationGoals.MEDIAN
                ):

        '''
        
        :param species_root: directory holding .png files
        :type species_root: str
        :param output_dir_path: root of destination dir under
            which each species' subdirectories will be placed.
            Augmentations will be placed in those subdirs.
        :type output_dir_path: str
        :param plot: whether or not to plot informative charts 
            along the way
        :type plot: bool
        :param overwrite_policy: if true, don't ask each time
            previously created work will be replaced
        :type overwrite_policy: bool 
        :param aug_goal: either an AugmentationGoals member,
               or a dict with a separate AugmentationGoals
               for each species: {species : AugmentationGoals}
               (See definition of AugmentationGoals; TENTH/MAX/MEDIAN)
        :type aug_goal: {AugmentationGoals | {str : AugmentationGoals}}
        '''

        self.log = LoggingService()

        if not isinstance(overwrite_policy, WhenAlreadyDone):
            raise TypeError(f"Overwrite policy must be a member of WhenAlreadyDone, not {type(overwrite_policy)}") 

        if not os.path.isabs(species_root):
            raise ValueError(f"Input path must be a full, absolute path; not {species_root}")

        self.input_dir_path   = species_root
        self.output_dir_path  = output_dir_path
        self.plot             = plot
        self.overwrite_policy = overwrite_policy
        
        self.species_names = Utils.find_species_names(self.input_dir_path)

        # Get dataframe with row lables being the
        # species, and one col with number of samples
        # in the respective species:
        #       num_species
        # sp1       10
        # sp2       15
        #      ..

        self.sample_distrib_df = Utils.sample_compositions_by_species(species_root, 
                                                                      augmented=False)
        
        if plot:
            # Plot a distribution:
            self.sample_distrib_df.plot.bar()

        # Build a dict with number of augmentations to do
        # for each species:
        self.augs_to_do = Utils.compute_num_augs_per_species(aug_goal, 
                                                             self.sample_distrib_df)
        
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
        Takes one species, and a number of spectrogram
        augmentations to do. Generates the files,
        and returns a list of the newly created 
        files (full paths).
        
        The maximum number of augmentations created
        depends on the number of spectrogram augmentation 
        methods available (currently 3), and the number
        of spectgrogram files available for the given species:
        
           num-available-spectro-augs * num-of-spectro-files
        
        If num_augs_to_do is higher than the above maximum,
        only that maximum is created. The rest will need to 
        be accomplished by spectrogram augmentation in a 
        different portion of the workflow.

        Augmentations are effectively done round robin across all of
        the species' spectro files such that each file is
        augmented roughly the same number of times until
        num_augs_to_do is accomplished.

        :param in_dir: directory holding one species' spectro files
        :type in_dir: str
        :param out_dir: destination for new spectro files
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
            return []
    

        # Get dict: {full-path-to-an-audio_file : 0}
        # The zeroes will be counts of augmentations
        # needed for that file:    
        in_spectro_files     = {full_in_path : 0
                                for full_in_path
                                in Utils.listdir_abs(in_dir)
                                } 
        # Cannot do augmentations for species with 0 samples
        if len(in_spectro_files) == 0:
            self.log.info(f"Skipping for {species_name} since there are no original samples.")
            return []

        # Distribute augmenations across the original
        # input files:
        aug_assigned = 0
        while aug_assigned < num_augs_to_do:
            for fname in in_spectro_files.keys():
                in_spectro_files[fname] += 1
                aug_assigned += 1
                if aug_assigned >= num_augs_to_do:
                    break

        new_sample_paths = []
        failures = 0

        for in_fname, num_augs_this_file in in_spectro_files.items():

            # Create augs with different methods:

            # Pick audio aug methods to apply (without replacement)
            # Note that if more augs are to be applied to each file
            # than methods are available, some methods will need
            # to be applied multiple times; no problem, as each
            # method includes randomness:
            max_methods_sample_size = min(len(list(ImgAugMethod)), num_augs_this_file)
            methods = random.sample(list(ImgAugMethod), max_methods_sample_size)
            
            # Now have something like:
            #     [volume, time-shift], or all methods: [volume, time-shift, noise]
            
            if num_augs_this_file > len(methods):
                # Repeat the methods as often as
                # needed:
                num_method_set_repeats = int(math.ceil(num_augs_this_file/len(methods)))
                # The slice to num_augs_this_file chops off
                # the possible excess from the array replication: 
                method_seq = (methods * num_method_set_repeats)[:num_augs_this_file]
                
                # Assuming num_augs_per_file is 7, we not have method_seq:
                #    [m1,m2,m3,m1,m2,m3,m1]
            else:
                method_seq = methods
                
            for method in method_seq:
                out_path_or_err = self.create_new_sample(in_fname, out_dir, method)
                if isinstance(out_path_or_err, Exception):
                    failures += 1
                else:
                    new_sample_paths.append(out_path_or_err)

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

        Returns the full path of the newly created spectrogram file:
        
        :param sample_path: absolute path to spectrogram
        :type sample_path: str
        :param out_dir: destination of resulting new spectros
        :type out_dir: src
        :param method: the (spectrogram) image augmentation method to apply
        :type method: ImgAugMethod
        :return: Newly created spectro file (full path) or None,
            if a failure occurred.
        :rtype: {str | None|
        '''
        
        success = False
        spectro, metadata = SoundProcessor.load_spectrogram(sample_path) 
        if method == ImgAugMethod.NOISE:
            try:
                # Default is uniform noise:
                new_spectro, out_fname = SoundProcessor.random_noise(spectro)
                metadata['augmentation'] = 'noise'
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
                metadata['augmentation'] = 'fmask' 
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
                metadata['augmentation'] = 'tmask' 
                success = True
            except Exception as e:
                sample_fname = Path(sample_path).stem                
                self.log.err(f"Failed to time shift on {sample_fname} ({repr(e)})")
            
        if success:
            sample_p = Path(sample_path)
            appended_fname = sample_p.stem + out_fname + sample_p.suffix
            out_path = os.path.join(out_dir, appended_fname)
            SoundProcessor.save_image(new_spectro, out_path, metadata)
        return out_path if success else None
    

# ------------------------ Main ------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Augment existing set of spectrograms"
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
    
    parser.add_argument('-g', '--goal',
                        help='one of median, max, or tenth; how many augmentations relative to most populated class',
                        choices=['median', 'max', 'tenth'],
                        default='median'
                        )

    parser.add_argument('input_dir_path',
                        help='path to .png files',
                        default=None)

    parser.add_argument('output_dir_path',
                        help='destination root directory'
                        )

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
        
    goal_txt = args.goal
    if goal_txt == 'median':
        goal = AugmentationGoals.MEDIAN
    elif goal_txt == 'tenth':
        goal = AugmentationGoals.TENTH
    else:
        goal = AugmentationGoals.MAX

    augmenter = SpectrogramAugmenter(args.input_dir_path,
                                     args.output_dir_path,
                                     plot=args.plot,
                                     overwrite_policy=overwrite_policy,
                                     aug_goals=goal
                                     )

    augmenter.generate_all_augmentations()
