#!/usr/bin/env python
# coding: utf-8

import argparse
import math
from pathlib import Path
import random
import sys, os
import warnings

import pandas as pd

from logging_service import LoggingService

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import AugmentationGoals, WhenAlreadyDone
from data_augmentation.utils import Utils, AudAugMethod

#---------------- Class AudioAugmenter ---------------

class AudioAugmenter:
    '''
    Process a directory subtree of species recordings
    to create new recordings with modifications. Those
    include adding background noise, volume changes, and
    time shifting.
    
    Starting point is always a directory tree whose
    subdirectories are species names containing recordings
    of that species.
    
              root
                 CCRO
                     recording ccro1
                     recording ccro2
                       ...
                 PATY
                     recording paty1
                     recording paty2
                       ...
                 TRGN
                      ...
                 ...
    
    The species dirs may contain varying numbers of
    recordings. Depending on goal settings, they are all
    brought up to a given level in number of recordings. By
    default, all species are augmented so that at least
    the median number of the original recording number over
    all species is reached.
    
    Command line users can specify the repeatable --species
    command line option to specify particular species
    and number of augmentations.  
    
    '''

    ADD_NOISE   = 1/3 # Add background noise, such as wind or water
    TIME_SHIFT  = 1/3 # Cut audio at random point into snippets A & B
    VOLUME      = 1/3 #    then create new audio: B-A
    
    NOISE_PATH = os.path.join(os.path.dirname(__file__),'lib')
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 input_dir_path,
                 plot=False,
                 overwrite_policy=False,
                 species_filter=None,
                 aug_goals=AugmentationGoals.MEDIAN,
                 random_augs=False,
                 multiple_augs=False,
                 unittesting=False
                 ):

        '''
        If species_filter is provided, it is expected
        to be a dict mapping species names to a number
        of augmentations to perform. Only those species
        will be augmented.
        
        Note: the species_filter feature was added. So
             a number of unnecessary computations are performed
             before the filtering occurs. Could be optimized,
             but time performance is not an issue. 
        
        :param input_dir_path: directory holding .wav files
        :type input_dir_path: str
        :param plot: whether or not to plot informative chars 
            along the way
        :type plot: bool
        :param overwrite_policy: if true, don't ask each time
            previously created work will be replaced
        :type overwrite_policy: bool
        :param species_filter, if provided is a dict mapping
            species names to number of desired augmentations.
        :type species_filter: {None | {str : int} 
        :param aug_goals: either an AugmentationGoals member,
               or a dict with a separate AugmentationGoals
               for each species: {species : AugmentationGoals}
               (See definition of AugmentationGoals; TENTH/MAX/MEDIAN)
        :type aug_goals: {AugmentationGoals | {str : AugmentationGoals}}
        :param random_augs: if this is true, will randomly choose augmentation 
            to use for each new sample
        :type random_augs: bool
        :param multiple_augs: if we want to allow multiple augmentations per sample 
            (e.g. time shift and volume)):
        :type multiple_augs: bool
        :param unittesting: do no initialization
        :type unittesting: bool
        '''

        self.log = LoggingService()
        
        if unittesting:
            return
        
        if not isinstance(overwrite_policy, WhenAlreadyDone):
            raise TypeError(f"Overwrite policy must be a member of WhenAlreadyDone, not {type(overwrite_policy)}") 

        if not os.path.isabs(input_dir_path):
            raise ValueError(f"Input path must be a full, absolute path; not {input_dir_path}")
        
        self.input_dir_path   = input_dir_path
        self.multiple_augs    = multiple_augs
        self.plot             = plot
        self.overwrite_policy = overwrite_policy
        self.species_filter   = species_filter
        
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
        #       num_samples
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
        if random_augs:
            os.path.join(Path(input_dir_path).parent, 'augmented_samples_random')
            self.output_dir_path = os.path.join(Path(input_dir_path).parent, 
                                                'augmented_samples_random')
        else:
            assert(self.ADD_NOISE + self.TIME_SHIFT + self.VOLUME == 1)
            dir_nm = f"Augmented_samples_-{self.ADD_NOISE:.2f}n-{self.TIME_SHIFT:.2f}ts-{self.VOLUME:.2f}w"
            self.output_dir_path = os.path.join(Path(input_dir_path).parent, dir_nm)

        if self.multiple_augs:
            self.output_dir_path += "/"
        else:
            # Indicate that augmentations are mutually exclusive
            self.output_dir_path += "-exc/"  

        self.log.info(f"Results will be in {self.output_dir_path}")

        Utils.create_folder(self.output_dir_path, self.overwrite_policy)

        # Hide the UserWarning: PySoundFile failed. Trying audioread instead.
        warnings.filterwarnings(action="ignore",
                                message="PySoundFile failed. Trying audioread instead.",
                                category=UserWarning, 
                                module='', 
                                lineno=0)


    #------------------------------------
    # generate_all_augmentations
    #-------------------

    def generate_all_augmentations(self):
        '''
        Workhorse:
        Create new samples via augmentation for each species. 
        Augment the audio files to reach the number of audio files
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
        failures = 0
        
        for species, _rows in self.sample_distrib_df.iterrows():
            # For each species, create as many augmentations
            # as was computed earlier, or was provided from
            # the command line in the species filter option:
            
            if self.species_filter is not None:
                if species not in list(self.species_filter.keys()):
                    continue
                num_needed_augs = self.species_filter[species]
            else:
                num_needed_augs = self.augs_to_do[species]

            if num_needed_augs == 0:
                continue
            in_dir = os.path.join(self.input_dir_path, species)
            out_dir = os.path.join(self.output_dir_path, species)
            aug_paths, failures = self.augment_one_species(in_dir,
                                                           out_dir,
                                                           num_needed_augs 
                                                           )
            num_augmentations += len(aug_paths)

        # Clean up directory clutter:
        search_root_dir = os.path.join(self.output_dir_path)
        os.system(f"find {search_root_dir} -name \".DS_Store\" -delete")
        
        self.log.info(f"Total of {num_augmentations} new audio files")
        if failures > 0:
            self.log.info(f"Grant total of audio augmentation failures: {failures}")
        
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
            return []

        # Get dict: {full-path-to-an-audio_file : 0}
        # The zeroes will be counts of augmentations
        # needed for that file:    
        in_wav_files     = {full_in_path : 0
                            for full_in_path
                            in Utils.listdir_abs(in_dir)
                            } 
        # Cannot do augmentations for species with 0 samples
        if len(in_wav_files) == 0:
            self.log.info(f"Skipping for {species_name} since there are no original samples.")
            return []

        # Distribute augmenations across the original
        # input files:
        aug_assigned = 0
        while aug_assigned < num_augs_to_do:
            for fname in in_wav_files.keys():
                in_wav_files[fname] += 1
                aug_assigned += 1
                if aug_assigned >= num_augs_to_do:
                    break
        new_sample_paths = []
        failures = 0

        for in_fname, num_augs_this_file in in_wav_files.items():

            # Create augs with different methods:

            # Pick audio aug methods to apply (without replacement)
            # Note that if more augs are to be applied to each file
            # than methods are available, some methods will need
            # to be applied multiple times; no problem, as each
            # method includes randomness:
            max_methods_sample_size = min(len(list(AudAugMethod)), num_augs_this_file)
            methods = random.sample(list(AudAugMethod), max_methods_sample_size)
            
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

        self.log.info(f"Audio aug report: {len(new_sample_paths)} new files; {failures} failures")
                
        return new_sample_paths, failures

    #------------------------------------
    # create_new_sample 
    #-------------------

    def create_new_sample(self,
                          sample_path,
                          out_dir,
                          method,
                          noise_path=None):
        '''
        Given one audio recording and an audio augmentation
        method name, compute that augmentation, create a file name
        that gives insight into the aug applied, and write that
        new audio file to out_dir.
        
        Currently available types of audio augmentation technique:
        
            o adding background sounds
            o randomly changing volume
            o random time shifts

        Returns the full path of the newly created audio file:
        
        :param sample_path: absolute path to audio sample
        :type sample_path: str
        :param out_dir: destination of resulting new samples
        :type out_dir: src
        :param method: the audio augmentation method to apply
        :type method: AudAugMethod
        :param noise_path: full path to audio files with background
            noises to overlay onto audio (wind, rain, etc.). Ignored
            unless method is AudAugMethod.ADD_NOISE.
        :type noise_path: str
        :return: Newly created audio file (full path) or an Exception
            object whose e.args attribute is a tuple with the error
            msg plus a manually added one 
        :rtype: {str | Exception}
        '''
        
        failures = None
        out_path = None
        if method == AudAugMethod.ADD_NOISE:
            if noise_path is None:
                noise_path = AudioAugmenter.NOISE_PATH
            # Add rain, wind, or such at random:
            try:
                out_path = SoundProcessor.add_background(
                        sample_path,
                        self.NOISE_PATH,
                        out_dir, 
                        len_noise_to_add=5.0)
            except Exception as e:
                sample_fname = Path(sample_path).stem
                msg = f"Failed to add background sounds to {sample_fname} ({repr(e)})"
                self.log.err(msg)
                e.args = tuple([e.args[0], msg])
                failures = e

        elif method == AudAugMethod.TIME_SHIFT:
            try:
                out_path = SoundProcessor.time_shift(sample_path, out_dir)
            except Exception as e:
                sample_fname = Path(sample_path).stem
                msg = f"Failed to time shift on {sample_fname} ({repr(e)})"
                self.log.err(msg)
                e.args = tuple([e.args[0], msg])
                failures = e
        elif method == AudAugMethod.VOLUME:
            try:
                out_path = SoundProcessor.change_sample_volume(sample_path, out_dir)
            except Exception as e:
                sample_fname = Path(sample_path).stem
                msg = f"Failed to modify volume on {sample_fname} ({repr(e)})"
                self.log.err(msg)
                e.args = tuple([e.args[0], msg])
                failures = e

        return out_path if failures is None else failures
    

    #------------------------------------
    # specify_augmentation_tasks
    #-------------------
    
    def specify_augmentation_tasks(self, 
                                   species_root, 
                                   aug_goal=AugmentationGoals.MEDIAN,
                                   absolute_seconds=None
                                   ):
        '''
        Returns a list of AugmentationTask instances that
        each specify all that is needed to start a parallel
        task for audio augmentation.
        
        The augmentations are spread across the available recordings
        of this asset's species until the additional number of seconds 
        needed is reached.
        
        Strategy:
            o Assume that the process of augmenting based on an audio file
              generates an additional audio file of the same length
            o Augment recordings round robin, proceeding shortest 
              to longest recording to generate variety
        
        :param species_root: directory holding one subdirectory
            for each species' recordings
        :type species_root: str
        :param aug_goal: how much to augment all species
        :type aug_goal: AugmentationGoals
        :param absolute_seconds: number of audio seconds required for
            each species. Ignored unless aug_goal is AugmentationGoals.NUMBER
        :type absolute_seconds: {int | float}
        :return list of AugmentationTask instances that can be
            executed in parallel
        :rtype [AugmentationTask]
        '''
        
        # Get dict mapping each species
        # to the number of seconds needed to reach
        # augmentation goal. Keys are SpeciesRecordingAsset
        # that contain information about the recordings
        # available for one species:
        asset_secs_needed_dict = self._required_species_seconds(species_root, 
                                                                aug_goal, 
                                                                absolute_seconds)

        aug_tasks = []
        for asset, secs_needed in asset_secs_needed_dict.items():
            aug_tasks.extend(self._create_aug_tasks_one_species(asset, secs_needed))
        
        return aug_tasks

    #------------------------------------
    # _create_aug_tasks_one_species
    #-------------------
    
    def _create_aug_tasks_one_species(self, asset, seconds_needed):
        '''
        Given one SpeciesRecordingAsset instance,
        return a list of augmentation tasks that will
        bring that species up to the aug_goal.
        
        :param asset: information about the recordings
            of a single species
        :type asset: SpeciesRecordingAsset
        :param seconds_needed: total number of additional 
            recording seconds wanted
        :type seconds_needed: {int | float}
        :return list of AugmentationTask instances to
            run in order to reach the target
        :rtype [AugmentationTask]
        '''

        aug_tasks = []
        secs_left_to_create = seconds_needed
        
        # The asset instance is a dict mapping recording paths 
        # to recording durations in rising order of duration:
        
        done = False
        # Keep going round robin, creating an aug task
        # for each recording in turn, then starting over
        # with the first recording until the needed seconds
        # are covered:
        while not done:
            for record_path, duration in asset.items():
                aug_tasks.append(AugmentationTask(record_path, duration))
                secs_left_to_create -= duration
                
                if secs_left_to_create <= 0:
                    done = True
                    break 
        return aug_tasks

    #------------------------------------
    # _required_species_seconds 
    #-------------------
    
    def _required_species_seconds(self, 
                                  root_all_species, 
                                  aug_goal,
                                  absolute_seconds=None
                                  ):
        '''
        Given the root directory with recordings from
        one species, return a dict mapping full paths 
        of recordings to the number of augmentations that
        need to be created from them.
        
        The aug_goal must be an AugmentationGoals enum element that
        specifies a target for the total number of recording seconds
        to have at the end: MAX means create enough augmentations from
        the different recordings such that species have as many 
        recording seconds as the species with the most seconds. Analogously
        for MIN, MEDIAN, TENTH, and (absolute) NUMBER. If given
        AugmentationGoals.NUMBER, then absolute_seconds must be a 
        target number of seconds.
        
        Returned dict will only include entries for which augmentations
        are required. The dict will map asset instances to required additional
        seconds:
        
                  species_asset --> num_secs
                  
        Asset instances know how to apportion their additional 
        seconds to parallelly executable tasks.

        :param root_all_species: path to recordings of one species
        :type root_all_species: src
        :param aug_goal:
        :type aug_goal:
        :param absolute_seconds: number of audio seconds required for
            each species. Ignored unless aug_goal is AugmentationGoals.NUMBER
        :type absolute_seconds: {int | float}
        :return dict mapping asset instances to their additionally
            required number of recording seconds
        :rtype {SpeciesRecordingAsset : int}
        '''

        if type(aug_goal) != AugmentationGoals:
            raise TypeError(f"aug_goal must be an AugmentationGoals enum member, not {aug_goal}")

        # Make a dict that maps each species to a
        # SpeciesRecordingAsset instance that contains
        # the duration of each recording of one species.
        
        species_assets = {Path(species_dir).stem : SpeciesRecordingAsset(species_dir)
                          for species_dir 
                          in Utils.listdir_abs(root_all_species)
                          }
        
        # Sort species_assets by rising number of 
        # recording seconds:
        
        sorted_species_assets = {species : rec_secs 
                                 for species, rec_secs 
                                 in sorted(species_assets.items(), 
                                           key=lambda pair: pair[1].available_seconds)}

        assets = list(sorted_species_assets.values())
        
        
        if aug_goal == AugmentationGoals.MAX:
            # Maximum avail. seconds among all species:
            target = assets[-1].available_seconds
        elif aug_goal == AugmentationGoals.MEDIAN:
            # Use classmethod to find median across all durations:
            target = SpeciesRecordingAsset.median(assets)
        elif aug_goal == AugmentationGoals.TENTH:
            # One tenth of max:
            target = assets[-1].available_seconds / 10.
        elif aug_goal == AugmentationGoals.NUMBER:
            # A particular goal in number of seconds:
            target = absolute_seconds

        # Compute number of seconds still needed
        # for each species (asset --> secsNeeded):
        additional_secs_needed = {}

        for species, asset in sorted_species_assets.items():

            # If available recording time suffices, done:
            if asset.available_seconds >= target:
                # No augmentation needed for this species.
                # Simply make no entry for it in augmentation_tasks:
                continue

            # How many seconds are missing?
            additional_secs_needed[asset] = target - asset.available_seconds 

        return additional_secs_needed
    
    
# --------------------- SpeciesRecordingAsset Class ------------

class SpeciesRecordingAsset(dict):
    '''
    Instances act like dicts that map a full 
    path to a recording to that recording's duration.
    The dict is ordered by raising duration length.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, species_path):
        '''
        Given the root directory with recordings from
        one species, act as a dict that maps each full recording
        path to the number of seconds that is the duration of
        that recording.
        
        Instances are sortable via the sort/sorted functions.
        Instances are hashable, and can therefore serve as
        keys in a dict. The class implements __eq__.
        
        In addition, retain the total number of recorded seconds
        for the species.
        
        Instances provide the following:
            
            <inst>.species  : name of species for which recording
                              information is enclosed in <inst>.
            <inst>.species_dir : directory holding the samples 
            <inst>.available_seconds : total number of recorded seconds
            <inst>[<recording_path>] --> number of seconds in that recording

        :param species_path: path to recordings of one species
        :type species_path: src
        '''
        
        self.species     = Path(species_path).stem
        self.species_dir = species_path
        self.available_seconds = 0
        # For each recording: its duration:
        rec_len_dict = {}
        for recording in Utils.listdir_abs(species_path):
            # The total_seconds() includes the microseconds,
            # so the value will be a float: 
            duration = SoundProcessor.soundfile_metadata(recording)['duration'].total_seconds()
            self.available_seconds += duration
            rec_len_dict[recording] = duration
            
        # Sort the recording length dict by 
        # length, smallest to largest and make
        # this instance's content be that dict:
        self.update({k : v 
                     for k, v 
                     in sorted(rec_len_dict.items(), key=lambda pair: pair[1])
                     })

    #------------------------------------
    # max
    #-------------------

    @classmethod
    def max(cls, inst_list):
        '''
        From a list of SpeciesRecordingAsset instances,
        return the highest of the available seconds values

        :param inst_list: list of SpeciesRecordingAsset
        :type inst_list: [SpeciesRecordingAsset]
        :return the duration of the species with the
            highest number of recording seconds
        :rtype float
        '''
        longest_dur = max([inst.available_seconds
                           for inst in inst_list
                           ])
        return longest_dur

    #------------------------------------
    # median
    #-------------------

    @classmethod
    def median(cls, inst_list):
        '''
        From a list of SpeciesRecordingAsset instances,
        return the median of the available recording
        seconds from among the  species.

        :param inst_list: list of SpeciesRecordingAsset
        :type inst_list: [SpeciesRecordingAsset]
        :return the median duration of the species
        :rtype float
        '''
    
        durations = [inst.available_seconds
                           for inst in inst_list
                           ]
        median_dur = pd.Series(durations).median()
        return median_dur

    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        duration = round(self.available_seconds, 2)
        return f"<SpeciesRecordingAsset {self.species}:{duration}secs {hex(id(self))}>"

    #------------------------------------
    # __str__
    #-------------------
    
    def __str__(self):
        return(self.__repr__())
    
    
    #------------------------------------
    # __hash__
    #-------------------
    
    def __hash__(self):
        return id(self)
    
    #------------------------------------
    # __eq__ 
    #-------------------
    
    def __eq__(self, other):
        return (self.available_seconds == other.available_seconds and
               self.species_dir == other.species_dir
               )

    #------------------------------------
    # __lt__
    #-------------------
    
    def __lt__(self, other):
        return self.available_seconds < other.available_seconds


# -------------------- Class AugmentationTask ---------------

class AugmentationTask:
    '''
    All information needed to run one augmentation
    on one audio recording.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, recording_path, duration_added):
        
        self.recording_path = recording_path
        self.duration_added = duration_added
        # Species is the name of the given audio file's
        # directory (without leading dirs):
        self.species        = Path(recording_path).parent.stem


# ------------------------ Main ------------

# NOTE: because the --species argument allows for repetition,
#       must add "--" before the input dir. Else get:
#       message that input_dir_path is required, even though
#       it was provided on the cmd line.
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Augment existing set of audio recordings."
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
    
    parser.add_argument('-s', '--species',
                        type=str,
                        nargs='+',
                        help='Repeatable; must be pairs: <species> <num_of_augs>')

    parser.add_argument('input_dir_path',
                        help='path to .wav files',
                        default=None)

    args = parser.parse_args()

    if (not os.path.exists(args.input_dir_path)) or \
       (not os.path.isdir(args.input_dir_path)):
        print(f"Wav file directory {args.input_dir_path} does not exist")
        sys.exit(1)

    # Ensure that the species are pairs like
    #   --species VASE 10 PATY 15
    
    # Build a dict of species name and requested
        # number of augs:
    species_filter = {}
    if args.species is not None:
        species_specs = args.species
        # List length must be even:
        if len(species_specs) % 2 != 0:
            print(f"Species arg must be a series of species/num_augs pairs; given info misses some")
            sys.exit(1)
            
        for i in range(0,len(species_specs),2):
            species  = species_specs[i]
            # Ensure a species of that name is a
            # subdirectory of the root dir:
            if not os.path.exists(os.path.join(args.input_dir_path, species)):
                print(f"Species {species} is not among subdirs of {args.input_dir_path}")
                sys.exit(1)
            try:
                num_augs = int(species_specs[i+1])
            except ValueError:
                print(f"Number of augmentations for {species} given as {species_specs[i+1]}, which is not an int.")
                sys.exit(1)
            
            species_filter[species] = num_augs 

    # Turn the overwrite_policy arg into the
    # proper enum mem
    if args.overwrite_policy:
        overwrite_policy = WhenAlreadyDone.OVERWRITE
    else:
        #overwrite_policy = WhenAlreadyDone.ASK
        overwrite_policy = WhenAlreadyDone.SKIP

    augmenter = AudioAugmenter(args.input_dir_path,
                          plot=args.plot,
                          overwrite_policy=overwrite_policy,
                          species_filter=species_filter if len(species_filter) > 0 else None
                          )

    augmenter.generate_all_augmentations()




