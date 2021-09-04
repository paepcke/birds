#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
import random
import sys, os
import datetime

from logging_service import LoggingService

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import AugmentationGoals, WhenAlreadyDone
from data_augmentation.utils import Utils, AudAugMethod
import multiprocessing as mp
import pandas as pd


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
    
    For the superimposed background noise method of
    audio augmentation, clients control the source of
    sounds to overlay.
    
    Randomness is inserted at every step: choice of 
    augmentation method, and again within those methods.
    For instance, background noise overlays random select
    noise files (unless contrained by client), and also 
    randomize the place in the augmented audio where the
    overlay occurs, as well as the blend multiplier. 
    
    '''

    ADD_NOISE   = 1/3 # Add background noise, such as wind or water
    TIME_SHIFT  = 1/3 # Cut audio at random point into snippets A & B
    VOLUME      = 1/3 #    then create new audio: B-A
    
    NOISE_PATH = os.path.join(os.path.dirname(__file__),'lib')
    
    TASK_QUEUE_SIZE = 50
    ''' Size of queue feeding the augmentation workers'''
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 species_root,
                 overwrite_policy=WhenAlreadyDone.ASK,
                 num_workers=None,
                 species_filter=None,
                 aug_goal=AugmentationGoals.MEDIAN,
                 absolute_seconds=None,
                 aug_methods=None,
                 noise_sources=None,
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
        
        :param species_root: directory holding .wav or .mp3 files
        :type species_root: str
        :param plot: whether or not to plot informative charts 
            along the way
        :type plot: bool
        :param overwrite_policy: if true, don't ask each time
            previously created work will be replaced
        :type overwrite_policy: bool
        :param species_filter, if provided is a list of species
            to involve in the augmentation
        :type species_filter: {None | [str]}
        :param aug_goal: an AugmentationGoals member,
               (See definition of AugmentationGoals; TENTH/MAX/MEDIAN/NUMBER)
        :type aug_goal: AugmentationGoals
        :param absolute_seconds: number of audio seconds required for
            each species. Ignored unless aug_goal is AugmentationGoals.NUMBER
        :type absolute_seconds: {int | float}
        :param unittesting: do no initialization
        :type unittesting: bool
        '''

        self.log = LoggingService()
        
        if unittesting:
            return
        
        if not isinstance(overwrite_policy, WhenAlreadyDone):
            raise TypeError(f"Overwrite policy must be a member of WhenAlreadyDone, not {type(overwrite_policy)}") 

        if not os.path.isabs(species_root):
            raise ValueError(f"Input path must be a full, absolute path; not {species_root}")

        if noise_sources is None:
            noise_sources = AudioAugmenter.NOISE_PATH
        if aug_methods is None:
            aug_methods = ['ADD_NOISE', 'TIME_SHIFT', 'VOLUME']
        
        self.species_root     = species_root
        self.overwrite_policy = overwrite_policy
        self.species_filter   = species_filter
        self.out_dir_root     = os.path.join(Path(species_root).parent, 'audio_augmentations')

        # Determine number of workers:
        num_cores = mp.cpu_count()
        # Use only a percentage of the cores:
        if num_workers is None:
            num_workers = round(num_cores * Utils.MAX_PERC_OF_CORES_TO_USE  / 100)
        elif num_workers > num_cores:
            # Limit pool size to number of cores:
            num_workers = num_cores

        # Get a list of AugmentationTask specifications:
        aug_task_list = self.specify_augmentation_tasks(species_root, 
                                                        aug_goal, 
                                                        absolute_seconds,
                                                        species_filter=species_filter)
        
        # Make sure an outdir exists under self.out_dir_root
        # for each species, and remember the outdirs:
        self.species_out_dirs = {}
        for species in [task.species for task in aug_task_list]:
            this_out_dir = os.path.join(self.out_dir_root, f"{species}")
            self.species_out_dirs[species] = this_out_dir
            os.makedirs(this_out_dir, exist_ok=True)
        
        self.run_jobs(aug_task_list, num_workers)

    #------------------------------------
    # run_jobs
    #-------------------

    def run_jobs(self, task_specs, num_workers):
        '''
        Create processes on multiple CPUs, and feed
        them augmentation tasks. Wait for them to finish.
        
        :param task_specs: list of AugmentationTask instances
        :type task_specs: [AugmentationTask]
        :param num_workers: number of CPUs to use simultaneously.
            Default is 
        :type num_workers: {None | int}
        '''
        
        if len(task_specs) == 0:
            self.log.warn("Audio augmentation task list was empty; nothing done.")
            return
        
        task_queue = mp.Manager().Queue(maxsize=self.TASK_QUEUE_SIZE)
        # For nice progress reports, create a shared
        # dict quantities that each process will report
        # on to the console as progress indications:
        #    o Which job-completions have already been reported
        #      to the console (a list)
        # All Python processes on the various cores
        # will have read/write access:
        manager = mp.Manager()
        global_info = manager.dict()
        global_info['jobs_status'] = manager.list()

        num_tasks  = len(task_specs)
        
        aug_jobs = []

        # Start all the workers:
        for i in range(min(num_tasks, num_workers)):
            ret_value_slot = mp.Value("b", False)
            job = mp.Process(target=self.create_new_sample,
                             args=(task_queue,),
                             name=f"aud_augmenter-{i}"
                             )
            job.ret_val = ret_value_slot
            aug_jobs.append(job)
            print(f"Starting augmenter {job.name}")
            # This job is not yet done:
            global_info['jobs_status'].append(False)
            job.start()

        # Start feeding tasks:
        
        for task in task_specs:
            species = task.species
            task.out_dir = self.species_out_dirs[species]
            task_queue.put(task)

        # Indicate end of tasks in to the queue, one
        # 'STOP' str for each worker to see:
        
        for _i in range(num_workers):
            task_queue.put('STOP')

        # Keep checking on each job, until
        # all are done as indicated by all jobs_done
        # values being True, a.k.a valued 1:
        
        while sum(global_info['jobs_status']) < num_workers:
            for job_idx, job in enumerate(aug_jobs):
                # Timeout 4 sec
                job.join(4)
                if job.exitcode is not None:
                    if global_info['jobs_status'][job_idx]:
                        # One of the processes has already
                        # reported this job as done. Don't
                        # report it again:
                        continue
                        
                    # Let other processes know that this job
                    # is done, and they don't need to report 
                    # that fact: we'll do it here below:
                    global_info['jobs_status'][job_idx] = True
                    
                    # This job finished, and that fact has not
                    # been logged yet to the console:
                    
                    res = "OK" if job.ret_val else "Error"
                    # New line after the single-line progress msgs:
                    print("")
                    print(f"Worker {job.name}/{num_workers} finished with: {res}")
                    #global_info['snips_done'] = self.sign_of_life(job, 
                    #                                              global_info['snips_done'],
                    #                                              outdir,
                    #                                              start_time,
                    #                                              force_rewrite=True)
                    # Check on next job:
                    continue
        
    #------------------------------------
    # create_new_sample 
    #-------------------

    def create_new_sample(self, task_queue):
        '''
        NOTE: This method is run in parallel. It is the
              'target' method for each of the workers
        
        Given one audio recording and an audio augmentation
        method name, compute that augmentation, create a file name
        that gives insight into the aug applied, and write that
        new audio file to out_dir.
        
        Currently available types of audio augmentation technique:
        
            o adding background sounds
            o randomly changing volume
            o random time shifts

        Returns the full path of the newly created audio file:
        
        :param task_queue: inter process communication queue
            from which AugmentationTask objects are read 
        :type multiprocessing.Manager queue
        :return: returns when STOP is read from task_queue. Return
            value is None, or an Exception object if an exception occurred.
        :rtype: {None | Exception}
        '''

        done = False
        failures = None
        
        # Ensure that each subprocess has
        # a different random seed, rather than
        # everyone inheriting from the parent:
        
        random.seed(os.getpid())

        while not done:
            
            task_obj = task_queue.get()
            try:
                if task_obj == 'STOP':
                    done = True
                    continue
                species = task_obj.species 
                sample_path = task_obj.recording_path
                out_dir = self.species_out_dirs[species]
                # Pick a random augmentation method:
                method = random.choice([AudAugMethod.ADD_NOISE,
                                        AudAugMethod.TIME_SHIFT,
                                        AudAugMethod.VOLUME]
                                        )
                
                if method == AudAugMethod.ADD_NOISE:
                    try:
                        _out_path = SoundProcessor.add_background(
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
                        _out_path = SoundProcessor.time_shift(sample_path, out_dir)
                    except Exception as e:
                        sample_fname = Path(sample_path).stem
                        msg = f"Failed to time shift on {sample_fname} ({repr(e)})"
                        self.log.err(msg)
                        e.args = tuple([e.args[0], msg])
                        failures = e
                elif method == AudAugMethod.VOLUME:
                    try:
                        _out_path = SoundProcessor.change_sample_volume(sample_path, out_dir)
                    except Exception as e:
                        sample_fname = Path(sample_path).stem
                        msg = f"Failed to modify volume on {sample_fname} ({repr(e)})"
                        self.log.err(msg)
                        e.args = tuple([e.args[0], msg])
                        failures = e

            finally:
                # Notify system that this task is done
                task_queue.task_done()

        return failures


    #------------------------------------
    # specify_augmentation_tasks
    #-------------------
    
    def specify_augmentation_tasks(self, 
                                   species_root, 
                                   aug_goal=AugmentationGoals.MEDIAN,
                                   absolute_seconds=None,
                                   species_filter=None
                                   ):
        '''
        Returns a list of AugmentationTask instances that
        each specify all that is needed to start a parallel
        task for augmentating one audio file.
        
        The run_jobs() method consumes
        the list of AugmentationTask produced here.
        
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
        :param species_filter: list of species to involve; all
            others under species_root are ignored
        :type species_filter: {None | [str]}
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
                                                                absolute_seconds,
                                                                species_filter=species_filter)

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
                                  absolute_seconds=None,
                                  species_filter=None
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
        :param species_filter: optional list of species to include;
            if none, all species under root_all_species are included
        :type species_filter: {None | [str]}
        :return dict mapping asset instances to their additionally
            required number of recording seconds
        :rtype {SpeciesRecordingAsset : int}
        '''

        if type(aug_goal) != AugmentationGoals:
            raise TypeError(f"aug_goal must be an AugmentationGoals enum member, not {aug_goal}")

        # All species (i.e. subdirectory names under root_all_species):
        species_subdir_list = Utils.listdir_abs(root_all_species)
        if species_filter is not None:
            species_subdirs_to_augment = \
                filter(lambda subdir_name: 
                       Path(subdir_name).stem in species_filter,
                       species_subdir_list 
                       )
        else:
            species_subdirs_to_augment = species_subdir_list

        # Make a dict that maps each species to a
        # SpeciesRecordingAsset instance that contains
        # the duration of each recording of one species.
        
        self.log.info(f"Start recording-time inventory (long operation)")
        start_time = datetime.datetime.now()
        species_assets = {Path(species_dir).stem : SpeciesRecordingAsset(species_dir)
                          for species_dir 
                          in species_subdirs_to_augment
                          }
        end_time = datetime.datetime.now()
        duration_str = Utils.time_delta_str(end_time - start_time)
        self.log.info(f"Finished recording-time inventory ({duration_str})")
        
        if len(species_assets) == 0:
            # If filter choked all species
            # out of the running:
            self.log.warn("No audio augmentation, because no species subdir qualifies")
            return {}
        
        # Sort species_assets by rising number of 
        # recording seconds:

        sorted_species_assets = {species : rec_secs 
                                 for species, rec_secs 
                                 in sorted(species_assets.items(), 
                                           key=lambda pair: pair[1].available_seconds)
                                 }

        assets = list(sorted_species_assets.values())

        if aug_goal == AugmentationGoals.MAX:
            # Maximum available seconds among all species:
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

    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        return f"<AudAugTask {self.species} +{self.duration_added} {hex(id(self))}>"

    #------------------------------------
    # __str__
    #-------------------
    
    def __str__(self):
        return self.__repr__()

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
   
    parser.add_argument('-y', '--overwrite_policy',
                        help='if set, overwrite existing out directories without asking; default: False',
                        action='store_true',
                        default=False
                        )
    
    parser.add_argument('-g', '--goal',
                        help="augmentation goal: 'median', 'max', or seconds as an integer; default is median",
                        default=None
                        )
    
    parser.add_argument('-s', '--species',
                        type=str,
                        nargs='+',
                        help='repeatable: only augment species listed here'
                        )

    parser.add_argument('-n', '--noisesources',
                        type=str,
                        nargs='+',
                        help='repeatable: noise sources to use for blending audio; audio files and directories',
                        default=None
                        )
    
    parser.add_argument('-a', '--augmethods',
                        type=str,
                        choices=['ADD_NOISE', 'TIME_SHIFT', 'VOLUME'],
                        nargs='+',
                        help='repeatable: augmentation methods to use',
                        default=None
                        )

    parser.add_argument('input_dir_path',
                        help='path to .wav files root',
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
        #overwrite_policy = WhenAlreadyDone.ASK
        overwrite_policy = WhenAlreadyDone.SKIP

    absolute_seconds = None
    if args.goal is not None:
        # Must be 'median', 'max', or a number of seconds
        goal = args.goal
        if goal == 'median':
            goal = AugmentationGoals.MEDIAN
        elif goal == 'max':
            goal = AugmentationGoals.MAX
        else:
            if type(goal) != int:
                print(f"The augmentation goal must be 'median', 'max', or a number of seconds, not {goal}")
                sys.exit(1)
            # Got a proper goal of min number of seconds:
            absolute_seconds = goal
            goal = AugmentationGoals.NUMBER
    else:
        # Default:
        goal = AugmentationGoals.MEDIAN
        
    # Noise files for overlays:
    if args.noisesources is not None:
        # Check existence:
        for noise_source in args.noisesources:
            if not os.path.exists(noise_source):
                print(f"Cannot find noise source {noise_source}; doing nothing")
                sys.exit(1)
    noise_sources = args.noise

    augmenter = AudioAugmenter(args.input_dir_path,
                          overwrite_policy=overwrite_policy,
                          aug_goal=goal,
                          absolute_seconds=absolute_seconds,
                          aug_methods=args.augmethods,
                          species_filter=args.species if len(args.species) > 0 else None,
                          noise_sources=noise_sources
                          )
