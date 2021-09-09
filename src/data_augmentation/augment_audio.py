#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
from pathlib import Path
import random
import sys, os

from logging_service import LoggingService

from data_augmentation.recording_length_inventory import RecordingsInventory
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
                 out_dir,
                 overwrite_policy=WhenAlreadyDone.ASK,
                 num_workers=None,
                 species_filter=None,
                 aug_goal=AugmentationGoals.MEDIAN,
                 absolute_seconds=None,
                 aug_methods=None,
                 noise_sources=None,
                 recording_inventory=None,
                 unittesting=False
                 ):

        '''
        
        The num_workers controls how many cores will be employed
        for the augmentation. If the given number is higher than the
        available cores, only as many processes as there are cores
        are created at a time. If None, then Utils.MAX_PERC_OF_CORES_TO_USE
        percentage of available cores is used.

        If species_filter is provided, it is expected
        to be a dict mapping species names to a number
        of augmentations to perform. Only those species
        will be augmented.
        
        The aug_goal specifies how many augmentations to create
        within each species. See documentation of AugmentationGoals in
        utils.py. If aug_goal is AugmentationGoals.NUMBER, then
        absolute_seconds must provide the minimal total number of seconds worth
        of recordings that every species must contain after augmentation. 
        
        The aug_methods argument may be any combination of AudAugMethod.ADD_NOISE,
        AudAugMethod.TIME_SHIFT, and AudAugMethod.VOLUME.
        
        The noise_sources argument points to audio recordings to use for
        the ADD_NOISE method. Random selections of the recordings will be used to 
        blend onto a given sample that is to be augmented. The argument
        may be an individual sound file, a directory of such sound files,
        or a list with combinations of the two. Default is AudioAugmenter.NOISE_PATH.
        
        The recording_inventory is an optional path to a previously saved
        json inventory of recorded seconds available for each species. The file is 
        assumed to be a json-formated saved dataframe with two columns:
        'total_recoring_length (secs)', and 'duration (hrs:mins_secs)'.
        Rows are one per species. Such a file is usually produced
        by recording_length_inventory(), and saved as manifest.json.
        
        :param species_root: directory holding .wav or .mp3 files
        :type species_root: str
        :param out_dir: augmentation directory destination; is
            created if not exists
        :type out_dir: str
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
        :param recording_inventory: optional path to a previously saved
            inventory of recording seconds for each species. Usually produced
            by recording_length_inventory(), and saved as manifest.json
        :type recording_inventory: str
        :param unittesting: do no initialization
        :type unittesting: bool
        '''

        self.log = LoggingService()
        
        self.recording_inventory = recording_inventory

        if not isinstance(overwrite_policy, WhenAlreadyDone):
            raise TypeError(f"Overwrite policy must be a member of WhenAlreadyDone, not {type(overwrite_policy)}") 

        if not os.path.isabs(species_root):
            raise ValueError(f"Input path must be a full, absolute path; not {species_root}")

        if noise_sources is None:
            self.noise_sources = AudioAugmenter.NOISE_PATH
        else:
            self.noise_sources = noise_sources

        # Determine number of workers:
        num_cores = mp.cpu_count()
        # Use only a percentage of the cores:
        if num_workers is None:
            self.num_workers = round(num_cores * Utils.MAX_PERC_OF_CORES_TO_USE  / 100)
        elif num_workers > num_cores:
            # Limit pool size to number of cores:
            self.num_workers = num_cores
        else:
            self.num_workers = num_workers

        if unittesting:
            return

        # Process requested augmentation 
        # mehods; any specified?
        if aug_methods is None:
            # No, use all three:
            self.augmethods = [AudAugMethod.ADD_NOISE,
                               AudAugMethod.TIME_SHIFT,
                               AudAugMethod.VOLUME
                               ]
        else:
            # Yes, a list of methods was
            # specified. 
            self.augmethods = aug_methods

        self.species_root     = species_root
        self.overwrite_policy = overwrite_policy
        self.species_filter   = species_filter
        self.out_dir_root     = out_dir

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
        
        self.run_jobs(aug_task_list)

    #------------------------------------
    # run_jobs
    #-------------------

    def run_jobs(self, task_specs):
        '''
        Create processes on multiple CPUs, and feed
        them augmentation tasks. Wait for them to finish.
        
        Assumption: self.num_workers holds number of CPUs to use
        
        :param task_specs: list of AugmentationTask instances
        :type task_specs: [AugmentationTask]
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
        for i in range(min(num_tasks, self.num_workers)):
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
        
        for _i in range(self.num_workers):
            task_queue.put('STOP')

        # Keep checking on each job, until
        # all are done as indicated by all jobs_done
        # values being True, a.k.a valued 1:
        
        while sum(global_info['jobs_status']) < self.num_workers:
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
                    print(f"Worker {job.name}/{self.num_workers} finished with: {res}")
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
                method = random.choice(self.augmethods)
                
                if method == AudAugMethod.ADD_NOISE:
                    try:
                        _out_path = SoundProcessor.add_background(
                                sample_path,
                                self.noise_sources,
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
        species_subdir_list = [one_dir
                               for one_dir
                               in Utils.listdir_abs(root_all_species)
                               if os.path.isdir(one_dir)]
        
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

        if self.recording_inventory is not None:
            # Great: save ourselves time loading metadata of
            # every recording, and adding the time durations:
                # Get like:
                #        total_recording_length (secs) duration (hrs:mins:secs)
                # BAFFG                          16216                  4:30:16
                # BANAG                           7309                  2:01:49
                # BBFLG                            974                  0:16:14
                #             ...     
            self.inventory_df = pd.io.json.read_json(self.recording_inventory)
            # Make an uninitialized list of asset instances:
            species_assets = [SpeciesRecordingAsset(species_path)
                              for species_path
                              in species_subdirs_to_augment
                              ]
            # Set the sum of recording seconds for each species,
            # which we know from the inventorty_df:
            for species_asset in species_assets:
                species_asset.available_seconds = self.inventory_df.loc[species_asset.species,
                                                                        'total_recording_length (secs)' 
                                                                        ]
            # The assets from the stored inventory
            # are already sorted:
            sorted_species_assets = species_assets
        else:
            # Create inventory of already available recording seconds:
            
            self.log.info(f"Start recording-time inventory (long operation)")
            start_time = datetime.datetime.now()
            inventory_readme_msg = f"Inventory for species under {root_all_species} ({Utils.datetime_str()})"
            self.inventory_df = RecordingsInventory(root_all_species, 
                                                    message=inventory_readme_msg, 
                                                    chart_result=True,
                                                    print_results=False,
                                                    num_workers=self.num_workers
                                                    ).inventory
            end_time = datetime.datetime.now()
            duration_str = Utils.time_delta_str(end_time - start_time)
            self.log.info(f"Finished recording-time inventory ({duration_str})")
            
            # Using that inventory, create a dict of 
            # recording assets, keyed from the species names:
            species_assets = {}
            for species_dir in species_subdirs_to_augment:
                species  = Path(species_dir).stem
                recordings_duration = self.inventory_df.loc[species, 'total_recording_length (secs)']
                asset   = SpeciesRecordingAsset(species_dir, 
                                                recordings_duration=recordings_duration)
                species_assets[species] = asset

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

class SpeciesRecordingAsset:
    '''
    Instances act like dicts that map a full 
    path to a recording to that recording's duration.
    The dict is ordered by raising duration length.
    
    Also available is: <inst>.available_seconds, which
    is the sum of all of this species' recordings' durations.
    
    Since determining recording durations is expensive,
    all duration lookups are done lazily.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, 
                 species_path,
                 recordings_duration=None
                 ):
        '''
        Given the root directory with recordings from
        one species, act as a dict that maps each full recording
        path to the number of seconds that is the duration of
        that recording. The reading of the number of seconds
        is lazy, done only when __getitem__() is called.

        In addition, retain the total number of recorded seconds
        for the species.
        
        Instances are sortable the total of recorded durations 
        via the sort/sorted functions.
        
        Instances are hashable, and can therefore serve as
        keys in a dict. The class implements __eq__.
        
        
        Instances provide the following:
            
            <inst>.species  : name of species for which recording
                              information is enclosed in <inst>.
            <inst>.species_dir : directory holding the samples 
            <inst>.available_seconds : total number of recorded seconds
            <inst>[<recording_path>] --> number of seconds in that recording

        :param species_path: path to recordings of one species
        :type species_path: str
        :param recordings_duration: if already known, the sum of 
            all recording durations of the species
        :type recordings_duration: {None | int | float}
        '''

        self.content = {}
        self.species     = Path(species_path).stem
        self.species_dir = species_path
        # Total number of seconds of recordings
        # available for this species
        self._available_seconds = recordings_duration
        
        # For each recording: its duration placeholder:
        for recording in Utils.listdir_abs(species_path):
            self.content[recording] = None
        # No need to sort yet if values()/items()/keys()
        # methods are called:
        self.must_sort = False

    #------------------------------------
    # available_seconds
    #-------------------
    
    @property
    def available_seconds(self):
        if self._available_seconds is None:
            # Initialize sum of recording durations:
            self.read_recording_lengths()
        else:
            return self._available_seconds
    
    @available_seconds.setter
    def available_seconds(self, new_val):
        self._available_seconds = new_val

    #------------------------------------
    # __getitem__
    #-------------------
    
    def __getitem__(self, sample_path):
        if self.content[sample_path] is None:
            # Have not computed every recording's 
            # duration yet. So, need to read the
            # requested recording's duration now:
            self.read_recording_lengths(sample_path)
        return self.content[sample_path]

    #------------------------------------
    # __setitem__
    #-------------------
    
    def __setitem__(self):
        raise NotImplementedError("Class SpeciesRecordingAsset's individual recordings' durations are read-only")

    #------------------------------------
    # keys
    #-------------------
    
    def keys(self):
        if self.must_sort:
            self.sort()
        return self.content.keys()

    #------------------------------------
    # values
    #-------------------
    
    def values(self):
        '''
        Implements the dict values() method in this
        lazy environment. Fills all recording durations
        that are still None, then sorts by increasing
        duration, and returns the usual dict-values iterable.
        
        :returns: a value iterator
        :rtype: iter({int | float})
        '''
        # Fill in all duration values that are
        # still null b/c they were never requested:
        
        self.read_recording_lengths()
        if self.must_sort:
            self.sort()

        return self.content.values()

    #------------------------------------
    # items
    #-------------------
    
    def items(self):
        '''
        Implements the dict items() method in this
        lazy environment. Fills all recording durations
        that are still None, then sorts by increasing
        duration, and returns the usual dict-items iterable.
        
        :returns: a key/value iterator
        :rtype: iter((str, {int | float}))
        '''
        
        # Fill in all duration values that are
        # still null b/c they were never requested:
        
        self.read_recording_lengths()
        if self.must_sort:
            self.sort()

        return self.content.items()

    #------------------------------------
    # read_recording_lengths
    #-------------------

    def read_recording_lengths(self, path=None):
        '''
        Update the self.content dict (audio-path : recording-duration).
        If path is None, updates all durations of this instance's
        species that are not yet filled in. Else only updates
        the recording length of the given path.
        
        Sets self.must_sort to True.
        
        :param path: if provided, only recording length
            of given path is updated in self.content dict.
            Else all paths in the species dir are read and
            updated.
        :type path: str
        '''
        
        # Start with the cache of recording
        # durations we already have: self.content
        if path is None:
            paths_to_examine = Utils.listdir_abs(self.species_dir)
        else:
            paths_to_examine = [path]
        for recording in paths_to_examine:
            # Already looked this one up?
            try:
                dur = self.content[recording]
            except KeyError:
                # Recording appeared after this instance
                # was created; unusual, but OK:
                dur = None
                
            if dur is None: 
                # The total_seconds() includes the microseconds,
                # so the value will be a float: 
                duration = SoundProcessor.soundfile_metadata(recording)['duration'].total_seconds()
                self.content[recording] = duration
                
                # Sorting is expensive, setting a var
                # multiple times is cheap, so do the latter:
                # we actually did make a change:
                self.must_sort = True

        if self.must_sort:
            # Made a change, so update the sum or recorded
            # sessions:
            self._available_durations = sum([duration
                                             for duration 
                                             in self.content.values()
                                             if duration is not None
                                             ])
        
    #------------------------------------
    # sort
    #-------------------
    
    def sort(self):
        '''
        Sort the self.content dict by rising
        recording durations (i.e. by rising values).
        Have all the entries with values equal None
        at the end.
        
        Sets self.must_sort to False
        '''
        
        # Sort the dict by increasing recording duration:
        
        # Make a tmp dict with just the 
        # values that are already set, i.e. are
        # not still None:
        
        non_nones_dict = {k : v
                          for k,v
                          in self.content.items()
                          if v is not None
                          }
        
        # Keys (i.e. path to recordings) of
        # entries whose values are still None:
        nones_entries  = [path 
                          for path 
                          in self.content.keys()
                          if self.content[path] is None
                          ]

        # Replace the self.content dict with the
        # sorted non_nones_dict:
        self.content = {k : v
                        for k,v
                        in sorted(non_nones_dict.items(), 
                                  key=lambda item_pair: item_pair[1])
                        }
        
        # Now add the still-None valued entries
        # at the end:
        for path in nones_entries:
            self.content[path] = None

        # We did the sort:
        self.must_sort = False

        
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
    
    cpu_percentage = Utils.MAX_PERC_OF_CORES_TO_USE
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Augment existing set of audio recordings."
                                     )
   
    parser.add_argument('-y', '--overwrite_policy',
                        help='if set, overwrite existing out directories without asking; default: False',
                        action='store_true',
                        default=False
                        )

    parser.add_argument('-w', '--workers',
                        type=int,
                        help=f"maximum number of CPUs to use; default is {cpu_percentage} percent of available",
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
    
    parser.add_argument('-i', '--inventory',
                        help='path to .json file of recording seconds available for each species',
                        default=None)

    parser.add_argument('input_dir_path',
                        help='path to .wav files root',
                        default=None)

    parser.add_argument('output_dir_path',
                        help='directory where results are to be placed; created if not exists'
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
            # Should be an int:
            try:
                goal = int(goal)
            except ValueError:
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
    noise_sources = args.noisesources
    
    if args.augmethods is not None:
        # Convert the CLI 
        # words NOISE, TIME_SHIFT, and VOLUME
        # into AudAugMethod enum members, if
        # needed:
        augmethods = []
        if 'NOISE' in args.augmethods:
            augmethods.append(AudAugMethod.ADD_NOISE)
        if 'TIME_SHIFT' in args.augmethods:
            augmethods.append(AudAugMethod.TIME_SHIFT)
        if 'VOLUME' in args.augmethods:
            augmethods.append(AudAugMethod.VOLUME)
            
    else:
        augmethods = None
        
    if args.inventory is not None:
        inventory = Path(args.inventory)
        if not inventory.is_file() or not inventory.suffix == '.json':
            print(f"Recording inventory must be a .json file, not '{args.inventory}'")
            sys.exit(1)
        

    augmenter = AudioAugmenter(args.input_dir_path,
                               args.output_dir_path,
                               overwrite_policy=overwrite_policy,
                               num_workers=args.workers,
                               aug_goal=goal,
                               absolute_seconds=absolute_seconds,
                               aug_methods=augmethods,
                               species_filter=args.species,
                               noise_sources=noise_sources,
                               recording_inventory=args.inventory
                               )
