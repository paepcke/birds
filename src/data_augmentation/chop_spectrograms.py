#!/usr/bin/env python3

from _collections import OrderedDict
import argparse
import datetime
import os, sys
from pathlib import Path

from logging_service.logging_service import LoggingService

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import ProcessWithoutWarnings
from data_augmentation.utils import Utils
from data_augmentation.utils import WhenAlreadyDone
import multiprocessing as mp
import numpy as np


# Needed when running headless:
# Do this before any other matplotlib
# imports; directly or indirectly 
# through librosa
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class SpectrogramChopper:
    '''
    Processes directories of .png files,
    chopping them into window_len seconds snippets.

    Assumes:

                        self.input
                        
          Species1        Species2   ...     Speciesn
        spectro1_1.png     spectro2_1.png     spectro_n_1.png
        spectro1_2.png     spectro2_2.png     spectro_n_2.png
                            ...
                            
    Saves the snippets in a new directory.
    
        Resulting directories under self.out_dir will be:
         
                         self.out_dir
          Species1        Species2   ...     Speciesn
         snip_1_1_1         snip_2_1_1        snip_n_1_1
         snip_1_1_2         snip_2_1_2        snip_n_1_2
         snip_1_1_3         snip_2_1_3        snip_n_1_3
                            snip_2_1_4
            ...
         snip_1_2_1         snip_2_2_1        snip_n_2_1
         snip_1_2_2         snip_2_2_2        snip_n_2_2
         snip_1_2_3                           snip_n_2_3
         snip_1_2_4

    With snip_b_f_s: 
       o b is the bird species (manifesting in the file system
                                  as one subdirectory under self.out_dir)
       o f is one spectrogram of a full audio recording
       o s is a snippet number.
         
    Note the possibility of different numbers of snippets
    in each species, and for each original audio recording 
    (which may be of unequal lengths).

    Because many spectrograms are created, speed requirements
    call for the use of parallelism. Since each audio file's processing
    is independent from the others, the multiprocessing library
    is used as follows.
    
        - If command line arg --workers is set to 1, no parallelism
          is used. 
        - If multiple cores are available, some percentage of 
          of them will be deployed to chopping. Each core runs 
          a separate copy of this file. The percentage is controlled
          by MAX_PERC_OF_CORES_TO_USE.
        
    Method chop_all() is used in the single core scenario.
    Method chop_from_file_list() is used when multiprocessing. This
    method is the 'target' in the multiprocessing library's sense.

    '''

    # Common logger for all workers:
    log = LoggingService()
    
    MIN_SNIPPET_WIDTH=256
    '''Minimum width of spectrogram snippets to satisfy the 
       torchvision pretrained model minimum value of 
       224 for both width and height'''

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, 
                 in_dir_or_spectro_file, 
                 outdir, 
                 specific_species=None,
                 overwrite_policy=WhenAlreadyDone.ASK
                 ):
        '''
        The overwrite_policy is one of the WhenAlreadyDone
        enum members: ASK, OVERWRITE, SKIP. If ASK,
        request user's permission for each encountered
        destination file. SKIP should be used when resuming
        an interrupted chopping session. Any sound file
        whose destination spectrogram exists is not processed
        again.
        
        @param in_dir_or_spectro_file: location of spectrogram root
        @type in_dir_or_spectro_file: str
        @param outdir: root of spectrograms to create
        @type outdir: src
        @param specific_species: process only a spectific list of species
        @type specific_species: {None | [str]}
        @param overwrite_policy: what to do when an output file already exists
        @type overwrite_policy: WhenAlreadyDone
        '''

        # Ensure the outdir and all its intermediate dirs exist:
        os.makedirs(outdir, exist_ok=True)

        self.in_dir = in_dir_or_spectro_file if os.path.isdir(in_dir_or_spectro_file) else None
        self.out_dir        	= outdir
        self.specific_species   = specific_species
        self.overwrite_policy   = overwrite_policy
        
        self.num_chopped = 0

        # The following used to be less convoluted until
        # the option to chop just a single spectro, rather
        # than a list of subdirectories with spectros 
        # in them. Sorry! :-(
         
        if self.in_dir is not None:
            # We are to process entire directory tree,
            # not just a single file:
            if self.specific_species is None:
                # Process all (species) subdirectories:
                self.species_list = os.listdir(self.in_dir)
            else:
                # Only process certain species:
                self.species_list = self.specific_species
            # Create destination directories for new spectrogram 
            # snippets, so that the dest tree will mirror the in tree:
            self.spectrogram_dir_path = self.create_dest_dirs(self.species_list)
        else:
            # Just do a single spectro, no need for destination
            # subdirs:
            self.spectrogram_dir_path = outdir

        # Allow others outside the instance to find the spectros: 
        SpectrogramChopper.spectrogram_dir_path = self.spectrogram_dir_path

    #------------------------------------
    # chop_from_file_list 
    #-------------------
    
    @classmethod
    def chop_from_file_list(
            cls,
            assignments, 
            in_dir,
            out_dir,
            global_info,
            overwrite_policy,
            return_bool,
            env=None): 
        '''
        Takes a list like:
        
           [(s1,f1),(s1,f2),(s4,f3)]
           
        where s_n is a species name, and f_m
        is the basename of a spectrogram file to chop.
        Example: foobar.png
        
        Returns True if all went well, else
        raises exception.
        
        Wrinkle: this method is called under two 
        very different scenarios (S1/S2). S1 is
        when the process started by the user calls
        this method. That happens when the command
        line arg --workers is set to 1, or on a machine
        where few enough cores are available that only
        one is used. In that case, env is left at None,
        and all is as normal.
        
        S2 occurs when the initial process (the one started
        from the command line) starts a new Process. That
        process normally contains a new environment, i.e. 
        some default value for all the environment variables.
        ************ NEEDED WITHOUT LIBROSA USAGE?
        In particular, DISPLAY and PYTHONPATH will not be
        what is needed. The result is that all spectrogram
        creating methods fail, because they cannot find a
        graphics backend. 
        
        In that case kwarg env is set to the environment of the 
        initiating process. At the start of this method this
        process' default environ is then set to match that
        of the initiating process.
        
        :param assignments: list of species/filename pairs
        :type assignments: [(str,str)]
        :param env: if provided, the environment of the
            parent process. If None, the current env
            is retained
        :type env: {str : Any}
        :param return_bool:
        :type return_bool:
        '''
       
        # During multiprocessing this method is
        # the 'target', i.e. the entry point for 
        # each child. In that case env will be 
        # the environment of the initiating process.
        # We adopt that environment for this new,
        # forked process as well:
        
        if env is not None:
            os.environ = env

        # Optimism!
        return_bool.value = True

        for species_name, fname in assignments:
            # Ex. species_name: AMADEC
            # Ex. fname       : dysmen_my_bird.png
            full_spectro_path = os.path.join(in_dir, species_name, fname)
            try:
                cls.chop_one_spectro_file(full_spectro_path,
                                          os.path.join(out_dir, species_name),
                                          species_name,
                                          overwrite_policy=overwrite_policy
                                          )
            except Exception as e:
                return_bool.value = False
                cls.log.err((f"One file could not be processed \n"
                             f"    ({full_spectro_path}):\n"
                             f"    {repr(e)}")
                             )
                continue

    #------------------------------------
    # chop_one_spectro_file 
    #-------------------

    @classmethod
    def chop_one_spectro_file(cls, 
                              spectro_fname, 
                              out_dir,
                              species_name, 
                              window_len = 5, 
                              skip_size=2,
                              original_duration=None,
                              overwrite_policy=WhenAlreadyDone.ASK,
                              ):
        """
        Generates window_len second spectrogram snippets
        from spectrograms files of arbitrary length. 
        
        To compute the number of time slices to extract
        for each snippet, the time_slices of the spectrogram time
        slices in fractional seconds must be known. The time_slices
        can be approximated if the play length of the underlying
        audio is known (even if the precise fft settings are unavailable).
        
        If the given .png file contains metadata with a 'duration' 
        key, then the corresponding value is used as the duration of 
        the original audio file in fractional seconds. This metadata
        will be present if the .png file was created with the 
        SoundProcessor.create_spectrogram(). 
        
        To enable use of spectrogram images created elsewhere, callers
        can instead supply original_duration in fractional seconds.
        
        For now, if neither the embedded metadata, nor the original_duration
        is supplied, a ValueError is raised. 
    
        :param spectro_fname: full path to spectrogram file to chop
        :type spectro_fname: str
        :param out_dir: root directory under which spectrogram
            snippets will be saved (in different subdirs)
        :type out_dir: str
        :param species_name: name of species to embed in the 
            metadata of this snippet, and use for determining
            subdirectory where to place the snippet
        :type species_name: str
        :param window_len: number of seconds to be covered by each snippet
        :type window_len: int
        :param skip_size: number of seconds to shift right in 
            time for the start of each chop
        :type skip_size: int
        :param original_duration:
        :raise ValueError: if neither embedded duration metadata is found
            in the given file, nor original_duration is provided
        """

        # Read the spectrogram, getting an np array:
        spectro_arr, metadata = SoundProcessor.load_spectrogram(spectro_fname)
        duration = metadata.get('duration', None)

        if duration is None:
            if original_duration is None:
                raise ValueError(f"Time duration of original recording cannot be determined for {spectro_fname}")
            else:
                duration = float(original_duration)
        else:
            duration = float(duration)
             
        # If original file is already at or below
        # the single window length, it's a snippet
        # in itself. Copy it to the output with an
        # appropriate snippet name to match the other
        # snippets: wall start time is zero:
        
        if duration < window_len:
            # No partial snippets
            return
        # Note: Also have sample rate ('sr') and species ('label')
        # in the metadata, but don't need those here.
        
        _freq_bands, time_slices  = spectro_arr.shape 
        # Time in fractions of second
        # per spectrogram column: 
        twidth = duration / time_slices
        
        # Integer of duration (which is in seconds):
        time_dur_int = int(np.ceil(duration))
        time_upper_bound = 1 + time_dur_int - skip_size
        
        # Caller specifies skip_size and window
        # length in *seconds*. Convert to spectrogram 
        # time slices (with rounding error):

        samples_win_len = int(window_len // twidth)
        # Does samples_win_len satisfy the 
        # minimum spectrogram snippet width for 
        # pretrained models?
        samples_win_len = max(cls.MIN_SNIPPET_WIDTH, samples_win_len)
        
        time_true_each_snippet = samples_win_len * twidth
        
        samples_skip_size   = int(skip_size // twidth)
        samples_upper_bound = int(time_upper_bound // twidth)

        assert(samples_upper_bound <= time_slices)
        
        for _snip_num, samples_start_idx in enumerate(range(0, 
                                                            samples_upper_bound, 
                                                            samples_skip_size)):

            # Absolute start time of this snippet
            # within the entire spectrogram:
            wall_start_time   = samples_start_idx * twidth
            # Create a name for the snippet file: 
            snippet_path = cls.create_snippet_fpath(spectro_fname,
                                                    round(wall_start_time), 
                                                    out_dir)
            
            spectro_done = os.path.exists(snippet_path)

            if spectro_done:
                if overwrite_policy == WhenAlreadyDone.SKIP:
                    # Next snippet:
                    continue
                elif overwrite_policy == WhenAlreadyDone.ASK:
                    if not Utils.user_confirm(f"Snippet {Path(snippet_path).stem} exists, overwrite?", default='N'):
                        continue

            # Chop: All rows, columns from current
            #       window start for window lenth samples:
            snippet_data = spectro_arr[:,samples_start_idx : samples_start_idx + samples_win_len]
            _num_rows, num_cols = snippet_data.shape
            if num_cols < samples_win_len:
                # Leave that little spectrogram
                # snippet leftover for Elijah:
                break

            snippet_info = metadata.copy()
            # Add the 
            snippet_info['duration(secs)']   = samples_win_len * twidth
            snippet_info['start_time(secs)'] = wall_start_time
            snippet_info['end_time(secs)']   = wall_start_time + (samples_win_len * twidth)
            snippet_info['species']          = species_name
            SoundProcessor.save_image(snippet_data, snippet_path, snippet_info)
        return time_true_each_snippet
    
    #------------------------------------
    # create_dest_dirs 
    #-------------------

    def create_dest_dirs(self, species_list):
        '''
        Creates all directories that will hold new 
        spectrogram snippets for each species.
        For each directory: if dir exists:
        
           o if overwrite_policy is True, wipe the dir
           o if overwrite_policy is SKIP, leave the
               directory in place, contents intact 
           o else ask user. 
                If response is Yes, wipe the dir
                else raise FileExistsError
                
        :param species_list: names of species to process
        :type species_list: [str]
        :return: top level dir for spectrograms (same as self.out_dir)
        :rtype: (str)
        :raise FileExistsError: if a dest dir exists and not allowed
            to wipe it.
        '''

        # Root dir of each species' spectro snippets:
        Utils.create_folder(self.out_dir, overwrite_policy=self.overwrite_policy)

        # One dir each for the spectrogram snippets of one species:
        
        for species in species_list:
            species_spectros_dir = os.path.join(self.out_dir, species)
            if not Utils.create_folder(species_spectros_dir,
                                       overwrite_policy=self.overwrite_policy):
                raise FileExistsError(f"Target dir {species_spectros_dir} exists; aborting")

        return self.out_dir

    #------------------------------------
    # create_snippet_fpath 
    #-------------------
    
    @classmethod
    def create_snippet_fpath(cls, origin_nm, wall_start_time, out_dir):
        '''
        Given constituent elements, construct the
        full output path of a new spectrogram snippet.
        
        Name format if full-length spectrogram file were
        named my_file.png:
        
              my_file_sw-start123.png
              
        where 123 is the snippet's start time in seconds
        from the beginning of the full length file
        
        :param origin_nm: name of full length file; either full path or
            just the file name are fine
        :type origin_nm: str
        :param wall_start_time: snippet start time from beginning
            of full length spectrogram
        :type wall_start_time: int
        :param out_dir: destination directory
        :type out_dir: str
        :return: full path to the future snippet's destination
        :rtype: str
        '''
        
        # Prepare snippet file name creation:
        
        #   From '/foo/bar/infile.png'
        # make 'infile'
        snippet_name_stem = Path(origin_nm).stem
        
        snippet_name = f"{snippet_name_stem}_sw-start{str(wall_start_time)}.png"
        snippet_path = os.path.join(out_dir, snippet_name)
        return snippet_path

    # -------------------- Class Methods ------------
    
    #------------------------------------
    # compute_worker_assignments 
    #-------------------
    
    @classmethod
    def compute_worker_assignments(cls, 
                                   in_dir,
                                   dst_dir,
                                   overwrite_policy=WhenAlreadyDone.ASK, 
                                   num_workers=None):
        '''
        Given the root directory of a set of
        directories whose names are species,
        and which contain spectrograms by species,
        return a multi processing worker assignment.
        
        Expected:
                         in_dir

          Species1        Species2   ...     Speciesn
           smpl1_1.png      smpl2_1.png         smpln_1.png
           smpl1_2.png      smpl2_2.png         smpln_2.png
                            ...
        
        Collects number of spectrograms available for
        each species. Creates a list of species name
        buckets such that all workers asked to process
        one of the buckets, will have roughly equal
        amounts of work.
        
        Example return:
            
            [['Species1', 'Species2], ['Species3', 'Species4', 'Species5']]
            
        The caller can then assign the first list to
        one worker, and the second list to another worker.
        
        The number of buckets, and therefore the number
        of eventual workers may be passed in. If None, 
        80% of the cores available on the current machine
        will be assumed. If num_workers is provided and
        the number is larger than the number of available
        cores, the number is reduced to the number of cores.
        
        Also returned is the number of workers on which the
        computation is based. This number is always the same
        as the number of species name lists in the return.
        But for clarity, the number is returned explicitly.

        :param in_dir: root of species recordings
        :type in_dir: str
        :param num_workers: number of buckets into which to partition 
        :type num_workers: {int | None}
        :return: list of species name lists, and number of workers.
        :rtype: ([[int]], int)
        '''

        # Create:
        #     {species : num-recordings}
        #     {species : recordings_dir}
        #     [(species1, fpath1), (species1, fpath2), (species2, fpath3)...]  
        
        sample_size_distrib = OrderedDict({})
        sample_dir_dict     = {}
        species_file_tuples = []
        
        for _dir_name, subdir_list, _file_list in os.walk(in_dir):
            for species_name in subdir_list:
                species_spectros_dir = os.path.join(in_dir, species_name)
                spectro_paths = os.listdir(species_spectros_dir)

                # Create new spectro_paths with only spectro files that
                # need chopping:
                new_rec_paths = cls.cull_spectro_paths(
                    species_name, 
                    dst_dir, 
                    spectro_paths, 
                    overwrite_policy
                    )

                sample_size_distrib[species_name] = len(spectro_paths)
                sample_dir_dict[species_name] = species_spectros_dir
                species_file_pairs = list(zip([species_name]*len(new_rec_paths), 
                                              new_rec_paths))

                species_file_tuples.extend(species_file_pairs)
            break 

        if len(species_file_tuples) == 0:
            # If no subdirectories with spectrograms were
            # found, warn:
            cls.log.warn((f"\n"
                          f"    All spectrograms in {in_dir} already chopped.\n"
                          f"    Or did you mean to create an individual file\n"
                          f"    rather than a set of species subdirs?")
                          )
        
        num_cores = mp.cpu_count()
        # Use 80% of the cores:
        if num_workers is None:
            num_workers = round(num_cores * Utils.MAX_PERC_OF_CORES_TO_USE  / 100)
        elif num_workers > num_cores:
            # Limit pool size to number of cores:
            num_workers = num_cores

        # Create a partitioning into equal sized files,
        # regardless of species association.
        
        assignments = cls.partition_by_recordings(species_file_tuples,
                                                  num_workers)
        num_workers_used = len(assignments)
        return assignments, num_workers_used

    #------------------------------------
    # partition_by_recordings 
    #-------------------
    
    @classmethod
    def partition_by_recordings(cls, species_file_pairs, num_workers):
        '''
        Given a list of species-name/file-path tuples, 
        partition that list into num_workers sublists,
        such that each list contains roughly the same
        number of tuples. If the number of species_file_pairs
        tuples is not divisible by num_workers, the left-over
        tuples are distributed over the first sublists.

        :param species_file_pairs:
        :type species_file_pairs:
        :param num_workers:
        :type num_workers:
        :return partitioning of the species_file_pairs tuples
        :rtype: [[(str, str)]]
        '''

        # Compute near-equal number of files per worker:
        num_spectros  = len(species_file_pairs)
        spectros_per_worker = int(np.ceil(num_spectros / num_workers))
        
        # Create list of species-file pair lists:
        #    [[(s1,f1), (s1,f2)], [s1,f3,s2:f4], ...]
        # Each inner list will be handled by one worker:
        
        assignments = []
        assign_idx  = 0
        for _worker_idx in range(num_workers):
            assign_sublist = species_file_pairs[assign_idx:assign_idx+spectros_per_worker]
            assignments.append(assign_sublist)
            assign_idx += spectros_per_worker
        
        num_tasks = sum([len(ass) for ass in assignments])
        # The following seems never to happen, but
        # too tired to figure out why:
        left_overs = num_spectros - num_tasks
        if left_overs > 0:
            # Can't have more than num_workers left overs,
            # meaning can't have more leftovers than
            # sublists. Distribute the leftovers:=
             
            for idx, left_over in enumerate(species_file_pairs[-left_overs:]):
                assignments[idx].append(left_over)

        # Remove empty assignments:
        assignments = [ass for ass in assignments if len(ass) > 0]
        return assignments

    #------------------------------------
    # run_workers 
    #-------------------
    
    @classmethod
    def run_workers(cls,
                    args,
                    global_info,
                    overwrite_policy=WhenAlreadyDone.ASK):
        '''
        Called by main to run the SpectrogramChopper in
        multiple processes at once. Partitions the
        audio files to be processed; runs the chopping
        while giving visual progress on terminal.
        
        Prints success/failure of each worker. Then
        returns. In order to avoid processes repeatedly
        reporting the same, or only locally kept info,
        the globally visible dict `global_info` is passed in.
        
        This method will add these key/val pairs:
        
           1 The total number of spectros to chop (key 'num_tasks')
           2 The number of already created snippets (key 'num_snips')
           3 A list with values False for each job, indicating
               that the corresponding job is not yet done (key 'jobs_status')

        Processes will update 2 and 3 as they report progress: 

        :param args: all arguments provided to argparse
        :type args: {str : Any}
        :param global_info: interprocess communication
            dict for reporting progress
        :type global_info: multiprocessing.manager.dict
        '''
        
        # Get a list of lists of species names
        # to process. The list is computed such
        # that each worker has roughly the same
        # number of recordings to chop. We let
        # the method determine the number of workers
        # by using 80% of the available cores. 
        
        (worker_assignments, num_workers) = SpectrogramChopper.compute_worker_assignments(
            args.input,
            args.outdir,
            num_workers=args.workers)
    
        print(f"Distributing workload across {num_workers} workers.")

        # Initialize the dict with shared information:
        
        # Fill the inter-process list with False.
        # Will be used to logging jobs finishing 
        # many times to the console (i.e. not used
        # for functions other than reporting progress:
        
        for _i in range(num_workers):
            # NOTE: reportedly one cannot just set the passed-in
            #       list to [False]*num_workers, b/c 
            #       a regular python list won't be
            #       seen by other processes, even if
            #       embedded in a multiprocessing.manager.list
            #       instance:
            global_info['jobs_status'].append(False)

        # Number of full spectrograms to chop:
        global_info['snips_to_do'] = len(Utils.find_in_dir_tree(args.input, 
                                                               pattern="*.png")) 
        
        
        # For progress reports, get number of already
        # existing .png files in out directory:
        global_info['snips_done'] = len(Utils.find_in_dir_tree(args.outdir, 
                                                               pattern="*.png")) 

        # Assign each list of species to one worker:
        
        chopping_jobs = []
        for ass_num, assignment in enumerate(worker_assignments):
            chopper = SpectrogramChopper(args.input, 
                                   args.outdir,
                                   overwrite_policy=overwrite_policy
                                   )
            ret_value_slot = mp.Value("b", False)
            job = ProcessWithoutWarnings(target=chopper.chop_from_file_list,
                                         args=(assignment,
                                               args.input,
                                               args.outdir,
                                               global_info,  # ***NEW
                                               overwrite_policy, 
                                               ret_value_slot),
                                         name=f"ass# {ass_num}"
                                         )
            job.ret_val = ret_value_slot
            
            chopping_jobs.append(job)
            print(f"Starting chops for {job.name}")
            job.start()
        
        start_time = datetime.datetime.now()

        # Keep checking on each job, until
        # all are done as indicated by all jobs_done
        # values being True, a.k.a valued 1:
        
        while sum(global_info['jobs_status']) < num_workers:
            for job_idx, job in enumerate(chopping_jobs):
                # Timeout 1 sec
                job.join(1)
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
                    global_info['snips_done'] = cls.sign_of_life(job, 
                                                                 global_info['snips_done'],
                                                                 args.outdir,
                                                                 start_time,
                                                                 force_rewrite=True)
                    # Check on next job:
                    continue
                
                # This job not finished yet.
                # Time for sign of life?
                global_info['snips_done'] = cls.sign_of_life(job, 
                                                             global_info['snips_done'],
                                                             args.outdir,
                                                             start_time,
                                                             force_rewrite=True)

    #------------------------------------
    # cull_spectro_paths
    #-------------------

    @classmethod
    def cull_spectro_paths(cls, 
                           species_or_recorder_name, 
                           dst_dir, 
                           rec_paths, 
                           overwrite_policy=WhenAlreadyDone.ASK):
        #******* DISABLED ************
        # method analogous to cull_rec_paths() in create_spectrograms()
        # Currently below is just a copy from create_spectrograms().
        # If we end up needing culling, update this body
        return rec_paths
        #******* DISABLED ************
        # NEVER REACHED
        new_rec_paths = []
        for aud_fname in rec_paths:
            fname_stem = Path(aud_fname).stem
            dst_path = os.path.join(dst_dir, species_or_recorder_name, f"{fname_stem}.png")
            if not os.path.exists(dst_path):
                # Destination spectrogram does not exist;
                # keep this audio file in the to-do list:
                new_rec_paths.append(aud_fname)
                continue
            if overwrite_policy == WhenAlreadyDone.OVERWRITE:
                os.remove(dst_path)
                new_rec_paths.append(aud_fname)
                continue
            if overwrite_policy == WhenAlreadyDone.SKIP:
                # Don't even assign audio file to a worker,
                # since its spectro already exists:
                continue
            if overwrite_policy == WhenAlreadyDone.ASK:
                if Utils.user_confirm(f"Spectrogram for {dst_path} exists; overwrite?"):
                    os.remove(dst_path)
                    new_rec_paths.append(aud_fname)
                    continue
        return new_rec_paths

    #------------------------------------
    # sign_of_life 
    #-------------------
    
    @classmethod
    def sign_of_life(cls, 
                     job, 
                     num_already_present_imgs, 
                     outdir, 
                     start_time, 
                     force_rewrite=False):
        
        # Time for sign of life?
        now_time = datetime.datetime.now()
        time_duration = now_time - start_time
        # Every 3 seconds, but at least 3:
        if force_rewrite \
           or (time_duration.seconds > 0 and time_duration.seconds % 3 == 0): 
            
            # A human readable duration st down to minutes:
            duration_str = Utils.time_delta_str(time_duration, granularity=4)

            # Get current and new spectro imgs in outdir:
            num_now_present_imgs = len(Utils.find_in_dir_tree(outdir, pattern="*.png"))
            num_newly_present_imgs = num_now_present_imgs - num_already_present_imgs

            # Keep printing number of done snippets in the same
            # terminal line:
            print((f"{job.name}---Number of spectros: {num_now_present_imgs} "
                   f"({num_newly_present_imgs} new) after {duration_str}"),
                  end='\r')
            return num_newly_present_imgs
        else:
            return num_already_present_imgs

# ------------------------ Main -----------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    parser.add_argument('input',
                        type=str,
                        help='the path to directory with .png files, or to a single .png file to chop')
    parser.add_argument('outdir',
                        type=str,
                        help='the path to output directory to write new .png files')
    parser.add_argument('-w', '--workers',
                        type=int,
                        help='number of cores to use; default: 80 percent of available cores',
                        default=None)
    parser.add_argument('-y', '--overwrite',
                        help='if set, overwrite existing out directories without asking; default: False (ask)',
                        action='store_true',
                        default=False)
    parser.add_argument('-r', '--resume',
                        help='if set, skip any spectrogram snippets that already exist; default: False',
                        action='store_true',
                        default=False)

    # Execute the parse_args() method
    args = parser.parse_args()

    # Input makes sense?
    if not os.path.exists(args.input):
        print(f"Input {args.input} does not exist")
        sys.exit(1)

    # Can't have both -y and -r:
    if args.overwrite and args.resume:
        print(f"Cannot have both, --overwrite/-y and --resume/-r")
        sys.exit(1)
        
    if args.overwrite:
        overwrite_policy = WhenAlreadyDone.OVERWRITE
    elif args.resume:
        overwrite_policy = WhenAlreadyDone.SKIP
    else:
        overwrite_policy = WhenAlreadyDone.ASK

    # Don't use multiprocessing if either the 
    # number of workers was explicitly set to 1 by
    # the caller, or only a single spectrogram is 
    # to be processed:
    if args.workers == 1 or os.path.isfile(args.input):
        chopper = SpectrogramChopper(args.input, 
                                     args.outdir,
                                     overwrite_policy=overwrite_policy
                                     )
        # Use single workers only for individual files:
        if os.path.isdir(args.input):
            raise ValueError(f"Use of single worker only for individual spectrogram file, not dirs: {args.input}")
        
        chopper.chop_one_spectro_file(args.input, args.outdir)
        print('Done')
        sys.exit(0)

    # Multiprocessing the chopping:
    # For nice progress reports, create a shared
    # dict quantities that each process will report
    # on to the console as progress indications:
    #    o Which job-completions have already been reported
    #      to the console (a list)
    #    o The most recent number of already created
    #      output images 
    # The dict will be initialized in run_workers,
    # but all Python processes on the various cores
    # will have read/write access:
    manager = mp.Manager()
    global_info = manager.dict()
    global_info['jobs_status'] = manager.list()

    
    # Use SKIP overwrite policy to ensure
    # workers don't step on each others' toes:
    SpectrogramChopper.run_workers(args,
                                   global_info,
                                   overwrite_policy=WhenAlreadyDone.SKIP
                                   )

    print("Done")
