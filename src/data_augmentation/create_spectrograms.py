#!/usr/bin/env python3
'''
Created on Apr 26, 2021

@author: paepcke
'''

from _collections import OrderedDict
import argparse
import datetime
import os
from pathlib import Path
import sys

from logging_service import LoggingService

from birdsong.utils.utilities import FileUtils
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import ProcessWithoutWarnings
from data_augmentation.utils import Utils, WhenAlreadyDone
import multiprocessing as mp
import numpy as np


class SpectrogramCreator:
    '''
    Spectrogram creation and manipulation
    '''
    
    # Get a Python logger that is
    # common to all modules in this
    # package:

    log = LoggingService()
    
    # Allow some actions to adjust during
    # unittesting:
    
    UNITTEST_ACTIONS_COUNTS = {}
    
    
    #------------------------------------
    # create_from_file_list 
    #-------------------
    
    @classmethod
    def create_from_file_list(cls,
                              assignments, 
                              in_dir,
                              out_dir,
                              overwrite_policy,
                              return_bool, 
                              env=None):
        '''
        Takes a list like:
        
           [(s1,f1),(s1,f2),(s4,f3)]
           
        where s_n is a species/audio-recorder name, and f_m
        is the basename of an audio file for which to
        create a spectrogram.
        
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
        :param in_dir: source of audio files: individual file,
            or root of subdirectories with audio files
        :type in_dir: str
        :param out_dir: destination directory of spectrogram(s) 
        :type out_dir: str
        :param overwrite_policy: what to do if dest spectrogram
            already exists
        :type overwrite_policy: WhenAlreadyDone 
        :param return_bool: place to place return value
        :type return_bool: mp.Value
        :param env: if provided, the environment of the
            parent process. If None, the current env
            is retained
        :type env: {str : Any}
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
            full_audio_path = os.path.join(in_dir, species_name, fname)
            try:
                cls.create_one_spectrogram(full_audio_path,
                                           os.path.join(out_dir, species_name),
                                           overwrite_policy=overwrite_policy,
                                           identifier=species_name
                                           )
            except Exception as e:
                return_bool.value = False
                cls.log.err((f"One file could not be processed \n"
                             f"    ({full_audio_path}):\n"
                             f"    {repr(e)}")
                             )
                continue

    #------------------------------------
    # compute_worker_assignments 
    #-------------------
    
    @classmethod
    def compute_worker_assignments(cls, 
                                   in_dir, 
                                   dst_dir, 
                                   overwrite_policy=WhenAlreadyDone.ASK, 
                                   num_workers=None
                                   ):
        '''
        Given the root directory of a set of
        directories whose names are species or field
        recorder IDs, and which contain audio recordings,
        return a multi processing worker assignment.
        
        :param cls:
        :type cls:
        Expected:
                         in_dir

          Species1/Recorder1      Species2/Recorder2   ... Speciesn/Recordern
           smpl1_1.wav      	    smpl2_1.wav              smpln_1.wav
           smpl1_2.wav      	    smpl2_2.wav              smpln_2.wav
                            	    ...
        
        Collects number of audio files available for each species 
        or audio recorder. Creates a list of species name buckets 
        such that all workers asked to process one of the buckets, 
        will have roughly equal amounts of work.
        
        Example return:
            
            [['Species1', 'Species2], ['Species3', 'Species4', 'Species5']]
            
        Where Species_n are directories.
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
        :param dst_dir: root of subdirectories that will contain
            spectrograms
        :type dst_dir: str
        :param num_workers: number of buckets into which to partition 
        :type num_workers: {int | None}
        :return: list of species name lists, and number of workers.
        :rtype: ([[int]], int)
        '''

        # Create the following structures:
        #     {species : num-recordings}
        #     {species : recordings_dir}
        #     [(species1, fpath1), (species1, fpath2), (species2, fpath3)...]  
        
        sample_size_distrib = OrderedDict({})
        sample_dir_dict     = {}
        species_file_tuples = []
        
        for _dir_name, subdir_list, _file_list in os.walk(in_dir):
            for species_or_recorder_name in subdir_list:
                species_recordings_dir = os.path.join(in_dir, species_or_recorder_name)
                rec_paths = os.listdir(species_recordings_dir)
                # Create new rec_paths with only audio files that
                # need a spectrogram:
                new_rec_paths = cls.cull_rec_paths(
                    species_or_recorder_name, 
                    dst_dir, 
                    rec_paths, 
                    overwrite_policy
                    )
                sample_size_distrib[species_or_recorder_name] = len(new_rec_paths)
                sample_dir_dict[species_or_recorder_name] = species_recordings_dir
                species_file_pairs = list(zip([species_or_recorder_name]*len(new_rec_paths), 
                                              new_rec_paths))
                species_file_tuples.extend(species_file_pairs)
            break 
        
        if len(species_file_tuples) == 0:
            # If no subdirectories with spectrograms were
            # found, warn:
            cls.log.warn((f"\n"
                          f"    All audio in {in_dir} already done.\n"
                          f"    Or meant to create an individual file\n"
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

        :param species_file_pairs: list of species or recorder name
            soundfile name tuples
        :type species_file_pairs: [(str, str)]
        :param num_workers: number of computer cores to use
        :type num_workers: int
        :return partitioning of the species_file_pairs tuples
        :rtype: [[(str, str)]]
        '''

        # Compute near-equal number of files per worker:
        num_recordings  = len(species_file_pairs)
        recs_per_worker = int(np.ceil(num_recordings / num_workers))
        
        # Create list of species-file pair lists:
        #    [[(s1,f1), (s1,f2)], [s1,f3,s2:f4], ...]
        # Each inner list will be handled by one worker:
        
        assignments = []
        assign_idx  = 0
        for _worker_idx in range(num_workers):
            assign_sublist = species_file_pairs[assign_idx:assign_idx+recs_per_worker]
            assignments.append(assign_sublist)
            assign_idx += recs_per_worker
        
        num_tasks = sum([len(ass) for ass in assignments])
        # The following seems never to happen, but
        # too tired to figure out why:
        left_overs = num_recordings - num_tasks
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
                    overwrite_policy=WhenAlreadyDone.ASK,
                    ):
        '''
        Called by main to run the SpectrogramCreator in
        multiple processes at once. Partitions the
        audio files to be processed; runs the spectrogram 
        creation while giving visual progress on terminal.
        
        Prints success/failure of each worker.

        :param args: all arguments provided to argparse
        :type args: {str : Any}
        :param global_info: interprocess communication
            dict for reporting progress
        :type global_info: multiprocessing.manager.dict
        :param overwrite_policy: what to do when a spectrogram already exists
        :type overwrite_policy: WhenAlreadyDone
        '''

        # Get a list of lists of species names
        # to process. The list is computed such
        # that each worker has roughly the same
        # number of recordings for which to create. 
        # spectrograms. We let the method determine the 
        # number of workers by using Utils.MAX_PERC_OF_CORES_TO_USE 
        # of the available cores. 
        
        (worker_assignments, num_workers) = SpectrogramCreator.compute_worker_assignments(
            args.input,
            args.outdir,
            overwrite_policy=overwrite_policy,
            num_workers=args.workers)

        num_assignments = sum([len(assignments) for assignments in worker_assignments])
        print(f"Distributing {num_assignments} tasks across {num_workers} workers.")
        
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
        
        spectro_creation_jobs = []
        for ass_num, assignment in enumerate(worker_assignments):
            
            ret_value_slot = mp.Value("b", False)
            job = ProcessWithoutWarnings(target=SpectrogramCreator.create_from_file_list,
                                         args=(assignment,
                                               args.input,
                                               args.outdir,
                                               overwrite_policy, 
                                               ret_value_slot),
                                         name=f"ass# {ass_num}"
                                         )
            job.ret_val = ret_value_slot

            spectro_creation_jobs.append(job)
            print(f"Starting spectro creation for {job.name}")
            job.start()

        start_time = datetime.datetime.now()
        for _i in range(num_workers):
            # NOTE: reportedly one cannot just set the passed-in
            #       list to [False]*num_workers, b/c 
            #       a regular python list won't be
            #       seen by other processes, even if
            #       embedded in a multiprocessing.manager.list
            #       instance:
            global_info['jobs_status'].append(False)

        # Keep checking on each job, until
        # all are done as indicated by all jobs_done
        # values being True, a.k.a valued 1:
        
        while sum(global_info['jobs_status']) < len(spectro_creation_jobs):
            for job_idx, job in enumerate(spectro_creation_jobs):
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
                    
                    # This job finished
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
    # sign_of_life 
    #-------------------
    
    @classmethod
    def sign_of_life(cls, job, num_already_present_imgs, outdir, start_time, force_rewrite=False):
        # Time for sign of life?
        now_time = datetime.datetime.now()
        time_duration = now_time - start_time
        # Every 3 seconds, but at least 3:
        if force_rewrite \
           or (time_duration.seconds > 0 and time_duration.seconds % 3 == 0): 
            
            # A human readable duration st down to minutes:
            duration_str = FileUtils.time_delta_str(time_duration, granularity=4)

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
    
    #------------------------------------
    # cull_rec_paths
    #-------------------

    @classmethod
    def cull_rec_paths(cls, 
                       species_or_recorder_name, 
                       dst_dir, 
                       rec_paths, 
                       overwrite_policy=WhenAlreadyDone.ASK):
        '''
        Check whether any of the audio recordings in rec_paths
        already contain a corresponding spectro under subdirectories
        of <dst_dir>/species_or_recorder_name. Collects the audio
        files that do not already have a spectrogram, and returns
        that TODO list. 

        :param species_or_recorder_name: species name (e.g. 'sparrow', or
            audio moth recorder name (e.g. 'AM03')
        :type species_or_recorder_name: str
        :param dst_dir: destination root for spectrograms
        :type dst_dir: str
        :param rec_paths: list of full paths to audio files
             to be spectrogrammed
        :type rec_paths: src
        :param overwrite_policy: policy when spectro already present
        :type overwrite_policy: WhenAlreadyDone
        :rturn new list of audio recordings for those that do need
            spectrograms to be created
        :rtype WhenAlreadyDone
        '''
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
    # create_one_spectrogram 
    #-------------------
    
    @classmethod
    def create_one_spectrogram(cls, 
                               aud_file, 
                               dst_dir,
                               overwrite_policy=WhenAlreadyDone.ASK, 
                               identifier='Unknown'):
        '''
        Convert an audio file into a spectrogram.
        The destination for the .png file is constructed
        from dst_dir, and the audio file name.
        
        The .png file will contain metadata:
        
            o 'sr'          : <sample rate>
            o 'duration'    : <number of seconds>
            o 'identifier'  : <any identifier as per arg to this method>
        
        If the destination file already exists, the 
        overwrite_policy is followed: ASK, OVERWRITE, or
        SKIP.

        :param aud_file: file to convert: .wav or .mp3
        :type aud_file: str
        :param dst_dir: directory for the resulting .png file
        :type dst_dir: str
        :param overwrite_policy: action if destination file exists
        :type overwrite_policy: WhenAlreadyDone
        :param identifier: typically ID of audio recorder or species
        :type identifier: str
        :raise FileNotFoundError, AudioLoadException
        '''

        sound, sr = SoundProcessor.load_audio(aud_file)

        # Get parts of the file: root, fname, suffix
        path_el_dict  = Utils.path_elements(aud_file)
        new_fname = path_el_dict['fname'] + '.png'
        
        dst_path = os.path.join(dst_dir, new_fname)
        # If dst exists, ask whether to overwrite;
        # skip file if not:
        if os.path.exists(dst_path) and overwrite_policy == WhenAlreadyDone.ASK:
            # During unittesting the following
            # throws an EOFError. Prevent that from showing:
            try:
                do_overwrite = Utils.user_confirm(f"Overwrite {dst_path}?", False)
            except EOFError:
                print("Utils.user_confirm() would have been called; pretend 'no'")
                # Act as if user had replied No:
                return
            if not do_overwrite:
                # Not allowed to overwrite
                return
            else:
                os.remove(dst_path)
        
        #cls.log.info(f"Creating spectrogram for {os.path.basename(dst_path)}")
        SoundProcessor.create_spectrogram(sound, sr, dst_path, info={'identifier' : identifier})


# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Here is what this package does."
                                     )

    parser.add_argument('-n', '--num',
                        type=int,
                        help='limit processing to first n audio files',
                        default=None)

    parser.add_argument('-y', '--overwrite',
                        help='overwrite existing output directories; default: ask, unless --resume',
                        action='store_true',
                        default=False
                        )
    
    parser.add_argument('-r', '--resume',
                        help='skip already present spectrograms; overrides --overwrite default: --overwrite setting',
                        action='store_true',
                        default=False
                        )
    
    parser.add_argument('-w', '--workers',
                        type=int,
                        help='number of cores to use; default: 80 percent of available cores',
                        default=None)
    
    parser.add_argument('-', '--species',
                        help="if input is an individual audio file, the species name; default: 'Unknown'",
                        default="Unknown")

    parser.add_argument('input',
                        help='directories of audio files whose spectrograms to create, or path to a single audio file',
                        default=None)
    parser.add_argument('outdir',
                        help='destination of spectrogram files',
                        default=None)

    args = parser.parse_args()

    # Turn the overwrite_policy arg into the
    # proper enum mem
    if args.resume:
        overwrite_policy = WhenAlreadyDone.SKIP
    elif args.overwrite:
        overwrite_policy = WhenAlreadyDone.OVERWRITE
    else:
        overwrite_policy = WhenAlreadyDone.ASK

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input {args.input} does not exist")

    # If input is a single audio file, just create 
    # a spectro for it; same if worker is explicitly
    # set to 1:
    if Utils.is_audio_file(args.input) or args.workers == 1:
        SpectrogramCreator.create_one_spectrogram(
            args.input, 
            args.outdir, 
            overwrite_policy=overwrite_policy, 
            identifier=args.species)
        sys.exit(0)
    
    # Spectrograms for a directory tree
    if not os.path.isdir(args.input):
        raise ValueError(f"Input must either be a sound file, or a directory, not {args.input}")
    
    
    else:
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
        
        SpectrogramCreator.run_workers(args,
                                       global_info,
                                       overwrite_policy=overwrite_policy
                                       )
    sys.exit(0)