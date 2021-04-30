#!/usr/bin/env python3

from _collections import OrderedDict
import argparse
import os, sys
from pathlib import Path
import warnings

# Needed when running headless:
# Do this before any other matplotlib
# imports; directly or indirectly 
# through librosa
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from logging_service.logging_service import LoggingService
from matplotlib import MatplotlibDeprecationWarning

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils
from data_augmentation.utils import WhenAlreadyDone
import multiprocessing as mp
import numpy as np

class SpectrogramChopper:
    '''
    Processes directories of .png files,
    chopping them into window_len seconds snippets.

    Assumes:

                        self.in_dir
                        
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
    Method chop_file_list() is used when multiprocessing. This
    method is the 'target' in the multiprocessing library's sense.

    '''

    # If multiple cores are available,
    # only use some percentage of them to
    # be nice:
    
    MAX_PERC_OF_CORES_TO_USE = 50

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, 
                 input_dir, 
                 output_dir, 
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
        
        @param input_dir: location of soundfile root
        @type input_dir: str
        @param output_dir: root of spectrograms to create
        @type output_dir: src
        @param specific_species: process only a spectific list of species
        @type specific_species: {None | [str]}
        @param overwrite_policy: what to do when an output file already exists
        @type overwrite_policy: WhenAlreadyDone
        '''

        # Ensure the outdir and all its intermediate dirs exist:
        os.makedirs(output_dir, exist_ok=True)

        self.in_dir         	= input_dir
        self.out_dir        	= output_dir
        self.specific_species   = specific_species
        self.overwrite_policy   = overwrite_policy
        
        self.log = LoggingService()
        
        self.num_chopped = 0

        if self.specific_species is None:
            self.species_list = os.listdir(self.in_dir)
        else:
            self.species_list = self.specific_species
        
        # Create directories for new spectrogram snippets
        
        self.spectrogram_dir_path = self.create_dest_dirs(self.species_list)
        
        # Allow others outside the instance to find the spectros: 
        SpectrogramChopper.spectrogram_dir_path = self.spectrogram_dir_path

    #------------------------------------
    # chop_all
    #-------------------

    def chop_all(self):
        '''
        Workhorse: Assuming self.in_dir is root of all
        species spectrograms

        Chops each .png file into window_len snippets.
        Saves those snippets in a new directory under 
        self.spectrogram_dir_path
                
        Resulting directories under self.spectrogram_dir_path 
        will be as per comment in SpectrogramChopper class
        definition.
            
        If self.specific_species is None, audio files under all
        species are chopped. Else, self.specific_species is 
        expected to be a list of species names that correspond
        to the names of species directories above: Species1, Species2, etc.
        
        Returns number of created .png spectrogram snippet files.
        
        :return total number of newly created spectrogram snippets
        :rtype: int
        '''
        for species in self.species_list:
            spectro_files = Utils.listdir_abs(os.path.join(self.in_dir, species))
            num_files   = len(spectro_files)
            for i, fpath in enumerate(spectro_files):
                # Chop one spectrogram file:
                self.log.info(f"Chopping {species} spectro {i}/{num_files}")
                self.chop_one_spectrogram_file(fpath, self.out_dir)
            self.num_chopped += num_files

        num_spectros = Utils.find_in_dir_tree(self.spectrogram_dir_path, pattern='*.png')
        return num_spectros
    
    #------------------------------------
    # chop_file_list 
    #-------------------
    
    def chop_file_list(self, assignment, return_bool, env=None):
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
        
        :param assignment: list of species/filename pairs
        :type assignment: [(str,str)]
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

        for species_name, fname in assignment:
            # Ex. species_name: AMADEC
            # Ex. fname       : dysmen_my_bird.png
            full_spectro_path = os.path.join(self.in_dir, species_name, fname)
            try:
                self.chop_one_spectro_file(full_spectro_path,
                                           os.path.join(self.out_dir, species_name)
                                           )
            except Exception as e:
                return_bool.value = False
                raise e
            
        return_bool.value = True
    
    #------------------------------------
    # chop_one_audio_file 
    #-------------------

    def chop_one_spectro_file(self, spectro_fname, 
                              out_dir, 
                              window_len = 5, 
                              skip_size=2,
                              original_duration=None):
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
        is supplied, an ValueError is raised. 
    
        :param spectro_fname: full path to spectrogram file to chop
        :type spectro_fname: str
        :param window_len: number of seconds to be covered by each snippet
        :type window_len: int
        :param skip_size: number of seconds to shift right in 
            time for the start of each chop
        :type skip_size: int
        :param out_dir: root directory under which spectrogram
            snippets will be saved (in different subdirs)
        :type out_dir: str
        :param original_duration:
        :raise ValueError: if neither embedded duration metadata is found
            in the given file, nor original_duration is provided
        """

        # Read the spectrogram, getting an np array:
        spectro_arr, metadata = SoundProcessor.load_spectrogram(spectro_fname)
        duration	   = metadata.get('duration', None)

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
        
        if duration <= window_len:
            snippet_path = self.create_snippet_fpath(spectro_fname, 0, out_dir)
            SoundProcessor.save_image(spectro_arr, snippet_path, metadata)
            return
        # Note: Also have sample rate ('sr') and species ('label')
        # in the metadata, but don't need those here.
        
        _freq_bands, time_slices  = spectro_arr.shape 
        # Time in franctions of second
        # per spectrogram column: 
        twidth       = duration / time_slices
        
        # Integer of duration (which is in seconds):
        time_dur_int = int(np.ceil(duration))
        time_upper_bound = 1 + time_dur_int - skip_size
        
        # Caller specifies skip_size and window
        # length in *seconds*. Convert to spectrogram 
        # time slices (with rounding error):

        samples_win_len     = int(window_len // twidth)
        samples_skip_size   = int(skip_size // twidth)
        samples_upper_bound = int(time_upper_bound // twidth)

        assert(samples_upper_bound <= time_slices)
        
        for snip_num, samples_start_time in enumerate(range(0, 
                                                            samples_upper_bound, 
                                                            samples_skip_size)):

            # Create a name for the snippet file:
            wall_start_time   = snip_num * skip_size
            snippet_path = self.create_snippet_fpath(spectro_fname,
                                                     wall_start_time, 
                                                     out_dir)
            
            spectro_done = os.path.exists(snippet_path)

            if spectro_done:
                if self.WhenAlreadyDone.SKIP:
                    # Next snippet:
                    continue
                elif self.WhenAlreadyDone.ASK:
                    if not Utils.user_confirm(f"Snippet {Path(snippet_path).stem} exists, overwrite?", default='N'):
                        continue

            # Now know that whether or not snippet
            # exists, we can (re)-create that file: 
            snippet_data = spectro_arr[:,samples_start_time : samples_start_time + samples_win_len]
            snippet_info = metadata.copy()
            snippet_info['duration'] = window_len
            SoundProcessor.save_image(snippet_data, snippet_path, snippet_info)

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
    
    def create_snippet_fpath(self, origin_nm, wall_start_time, out_dir):
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
        :param wall_start_time: snipet start time from beginning
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
    def compute_worker_assignments(cls, in_dir, num_workers=None):
        '''
        Given the root directory of a set of
        directories whose names are species,
        and which contain recordings by species,
        return a multi processing worker assignment.
        
        Expected:
                         in_dir

          Species1        Species2   ...     Speciesn
           smpl1_1.mp3      smpl2_1.mp3         smpln_1.mp3
           smpl1_2.mp3      smpl2_2.mp3         smpln_2mp3
                            ...
        
        Collects number of recordings available for
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
                species_recordings_dir = os.path.join(in_dir, species_name)
                rec_paths = os.listdir(species_recordings_dir)
                sample_size_distrib[species_name] = len(rec_paths)
                sample_dir_dict[species_name] = species_recordings_dir
                species_file_pairs = list(zip([species_name]*len(rec_paths), rec_paths))
                species_file_tuples.extend(species_file_pairs)
            break 
        
        num_cores = mp.cpu_count()
        # Use 80% of the cores:
        if num_workers is None:
            num_workers = round(num_cores * SpectrogramChopper.MAX_PERC_OF_CORES_TO_USE  / 100)
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
        
        left_overs = num_recordings % num_workers
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
    def run_workers(cls, args, overwrite_policy=WhenAlreadyDone.ASK):
        '''
        Called by main to run the SpectrogramChopper in
        multiple processes at once. Pajcrtitions the
        audio files to be processed; runs the chopping
        while giving visual progress on terminal.
        
        Prints success/failure of each worker. Then
        returns

        :param args: all arguments provided to argparse
        :type args: {str : Any}
        '''
        
        in_dir = args.input_dir
    
        # Get a list of lists of species names
        # to process. The list is computed such
        # that each worker has roughly the same
        # number of recordings to chop. We let
        # the method determine the number of workers
        # by using 80% of the available cores. 
        
        (worker_assignments, num_workers) = SpectrogramChopper.compute_worker_assignments(
            in_dir,
            num_workers=args.workers)
    
        print(f"Distributing workload across {num_workers} workers.")
        # Assign each list of species to one worker:
        
        chopping_jobs = []
        for ass_num, assignment in enumerate(worker_assignments):
            chopper = SpectrogramChopper(in_dir, 
                                   args.output_dir,
                                   overwrite_policy=overwrite_policy
                                   )
            ret_value_slot = mp.Value("b", False)
            job = ProcessWithoutWarnings(target=chopper.chop_file_list,
                                         args=([assignment, ret_value_slot]),
                                         name=f"ass# {ass_num}"
                                         )
            job.ret_val = ret_value_slot
            
            chopping_jobs.append(job)
            print(f"Starting chops for {job.name}")
            job.start()
        
        for job in chopping_jobs:
            job_done = False
            while not job_done:
                # Check for job done with one sec timeout: 
                job.join(1)
                # Get number of generated snippets:
                num_chopped_snippets = \
                    len(Utils.find_in_dir_tree(SpectrogramChopper.spectrogram_dir_path))
                # Keep printing number of done snippets in the same
                # terminal line:
                print(f"Number of audio snippets: {num_chopped_snippets}", end='\r')
                # If the call to join() timed out
                if job.exitcode is None:
                    # Job not done:
                    continue
                res = "OK" if job.ret_val else "Error"
                # New line after the progress msgs:
                print("")
                print(f"Chops of {job.name}/{num_workers}: {res}")
                job_done = True


# -------------------- Class ProcessWithoutWarnings ----------

class ProcessWithoutWarnings(mp.Process):
    '''
    Subclass of Process to use when creating
    multiprocessing jobs. Accomplishes two
    items in addition to the parent:
    
       o Blocks printout of various deprecation warnings connected
           with matplotlib and librosa
       o Adds ability for SpectrogramChopper instances to 
           return a result.
    '''
    
    def run(self, *args, **kwargs):

        # Don't show the annoying deprecation
        # librosa.display() warnings about renaming
        # 'basey' to 'base' to match matplotlib: 
        warnings.simplefilter("ignore", category=MatplotlibDeprecationWarning)
        
        # Hide the UserWarning: PySoundFile failed. Trying audioread instead.
        warnings.filterwarnings(action="ignore",
                                message="PySoundFile failed. Trying audioread instead.",
                                category=UserWarning, 
                                module='', 
                                lineno=0)
        try:
            self.ret_value = kwargs['ret_value']
            del kwargs['ret_value']
        except KeyError:
            pass
        
        return mp.Process.run(self, *args, **kwargs)
    
    @property
    def ret_val(self):
        try:
            return self._ret_val
        except NameError:
            return None
        
    @ret_val.setter
    def ret_val(self, new_val):
        if not type(new_val) == mp.sharedctypes.Synchronized:
            raise TypeError(f"The ret_val instance var must be multiprocessing shared C-type, not {new_val}")
        self._ret_val = new_val

# ------------------------ Main -----------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    parser.add_argument('input_dir',
                        type=str,
                        help='the path to original directory with .png files')
    parser.add_argument('output_dir',
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
                        help='if set, skip any soundfiles whose output spectrogram already exists; default: False',
                        action='store_true',
                        default=False)

    # Execute the parse_args() method
    args = parser.parse_args()

    in_dir = args.input_dir
    # Input makes sense?
    if not os.path.exists(in_dir) or not os.path.isdir(in_dir):
        print(f"In directory {in_dir} does not exist, or is not a directory")
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

    if args.workers == 1:
        chopper = SpectrogramChopper(in_dir, 
                               args.output_dir,
                               overwrite_policy=overwrite_policy
                               )
        chopper.chop_all()
    else:
        SpectrogramChopper.run_workers(args,
                                 overwrite_policy=overwrite_policy
                                 )

    print("Done")
