#!/usr/bin/env python3

from _collections import OrderedDict
import argparse
import os,sys
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from logging_service.logging_service import LoggingService

# Needed when running headless:
import matplotlib
matplotlib.use('TkAgg')

from matplotlib import MatplotlibDeprecationWarning
import librosa
import numpy as np
import soundfile as sf

from data_augmentation import utils
from data_augmentation.sound_processor import SoundProcessor
import multiprocessing as mp

class SoundChopper:
    '''
    Processes directories of .wav or .mp3 files,
    chopping them into window_len seconds snippets.
    Each audio snippet is saved, and spectrograms
    are created for each.
    
    Assumes:

                        self.in_dir
                        
          Species1        Species2   ...     Speciesn
           smpl1_1.mp3      smpl2_1.mp3         smpln_1.mp3
           smpl1_2.mp3      smpl2_2.mp3         smpln_2mp3
                            ...
                            
    Saves the snippets in a new directory. Creates a spectrogram 
    for each snippet, and saves those in a different, new directory.
        
        Resulting directories under self.out_dir will be:
         
                         self.out_dir
            spectrograms               wav-files

    Because many spectrograms are created, speed requirements
    call for the use of parallelism. Since each audio file's processing
    is independent from the others, the multiprocessing library
    is used as follows.
    
        - If command line arg --workers is set to 1, no parallelism
          is used. 
        - If multiple cores are available, 80% of them will be
          deployed to chopping. Each core runs a separate copy
          of this file.
        - If 80%

    Method chop_all() is used in the single core scenario.
    Method chop_file_list() is used when multiprocessing. This
    method is the 'target' in the multiprocessing library's sense.

    '''

    # If multiple cores are available,
    # only use some percentage of them to
    # be nice:
    
    MAX_PERC_OF_CORES_TO_USE = 80

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, 
                 input_dir, 
                 output_dir, 
                 specific_species=None,
                 overwrite_freely=False
                 ):
        self.in_dir       = input_dir
        self.out_dir      = output_dir
        self.specific_species = specific_species
        self.overwrite_freely = overwrite_freely
        
        self.log = LoggingService()
        
        self.num_chopped = 0

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

        if self.specific_species is None:
            self.species_list = os.listdir(self.in_dir)
        else:
            self.species_list = self.specific_species
        
        # Create directories for new audio snippets
        # and spectrograms:
        
        self.wav_dir_path, self.spectrogram_dir_path = self.create_dest_dirs(self.species_list)
        
        # Allow others outside the instance
        # find the audio snippet destination
        SoundChopper.wav_dir_path = self.wav_dir_path

    #------------------------------------
    # chop_all
    #-------------------

    def chop_all(self):
        '''
        Workhorse: Assuming self.in_dir is root of all
        species audio samples:
        
                        self.in_dir
                        
          Species1        Species2   ...     Speciesn
           smpl1_1.mp3      smpl2_1.mp3         smpln_1.mp3
           smpl1_2.mp3      smpl2_2.mp3         smpln_2mp3
                            ...
                            
        Chops each .mp3 (or .wav) file into window_len snippets.
        Saves those snippets in a new directory. Creates a spectrogram 
        for each snippet, and saves those in a different, new directory.
        
        Resulting directories under self.out_dir will be:
         
                         self.out_dir
            spectrograms               wav-files
            
        If self.specific_species is None, audio files under all
        species are chopped. Else, self.specific_species is 
        expected to be a list of species names that correspond
        to the names of species directories above: Species1, Species2, etc.
        
        Returns a 2-tuple: (number of created .wav audio snippet files,
                            number of created .png spectrogram snippet files,
        
        '''
        for species in self.species_list:
            audio_files = os.listdir(os.path.join(self.in_dir, species))
            num_files   = len(audio_files)
            for i, sample_name in enumerate(audio_files):
                # Chop one audio file:
                self.log.info(f"Chopping {species} audio {i}/{num_files}")
                self.chop_one_audio_file(self.in_dir, species, sample_name, self.out_dir)
            self.num_chopped += num_files

        num_spectros = utils.find_in_dir_tree(self.spectrogram_dir_path, pattern='*.png')
        num_audios   = utils.find_in_dir_tree(self.wav_dir_path, pattern='*.wav')
        return (num_audios, num_spectros)
    
    #------------------------------------
    # chop_file_list 
    #-------------------
    
    def chop_file_list(self, assignment, return_bool, env=None):
        '''
        Takes a list like:
        
           [(s1,f1),(s1,f2),(s4,f3)]
           
        where s_n is a species name, and f_m
        is the basename of an audio file to chop.
        Example: foobar.mp3
        
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
        In particular, DISPLAY and PYTHONPATH will not be
        what is needed. The result is that all spectrogram
        creating methods fail, because they cannot find a
        graphics backend. 
        
        In that case kwarg env is set to the environment of the 
        initiating process. At the start of this method this
        process' default environ is then set to match that
        of the initiating process.
        
        @param assignment: list of species/filename pairs
        @type assignment: [(str,str)]
        @param env: if provided, the environment of the
            parent process. If None, the current env
            is retained
        @type env: {str : Any}
        @param return_bool:
        @type return_bool:
        '''
       
        # During multiprocessing this method is
        # the target, i.e. the entry point for 
        # each child. In that case env will be 
        # the environment of the initiating process.
        # We adopt that environment for this new,
        # forked process as well:
        
        if env is not None:
            os.environ = env

        for species_name, fname in assignment:
            try:
                self.chop_one_audio_file(self.in_dir,
                                         species_name,
                                         fname,
                                         self.out_dir
                                         )
            except Exception as e:
                return_bool.value = False
                raise e
            
        return_bool.value = True
    
    #------------------------------------
    # chop_one_audio_file 
    #-------------------

    def chop_one_audio_file(self, in_dir, species, sample_name, out_dir, window_len = 5):
        """
        Generates window_len second sound file snippets
        and associated spectrograms from sound files of
        arbitrary length. 
        
        Performs a time shift on all the wav files in the 
        species directories. The shift is 'rolling' such that
        no information is lost.
    
        :param in_dir: directory of the audio file to chop 
        :type file_name: str
        :param species: the directory names of the species to 
            modify the wav files of. If species=None, all 
            subdirectories will be processed.
        :type species: {None | [str]}
        :param sample_name: basefile name of audio file to chop
        :type sample_name: str
        :param out_dir: root directory under which spectrogram
            and audio snippets will be saved (in different subdirs)
        :type out_dir: str
        """

        orig, sample_rate = librosa.load(os.path.join(in_dir, species, sample_name))
        length = int(librosa.get_duration(orig, sample_rate))
        for start_time in range(length - window_len):
            window_audio, sr = librosa.load(os.path.join(in_dir, species, sample_name),
                                      offset=start_time, duration=window_len)
            window_name = sample_name[:-len(".wav")] + '_sw-start' + str(start_time)
            outfile_spectro = os.path.join(out_dir, 
                                           'spectrograms/', 
                                           species,
                                           f"{window_name}.{'png'}")
            SoundProcessor.create_spectrogram(window_audio,sr,outfile_spectro)
            outfile_audio = os.path.join(out_dir, 
                                         'wav-files', 
                                         species, 
                                         f"{window_name}.{'wav'}")
            sf.write(outfile_audio, window_audio, sr)

    #------------------------------------
    # create_dest_dirs 
    #-------------------

    def create_dest_dirs(self, species_list):
        '''
        Creates all directories that will hold new 
        audio snippets and spectrograms for each species.
        For each directory: if dir exists:
           o if overwrite_freely is True, wipe the dir
           o else ask user. 
                If response is Yes, wipe the dir
                else raise FileExistsError
                
        @param species_list: names of species to process
        @type species_list: [str]
        @return: top level dirs for audio snippets and spectrograms
        @rtype: (str)
        @raise FileExistsError: if a dest dir exists and not allowed
            to wipe it.
        '''

        # Root dir of the two dirs that will hold new 
        # audio snippet and spectrogram files
        utils.create_folder(self.out_dir, overwrite_freely=self.overwrite_freely)

        # Below the rootP
        spectrogram_dir_path = os.path.join(self.out_dir,'spectrograms/')
        wav_dir_path = os.path.join(self.out_dir,'wav-files/')

        if not utils.create_folder(spectrogram_dir_path, overwrite_freely=self.overwrite_freely):
            raise FileExistsError(f"Target dir {spectrogram_dir_path} exists; aborting")
        if not utils.create_folder(wav_dir_path, overwrite_freely=self.overwrite_freely):
            raise FileExistsError(f"Target dir {spectrogram_dir_path} exists; aborting")
        
        # One dir each for the audio and spectrogram
        # snippets of one species:
        
        for species in species_list:
            species_spectros_dir = os.path.join(spectrogram_dir_path, species)
            if not utils.create_folder(species_spectros_dir,
                                       overwrite_freely=self.overwrite_freely):
                raise FileExistsError(f"Target dir {species_spectros_dir} exists; aborting")
            
            species_audio_dir = os.path.join(wav_dir_path, species)
            if not utils.create_folder(species_audio_dir,
                                       overwrite_freely=self.overwrite_freely):
                raise FileExistsError(f"Target dir {species_audio_dir} exists; aborting")

        return(wav_dir_path, spectrogram_dir_path)

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

        @param in_dir: root of species recordings
        @type in_dir: str
        @param num_workers: number of buckets into which to partition 
        @type num_workers: {int | None}
        @return: list of species name lists, and number of workers.
        @rtype: ([[int]], int)
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
            num_workers = round(num_cores * SoundChopper.MAX_PERC_OF_CORES_TO_USE  / 100)
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

        @param species_file_pairs:
        @type species_file_pairs:
        @param num_workers:
        @type num_workers:
        @return partitioning of the species_file_pairs tuples
        @rtype: [[(str, str)]]
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
    def run_workers(cls, args):
        '''
        Called by main to run the SoundChopper in
        multiple processes at once. Partitions the
        audio files to be processed; runs the chopping
        while giving visual progress on terminal.
        
        Prints success/failure of each worker. Then
        returns

        @param args: all arguments provided to argparse
        @type args: {str : Any}
        '''
        
        in_dir = args.input_dir
    
        # Get a list of lists of species names
        # to process. The list is computed such
        # that each worker has roughly the same
        # number of recordings to chop. We let
        # the method determine the number of workers
        # by using 80% of the available cores. 
        
        (worker_assignments, num_workers) = SoundChopper.compute_worker_assignments(
            in_dir,
            num_workers=args.workers)
    
        print(f"Distributing workload across {num_workers} workers.")
        # Assign each list of species to one worker:
        
        chopping_jobs = []
        for ass_num, assignment in enumerate(worker_assignments):
            chopper = SoundChopper(in_dir, 
                                   args.output_dir,
                                   overwrite_freely=args.overwrite_freely
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
                num_chopped_snippets = len(utils.find_in_dir_tree(SoundChopper.wav_dir_path))
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
       o Adds ability for SoundChopper instances to 
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
                           metavar='IN_DIR',
                           type=str,
                           help='the path to original directory with .wav files')
    parser.add_argument('output_dir',
                           metavar='OUT_DIR',
                           type=str,
                           help='the path to output directory to write new .wav/.png files')
    parser.add_argument('-w', '--workers',
                        type=int,
                        help='number of cores to use; default: 80% of available cores',
                        default=None)
    parser.add_argument('-y', '--overwrite_freely',
                        help='if set, overwrite existing out directories without asking; default: False',
                        action='store_true',
                        default=False
                        )

    # Execute the parse_args() method
    args = parser.parse_args()

    in_dir = args.input_dir
    # Input makes sense?
    if not os.path.exists(in_dir) or not os.path.isdir(in_dir):
        print(f"In directory {in_dir} does not exist, or is not a directory")
        sys.exit(1)

    if args.workers == 1:
        chopper = SoundChopper(in_dir, 
                               args.output_dir,
                               overwrite_freely=args.overwrite_freely
                               )
        chopper.chop_all()
    else:
        SoundChopper.run_workers(args)

    print("Done")
