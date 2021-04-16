#!/usr/bin/env python3

from _collections import OrderedDict
import argparse
import os,sys
import warnings

import librosa.display
from logging_service.logging_service import LoggingService
from matplotlib import MatplotlibDeprecationWarning

from data_augmentation import utils
from data_augmentation.sound_processor import SoundProcessor as aug
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import soundfile as sf


class SoundChopper:
    '''
    Processes directories of .wav or .mp3 files,
    chopping them into window_len second snippets.
    Each audio snippet is saved, and spectrograms
    are created for each.
    
    Assumes:

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

    '''
    
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
        
        (num_audios, num_spectros) = self.chop_all()
        #assert(num_audios == num_spectros)
        self.log.info(f"Created {num_audios} audio snippets and {num_spectros} spectrogram snippets")

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
        if self.specific_species is None:
            species_list = os.listdir(self.in_dir)
        else:
            species_list = self.specific_species
            
        # Root dir of the two dirs that will hold new 
        # audio snippet and spectrogram files
        utils.create_folder(self.out_dir, overwrite_freely=self.overwrite_freely)

        # Below the rootP
        spectrogram_dir_path = os.path.join(self.out_dir,'spectrograms/')
        wav_dir_path = os.path.join(self.out_dir,'wav-files/')
        utils.create_folder(spectrogram_dir_path, overwrite_freely=self.overwrite_freely)
        utils.create_folder(wav_dir_path, overwrite_freely=self.overwrite_freely)
        
        for species in species_list:
            # One dir each for the audio and spectrogram
            # snippets of one species:
            if (utils.create_folder(os.path.join(spectrogram_dir_path, species),
                                    overwrite_freely=self.overwrite_freely
                                    )
            and utils.create_folder(os.path.join(wav_dir_path, species),
                                    overwrite_freely=self.overwrite_freely)):
                audio_files = os.listdir(os.path.join(self.in_dir, species))
                num_files   = len(audio_files)
                for i, sample_name in enumerate(audio_files):
                    # Chop one audio file:
                    self.log.info(f"Chopping {species} audio {i}/{num_files}")
                    self.chop_one_audio_file(self.in_dir, species, sample_name, self.out_dir)
                self.num_chopped += num_files
                
            else:
                print(f"Skipping audio chopping for {species}")

        num_spectros = utils.find_in_dir_tree(spectrogram_dir_path, pattern='*.png')
        num_audios   = utils.find_in_dir_tree(wav_dir_path, pattern='*.wav')
        return (num_audios, num_spectros)
        
    #------------------------------------
    # create_spectrogram 
    #-------------------

    def create_spectrogram(self, sample_name, sample, sr, out_dir, n_mels=128):
        # Use bandpass filter for audio before converting to spectrogram
        audio = aug.filter_bird(sample, sr)
        mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels)
        # create a logarithmic mel spectrogram
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # create an image of the spectrogram and save it as file
        fig = plt.figure(figsize=(6.07, 2.02))
        librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', cmap='gray_r')
        plt.tight_layout()
        plt.axis('off')
        spectrogramfile = os.path.join(out_dir, sample_name + '.png')

        # Workaround for Exception in Tkinter callback
        fig.canvas.start_event_loop(sys.float_info.min) 
        #*****
        plt.savefig(spectrogramfile, dpi=100, bbox_inches='tight', pad_inches=0, format='png', facecolor='none')
        plt.close()

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
            window, sr = librosa.load(os.path.join(in_dir, species, sample_name),
                                      offset=start_time, duration=window_len)
            window_name = sample_name[:-len(".wav")] + '_sw-start' + str(start_time)
            self.create_spectrogram(window_name, 
                                    window, 
                                    sr, 
                                    os.path.join(out_dir, 'spectrograms/', species)
                                    )
            sf.write(os.path.join(out_dir, 'wav-files', species, window_name + '.wav'), window, sr)


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
        
        sample_size_distrib = OrderedDict({})
        for _dir_name, subdir_list, _file_list in os.walk(in_dir):
            for species_name in subdir_list:
                sample_size_distrib[species_name] = (len(os.listdir(os.path.join(in_dir, species_name))))
            break 
        
        num_cores = mp.cpu_count()
        # Use 80% of the cores:
        if num_workers is None:
            num_workers = round(num_cores * 80 /100)
        elif num_workers > num_cores:
            # Limit pool size to number of cores:
            num_workers = num_cores

        sample_sizes = sample_size_distrib.values()
        
        # Get list of sample sizes for birds, 
        # roughly equally distributed across the
        # workers. Example, asssuming 5 species
        # with 10, 3, 6, 1, and 6 number of recordings,
        # respectively, we may get:
        #
        #   [[10, 3], [6,1,6],]
        
        worker_assignment_sizes = cls.partition_list(sample_sizes, num_workers)
        
        # Get the corresponding species names.
        # We can do that, b/c partition_list preserves
        # the order of class sizes we pass in, and
        # b/c we use an ordered dict:
        
        species_lists = []
        species_names = list(sample_size_distrib.keys())
        name_idx = 0
        for num_samples_bucket in worker_assignment_sizes:
            species_lists.append(species_names[name_idx:name_idx+len(num_samples_bucket)])
            name_idx = name_idx+len(num_samples_bucket)
        
        return species_lists, num_workers 

    #------------------------------------
    # partition_list 
    #-------------------

    @classmethod
    def partition_list(cls, a, k):
        '''
        Takes a list 'a' of integers, and a bucket size 'k'.
        Partitions the list into k sublists such as the sum
        within the sublists are as equal as possible. Order
        is preserved.
        
        [converted to Pyhon 3 from accepted answer in 
         https://stackoverflow.com/questions/35517051/split-a-list-of-numbers-into-n-chunks-such-that-the-chunks-have-close-to-equal
        ]
        
        Examples:
            l = [1, 6, 2, 3, 4, 1, 7, 6, 4]
            
            best = SoundChopper.partition_list(l, 1)
            assert(best == [[1, 6, 2, 3, 4, 1, 7, 6, 4]])
            best = SoundChopper.partition_list(l, 2)
            assert(best == [[1, 6, 2, 3, 4, 1], [7, 6, 4]])
        
            best = SoundChopper.partition_list(l, 3)
            assert(best == [[1, 6, 2, 3], [4, 1, 7], [6, 4]])
        
            best = SoundChopper.partition_list(l, 4)
            assert(best == [[1, 6], [2, 3, 4, 1], [7], [6, 4]])
        
        @param a: list of integers to partition
        @type a: [int]
        @param k: number of buckets
        @type k: int
        @return: list of buckets
        @rtype: [[int]]
        '''
        if k <= 1: return [a]
        if k >= len(a): return [[x] for x in a]
        partition_between = [(i+1)*len(a)/k for i in range(k-1)]
        average_height = float(sum(a))/k
        best_score = None
        best_partitions = None
        count = 0
        while True:
            starts = [0]+partition_between
            ends = partition_between+[len(a)]
            partitions = [a[round(starts[i]):round(ends[i])] for i in range(k)]
            heights = [*map(sum, partitions)]
            abs_height_diffs = []
            for height in heights:
                abs_height_diffs.append(abs(average_height - height))            
            worst_partition_index = abs_height_diffs.index(max(abs_height_diffs))
            worst_height_diff = average_height - heights[worst_partition_index]
            if best_score is None or abs(worst_height_diff) < best_score:
                best_score = abs(worst_height_diff)
                best_partitions = partitions
                no_improvements_count = 0
            else:
                no_improvements_count += 1
            if worst_height_diff == 0 or no_improvements_count > 5 or count > 100:
                return best_partitions
            count += 1
            move = -1 if worst_height_diff < 0 else 1
            bound_to_move = 0 if worst_partition_index == 0\
                            else k-2 if worst_partition_index == k-1\
                            else worst_partition_index-1 if (worst_height_diff < 0) ^ (heights[worst_partition_index-1] > heights[worst_partition_index+1])\
                            else worst_partition_index
            direction = -1 if bound_to_move < worst_partition_index else 1
            partition_between[bound_to_move] += move * direction
            
        return best_partitions

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
    # parser.add_argument('-s', '--species',
    #                     type=str,
    #                     nargs='+',
    #                     help='repeatable: specific species to use sliding window on',
    #                     default=None)
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
    
    # Get a list of lists of species names
    # to process. The list is computed such
    # that each worker has roughly the same
    # number of recordings to chop. We let
    # the method determine the number of workers
    # by using 80% of the available cores. 
    
    (worker_assignments, num_workers) = SoundChopper.compute_worker_assignments(in_dir)

    #****def create_sound_chopper(*args, **kwargs):
    print(f"Distributing workload across {num_workers} workers.")
    # Assign each list of species to one worker:
    
    chopping_jobs = []
    for species_to_process in worker_assignments:
        kwargs = {'specific_species' : species_to_process,
                  'overwrite_freely' : args.overwrite_freely
                  }
        job = mp.Process(target=SoundChopper,
                         args=(in_dir, args.output_dir),
                         kwargs=kwargs,
                         name=species_to_process.join(',')
                         )
        chopping_jobs.append(job)
        print(f"Starting chops for {job.name}")
        job.start()
    
    for job in chopping_jobs:
        job.join()
        res = "OK" if job.exitcode == 1 else "Error"
        print(f"Chops of {job.name}: {res}")

    print("Done")