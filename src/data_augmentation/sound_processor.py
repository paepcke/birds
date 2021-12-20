import datetime
import gzip
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import uuid
import warnings

from PIL import Image
from PIL.PngImagePlugin import PngImageFile, PngInfo
import librosa
from logging_service import LoggingService
from scipy.signal import butter
from scipy.signal import lfilter
import skimage.io
import soundfile

from birdsong.utils.utilities import FileUtils
from data_augmentation.multiprocess_runner import MultiProcessRunner, Task
from data_augmentation.utils import Interval, Utils
import numpy as np
import pandas as pd


# ------------------------ Exception Classes -----------------
class AudioLoadException(Exception):
    '''
    Raised when errors occur while loading 
    audio files. Property 'audio_file' is
    available in all instances of this exception.
    The other_msg property, if not None will contain
    the original error message in case the error was
    due to some OS or other problem.
    
    '''
    def __init__(self, msg, aud_file, other_msg=None):
        '''
        
        :param msg: main error message
        :type msg: str
        :param aud_file: path to problematic audio file
        :type aud_file: str
        :param other_msg: optionally communicates more info
            based on an originally triggering error
        :type other_msg: {str | None}
        '''
        super().__init__(msg)
        self.audio_file = aud_file
        self.other_msg  = other_msg

# --------------------- Sound Processor ---------------

class SoundProcessor:
    '''
    Facilities to modify audio files and spectrograms.
    All methods are class methods. So no instances
    are made of this class.
    '''

    # Get a Python logger that is
    # common to all modules in this
    # package:

    log = LoggingService()

    SPECTRO_WIN_HEIGHT=224
    '''Number of frequency bands in spectrogram'''
    
    # Pick out the 00:03:34.60 from
    #   '  Duration: 00:03:34.60, start: 0.025056, bitrate: 223 kb/s'
    DURATION_PATTERN = re.compile(b"[\s]*Duration: ([0-9:.]*).*")
    # Pick out the 44100 from:
    #    'Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 211 kb/s'
    
    # The '?' after the ".*" makes the search non-greedy
    STREAM_PATTERN   = re.compile(b"[\s]*Stream.*?([0-9]+) Hz.*")

    #------------------------------------
    # Constructor Stub 
    #-------------------
    
    def __init__(self):
        raise NotImplementedError("Class SoundProcessor is not intended for instantiation")

    # ------------------ Operations on Sound Files --------------

    #------------------------------------
    # add_background
    #-------------------

    @classmethod
    def add_background(cls, 
                       file_name, 
                       noise_sources, 
                       out_dir, 
                       len_noise_to_add=5.0):
        '''
        Takes an absolute file path, and noise source
        files. Randomly selects a noise file to overlay onto 
        the given sound file (wind, rain, another bird, etc.).
        
        Creates a new sounfile with the original audio and
        the noise overlaid. A file name is created
        for the sample. It is composed of elements such 
        as the nature and duration of the noise. The new
        audio is saved to that file.

        In addition to randomly selecting the noise file,
        the position in the file to be modified where the
        noise is overlaid is chosen randomly.

        :param file_name: absolute path to sound file
        :type file_name: str
        :param noise_sources: either absolute path to directory of
            noise files, or path to individual noise file to use,
            or a list of directories/files
        :type noise_sources: str
        :param out_dir: destination directory of new audio file
        :type out_dir: str
        :param len_noise_to_add: how much of a noise snippet
            to overlay (seconds)
        :type len_noise_to_add: float
        :return: full path of new audio file, and full path
            of the noise file used
        :rtype: (str, str)
        '''
        
        len_noise_to_add = float(len_noise_to_add)

        # Make a dict:
        #    noise-src-dir : [noise-fname1, noise-fname2,...]
        #    None          : [full-noise-file_path1, full-noise-file_path2, ...]
        
        # (possibly) with a key of None for collection of 
        # individually specified files in noise_sources:
        
        noise_src_dict = FileUtils.find_files_by_type(noise_sources, 
                                                      FileUtils.AUDIO_EXTENSIONS)

        # To pick a random noise file, first
        # pick a random directory (i.e. key) from
        # the noise_src_dict:
        key_dir_idx = random.randint(0, len(noise_src_dict) - 1)
        key = list(noise_src_dict)[key_dir_idx]
        # Next, a random file within the list of files
        file_list = noise_src_dict[key]
        file_idx    = random.randint(0, len(file_list) - 1)
        
        # Top level noise files (i.e. ones in argument
        # noise_sources that are paths to files, not directories),
        # are under the None key in noise_src_dir.
        
        full_background_path = file_list[file_idx] \
            if key is None \
            else os.path.join(key, file_list[file_idx])

        # Make shortened paths of file and background_name just for log msgs
        short_file_nm     = FileUtils.ellipsed_file_path(file_name)
        short_backgrnd_nm = FileUtils.ellipsed_file_path(full_background_path)
        cls.log.info(f"Adding {short_backgrnd_nm} to {short_file_nm}.")
        
        # We will be working with 1 second as the smallest unit of time
        # load all of both wav files and determine the length of each
        noise, noise_sr = SoundProcessor.load_audio(full_background_path)
        orig_recording, orig_sr = SoundProcessor.load_audio(file_name)
    
        new_sr = math.gcd(noise_sr, orig_sr)
        if noise_sr != orig_sr:
            # Resample both noise and orig records so that they have same sample rate
            cls.log.info(f"Resampling: {short_backgrnd_nm} and {short_file_nm}")
            noise = librosa.resample(noise, noise_sr, new_sr)
            orig_recording = librosa.resample(orig_recording, orig_sr, new_sr)
            # input("ready?")
    
        noise_duration = librosa.get_duration(noise, noise_sr)
        if noise_duration < len_noise_to_add:
            cls.log.info(f"Duration:{noise_duration} < len_noise_to_add:{len_noise_to_add}. Will only add {noise_duration}s of noise")
            samples_per_segment = len(noise)
        elif noise_duration >= len_noise_to_add:  # randomly choose noise segment
            samples_per_segment = int(new_sr * len_noise_to_add)  # this is the number of samples per 5 seconds
            # Place noise randomly:
            subsegment_start = random.randint(0, len(noise) - samples_per_segment)
            noise = noise[subsegment_start: subsegment_start + samples_per_segment]
        cls.log.info(f"len(noise) after random segment: {len(noise)}; noise duration: {len(noise)/new_sr}")
    
    
        orig_duration = librosa.core.get_duration(orig_recording, orig_sr)
        # if orig_recording is shorter than the noise we want to add, just add 5% noise
        if orig_duration < len_noise_to_add:
            cls.log.info(f"Recording: {file_name} was shorter than len_noise_to_add. Adding 5% of recording len worth of noise.")
            new_noise_len = orig_duration * 0.05
            noise = noise[:int(new_noise_len * new_sr)]
        noise_start_loc = random.randint(0, len(orig_recording) - samples_per_segment)
        cls.log.info(f"Inserting noise starting at {noise_start_loc/new_sr} seconds.")
        # split original into three parts: before_noise, during_noise, after_noise
        before_noise = orig_recording[:noise_start_loc]
        during_noise = orig_recording[noise_start_loc: noise_start_loc + samples_per_segment]
        after_noise = orig_recording[noise_start_loc + samples_per_segment:]
    
        assert len(during_noise) == len(noise)
    
        # Compute the overlay: to the chosen portion of the
        # original file, add a fraction of the noise:
        segment_with_noise = during_noise + cls.noise_multiplier(orig_recording, noise) * noise
        first_half   = np.concatenate((before_noise, segment_with_noise))
        new_sample   = np.concatenate((first_half, after_noise)) # what i think it should be
        new_duration = librosa.get_duration(new_sample, float(new_sr))
    
        assert new_duration == orig_duration
        # File name w/o extension:
        sample_file_stem = Path(file_name).stem
        noise_file_stem  = Path(full_background_path).stem
        noise_dur = str(int(noise_start_loc/new_sr * 1000))
        file_name= f"{sample_file_stem}-{noise_file_stem}_bgd{noise_dur}ms_{uuid.uuid1().hex}.wav"
        
        # Ensure that the fname doesn't exist:
        uniq_fname = Utils.unique_fname(out_dir, file_name)
        out_path = os.path.join(out_dir, uniq_fname)
        
        soundfile.write(out_path, new_sample, new_sr)
        return out_path, full_background_path

    #------------------------------------
    # change_all_volumes 
    #-------------------

    @classmethod
    def change_all_volumes(cls, in_dir, out_dir, species=None):
        """
        Adjusts the volume of all the wav files in the species directory.
    
        :param in_dir: the path to the directory to fetch samples from
        :type in_dir: str
        :param out_dir: the path to the directory to save the new files to
        :type out_dir: str
        :param species: the directory names of the species to modify the wav 
            files of. If species=None, all subdirectories will be used.
        :type species: str
        """
        for species_dir in os.listdir(in_dir):
            if species is None or species_dir in species:
                full_species_dir = os.path.join(in_dir, species_dir)
                for sample_file_nm in os.listdir(full_species_dir):
                    sample_path = os.path.join(in_dir, sample_file_nm)
                    cls.change_sample_volume(sample_path, out_dir)

    #------------------------------------
    # change_sample_volume 
    #-------------------

    @classmethod
    def change_sample_volume(cls, sample_path, out_dir):
        '''
        Randomly changes an audio clip's volume, and writes
        the new audio file to out_dir

        :param sample_path: full path to sample
        :type sample_path: src
        :param out_dir: destination directory
        :type out_dir: src
        :return full path to the new audio file
        :rtype str
        '''
        y0, sample_rate0 = SoundProcessor.load_audio(sample_path)

        # Adjust the volume
        factor = random.randrange(-12, 12, 1)
        
        cls.log.info(f"Changing volume of {sample_path} by factor {factor}.")

        y1 = y0 * (10 ** (factor / 20))

        # Output the new wav data to a file
        # Just the foofile part of /home/me/foofile.mp3:
        sample_root = Path(sample_path).stem
        new_sample_fname = f"{sample_root}-volume{factor}_{uuid.uuid1().hex}.wav"
        out_file = os.path.join(out_dir, new_sample_fname)
        soundfile.write(out_file, y1, sample_rate0)
        return out_file

    #------------------------------------
    # time_shift 
    #-------------------

    @classmethod
    def time_shift(cls, file_name, out_dir):
        """
        Performs a time shift on all the wav files in the 
        species directory. The shift is 'rolling' such that
        no information is lost: a random-sized snippet of
        the audio is moved from the start of the clip to
        its end.
    
        :param file_name: full path to audio file
        :type file_name: str
        :param out_dir: the path to the directory to save the new file to
        :type out_dir: str
        :return full path to the new audio file
        :rtype str
        :raise AssertionError when before & after lengths disagree
        """
        y, sample_rate = SoundProcessor.load_audio(file_name)
        length = librosa.get_duration(y, sample_rate)  # returns length in seconds
        # shifts the recording by a random amount between 0 and length of recording by a multiple of 10 ms
        amount = random.randrange(0, int(length)*10, 1)/10  # shift is in seconds
    
        # Create two seperate sections of the audio
        # Snippet after the shift amount:
        y0, sample_rate0 = SoundProcessor.load_audio(file_name, offset=amount)
        # Snippet before the shift amount:
        y1, _sample_rate1 = SoundProcessor.load_audio(file_name, duration=amount)
    
        # Append the before-snippet to the 
        # end of the after-snippet: 
        y2 = np.append(y0, y1)
        # print(f"Amount: {amount}ms")
        assert len(y) == len(y2), f"Before-len: {len(y)}; after-len: {len(y2)}"

        # Output the new wav data to a file
        # Get just the 'foo' part of '/blue/red/foo.mp3':
        file_stem = Path(file_name).stem
        aug_sample_name = f"{file_stem}-shift{str(int(amount * 1000))}ms_{uuid.uuid1().hex}.wav"
        out_path = os.path.join(out_dir, aug_sample_name)
        soundfile.write(out_path, y2, sample_rate0)
        return out_path


    #------------------------------------
    # find_recording_lengths
    #-------------------
    
    @classmethod
    def find_recording_lengths(cls, recordings_src):
        '''
        Given a directory with .wav or .mp3 recordings,
        return a dataframe row labels (index) being 
        a recording file name, and two columns:
        
            o 'recording_length_secs' the duration in seconds
            o 'recording_length_hhs_mins_secs' in 'hrs:mins:secs'
            
        Like:
                     recording_length_secs recording_length_hhs_mins_secs
            fname1           12                  0:00:12
            fname2           62                  0:01:02 
            

        :param recordings_src: directory containing recordings
        :type recordings_src: str
        :return dataframe with duration information for each
            file
        :rtype pd.DataFrame
        '''
        res_dict = {}

        # Distinguish between recordings_src being an
        # individual file vs. a directory of recordings:
        
        if os.path.isdir(recordings_src):
            recordings_src = Utils.listdir_abs(recordings_src)
        else:
            # Just get duration of an individual sound file:
            recordings_src = [recordings_src]  
            
        for recording in recordings_src:
            # Skip non-audio files:
            if not Path(recording).suffix in FileUtils.AUDIO_EXTENSIONS:
                continue
            dur_sr_dict = SoundProcessor.soundfile_metadata(recording)
            dur_delta   = dur_sr_dict['duration']
            duration = dur_delta.seconds + 10**-6 * dur_delta.microseconds
            res_dict[Path(recording).name] = duration
        res_df = pd.DataFrame.from_dict(res_dict, 
                                        orient='index', 
                                        columns=['recording_length_secs']
                                        )
        res_df['recording_length_hhs_mins_secs'] = \
            res_df['recording_length_secs'].map(lambda el: 
                                                str(datetime.timedelta(seconds=el))
                                                )
        return res_df

    #------------------------------------
    # recording_len
    #-------------------
    
    @classmethod
    def recording_len(cls, audio, sr):
        '''
        Given an audio clip as an np array and
        its sampling rate, return length in seconds
        
        :param audio: audio data 
        :type audio: np.array[float]
        :param sr: sampling rate
        :type sr: int
        :return recording time
        :rtype float
        '''
        return len(audio) / sr

    #------------------------------------
    # time_resolution
    #-------------------
    
    @classmethod
    def time_resolution(cls, audio, sr):
        '''
        Return the time width between each audio sample. 

        :param audio: sound
        :type audio: np.ndarray
        :param sr: sample rate
        :type sr: float
        :return time resolution (seconds)
        :rtype float
        '''
        return sr/len(audio)

    #------------------------------------
    # find_total_recording_length
    #-------------------

    @classmethod
    def find_total_recording_length(cls, 
                                    species_dir_path,
                                    save_recording_durations=None
                                    ):
        '''
        Given a directory with .wav or .mp3 recordings,
        return the sum of recording lengths.
        
        If save_recording_durations a directory path, 
        write a gzipped json file of a dataframe that
        lists the durations of each file.
        
        If save_recording_durations is absent or False, this
        detailed information is not saved.

        :param species_dir_path: directory containing recordings
        :type species_dir_path: str
        :param save_recording_durations: directory where to place
            json-encoded, zipped files of dataframes with recording 
            lengths for each file. If None, only the duration sums
            are saved.
        :type save_recording_durations: {None | str | True}
        :return sum of recording lengths
        :rtype float
        '''
        df_fname_secs = cls.find_recording_lengths(species_dir_path)
        # Are we to save all individual audio durations?
        if save_recording_durations is not None:
            if not os.path.exists(save_recording_durations):
                os.makedirs(save_recording_durations, exist_ok=True)
            # Save the df as JSON, and compress it:
            species = Path(species_dir_path).stem
            fname = os.path.join(save_recording_durations, f"{species}_manifest.json")
            
            cls.log.info(f"Saving {species} recording durations to {save_recording_durations}") 
            df_fname_secs.to_json(fname)
            # Compress the manifest file, and 
            # delete the uncompressed version:
            cls.gzip_file(fname, delete=True)

        total_duration = df_fname_secs['recording_length_secs'].sum()
        return total_duration



    #------------------------------------
    # recording_lengths_by_species
    #-------------------

    @classmethod
    def recording_lengths_by_species(cls, 
                                     species_root, 
                                     num_workers=None,
                                     save_recording_durations=None):
        '''
        Given directory that contains subidrectories with
        .wav or .mp3 recordings, return a dataframe with 
        recording length information:
        
                         total_recording_length (secs)   duration (hrs:mins:secs)
            species1            10.5                        0:10:30
            species2             2.0                        0:02:00
               ...              ...
               
        Rows will be alpha-sorted by species_dir.
        Work is done in parallel
        
        If save_recording_durations is a directory, then
        intermediate files are saved beyond the one above.
        Each such additional file is named <species_manifest>.json.gz, and
        contains a dataframe like:
        
                       'seconds'
               file1   10.3
               file2    4.2
                ...     ...
                
        If save_recording_durations has value True:
        a standard directory location is chosen:
        
          <parent-of-species_dir_path>/Audio_Manifest_<species_dir_path-name>
          
        Example:
          given   /foo/bar
         manifest directory will be /foo/Audio_Manifest_bar
        
        If save_recording_durations is absent or False, this
        detailed information is not saved.

        :param species_root: root of the recording subdirectories
        :type species_root: str
        :param num_workers: maximum number of CPUs to use. Default:
            Utils.MAX_PERC_OF_CORES_TO_USE
        :type num_workers: {None | int}
        :param save_recording_durations: directory where to place
            json-encoded, zipped files of dataframes with recording 
            lengths for each file. If None, only the duration sums
            are saved.
        :type save_recording_durations: {None | str | True}
        :return sum of recording length of all recodings
            of each species_dir. If no recordings found, returns
            None
        :rtype {pd.DataFrame | None}
        :raise RuntimeError if any of the species
            tallies fails
        '''
        
        # If save_recording_durations is set to True,
        # place detail json files in Audio_Manifest_<species_root-name>
        # of species_root's parent dir:
        if save_recording_durations == True:
            save_recording_durations = FileUtils.make_manifest_dir_name(species_root)
        
        num_samples_in = {} # initialize dict - usage num_samples_in['CORALT_S'] = 64
        
        # Just for progress reports: get number
        # of species dirs to examine:
        num_species = sum([1 
                           for dir_entry
                           in os.scandir(species_root)
                           if dir_entry.is_dir()
                           ])
        
        cls.log.info({f"Analyzing metadata for audio files of {num_species} species..."})

        inventory_tasks = []
        species_list    = []
        for species_dir in Utils.listdir_abs(species_root):
            # Skip file names that are at top level;
            # i.e. only consider files under the species
            # subdirs:
            if not os.path.isdir(species_dir):
                continue
            species = Path(species_dir).stem
            species_list.append(species)
            inventory_task = Task(f"rec_inventory_{species}",
                                  cls.inventory_one_species,
                                  species_dir,
                                  save_recording_durations
                                  )
            inventory_tasks.append(inventory_task) 

        mp_runner = MultiProcessRunner(inventory_tasks, num_workers=num_workers)

        # Check whether any of the results is an 
        # exception: a result dict that reports an
        # error will have two keys: 'Exception', and
        # 'Traceback'. Get a list of such dicts:
        exceptions = [res_dict
                      for res_dict
                      in mp_runner.results
                      if 'Exception' in list(res_dict.keys())
                      ]
        # If at least one err found, log 
        # it, and raise a RuntimeError: 
        if len(exceptions) > 0:
            # Grab the first one and log it such that 
            # it includes the traceback:
            exc_entry = exceptions[0]
            msg = f"First of {len(exceptions)} exception(s):\n{exc_entry['Traceback']}"
            cls.log.err(msg)
            raise RuntimeError(msg)

        # Sigh of relief! All worked.
        
        # Result for each species subdir will be a one-entry dict:
        #                 total_recording_length
        #    species_name : recording-duration-float
        for res_dict in mp_runner.results:
            for maybe_species, maybe_duration in res_dict.items():
                if maybe_species in species_list:
                    num_samples_in[maybe_species] = {"total_recording_length (secs)": maybe_duration}

        if len(num_samples_in) == 0:
            return None 

        # A an hrs:mins:secs column:
        df = pd.DataFrame.from_dict(num_samples_in, orient='index').sort_index()
        
        df['duration (hrs:mins:secs)'] = \
            df['total_recording_length (secs)'].map(lambda el: 
                                             str(datetime.timedelta(seconds=el)))
            
        
        return df.sort_index()

    #------------------------------------
    # extract_clip
    #-------------------
    
    @classmethod
    def extract_clip(cls, audio, start, stop, sr=22050):
        '''
        Given audio as an np array, return a slice
        from the audio array, which starts at time
        start, and ends at time stop.
        
        :param audio: audio array as returned by
            load_audio()
        :type audio: np.array[float]
        :param start: start in seconds
        :type start: float
        :param stop: stop points in seconds
        :type stop: float
        :param sr: sample rate. If None, and loading
            from file, use file's native sr.
        :type sr: int
        :returned audio clip
        :rtype np.array[float]
        :raise ValueError if time segment exceeds
            length of given audio, or is negative
        '''
        
        if start < 0 or stop < start:
            raise ValueError(f"Start and stop must lie within audio clip, not {start}, {stop}")

        num_samples = len(audio)
        full_time_len = num_samples / sr
        if stop > full_time_len:
            raise ValueError(f"Stop time is gt audio length ({full_time_len}, {stop})")
        start_sample = int(sr * start)
        end_sample   = int(sr * stop)
        clip = audio[start_sample:end_sample]
        return clip 

    #------------------------------------
    # inventory_one_species
    #-------------------
    
    @classmethod
    def inventory_one_species(cls, species_dir, save_recording_durations):
        '''
        Given directory of recordings of one species,
        find the duration of each audio file in the
        directory. Add them up, and return a one-entry
        dict {<species-name> : <rec-length-in-seconds>}
        
        If save_recording_durations is not None, it must
        be a directory where all the individual audio file recording 
        lengths will be saved along the way as a compressed json file.
        
        If save_recording_durations is absent or False, this
        detailed information is not saved.

        :param species_dir: root of the recordings to tally
        :type species_dir: str
        :param save_recording_durations: directory where to
            place compressed json export of dataframe with
            the individual recording durations.
        :type save_recording_durations: {None | str | True}
        '''
        
        species = Path(species_dir).stem
        # For progress reports, get number of
        # audio files in this subdir:
        num_aud_files = sum([1
                             for dir_entry
                             in os.scandir(species_dir)
                             if dir_entry.is_file() and \
                                Path(dir_entry).suffix in FileUtils.AUDIO_EXTENSIONS
                             ])
        cls.log.info(f"Analyzing {num_aud_files} audio files of species {species}")
        rec_len = cls.find_total_recording_length(species_dir, save_recording_durations)
        return {species : rec_len}

    # --------------- Operations on Spectrograms Files --------------

    #------------------------------------
    # create_spectrogram 
    #-------------------

    @classmethod
    def create_spectrogram(cls,
                           audio_sample, 
                           sr, 
                           outfile=None, 
                           n_mels=None, # Default: SoundProcessor.SPECTRO_WIN_HEIGHT
                           info=None):
        '''
        Create and save a spectrogram from an audio 
        sample. Bandpass filter is applied, Mel scale is used,
        and power is converted to decibels.
        
        For spectrograms as the Raven labeling program produces
        with its default settings, see SignalAnalyzer.raven_spectrogram()
        in file signal_analysis.py.
        
        If outfile is None, the spectrogram is returned
        as a an np.ndarray. Else the spectrogram is converted
        to an image and saved.
        
        If outfile is a filename:
        
        The sampling rate (sr) and time duration in fractional
        seconds is added as metadata in the .png file under keys 
        "sr" and "duration". Additional info may be included in 
        the .png if info is a dict of key/value pairs.
        
        The outfile's parent directories are created, if necessary.
        
        Retrieving the metadata can be done via SoundProcessor.load_spectrogram(),
        or any other PNG reading software that handles the PNG specification.
        To print the information from the command line, use 
        <proj-root>src/data_augmentation/list_png_metadata.py 
         
        :param audio_sample: audio
        :type audio_sample: np.array
        :param sr: sample rate
        :type sr: int
        :param outfile: where to store the result
        :type outfile: str
        :param n_mels: number of mel scale bands 
        :type n_mels: int
        :param info: if provided,  a dict of information to
            store as text-only key/value pairs in the png.
            Retrieve info via SoundProcessor.load_spectrogram()
            or other standard PNG reading software that supports
            PNG metadata
        :type info: {str : str}
        '''
        
        if info is not None and type(info) != dict:
            raise TypeError(f"If provided, info must be a dict, not {type(info)}")
        
        if n_mels is None:
            n_mels = SoundProcessor.SPECTRO_WIN_HEIGHT
        
        # Use bandpass filter for audio before converting to spectrogram
        audio = SoundProcessor.filter_bird(audio_sample, sr)
        mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels)
        # create a logarithmic mel spectrogram
        log_mel = librosa.power_to_db(mel, ref=np.max)

        if outfile is None:
            return log_mel

        # Create an image of the spectrogram and save it as file
        img = cls.scale_minmax(log_mel, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image
        img = 255-img # invert. make black==more energy

        # Save as PNG, including sampling rate and
        # (superfluously) duration in seconds:
        duration = round(len(audio)/sr, 1)
        
        # Create metadata to include in the 
        # spectrogram .png file:
        
        metadata = PngInfo()
        metadata.add_text("sr", str(sr))
        metadata.add_text("duration", str(duration))
        if info is not None:
            for key, val in info.items():
                metadata.add_text(key, str(val))

        outdir = os.path.dirname(outfile)
        os.makedirs(outdir, exist_ok=True)
        skimage.io.imsave(outfile, img, pnginfo=metadata)


    #------------------------------------
    # add_time_freq_masks
    #-------------------

    @classmethod
    def add_time_freq_masks(cls, file_name, in_dir, out_dir=None):
        '''
        Performs Frequency and Time Masking for Spectrograms. The
        masks add horizontal (freq), and vertical (time) masks.

        :param file_name: name of spectrogram file without parents
        :type file_name: str
        :param in_dir: directory in which the spectrogram resides
        :type in_dir: str
        :param out_dir: optionally: destination directory. If None,
            augmented copy is written to in_dir
        :type out_dir: {None | str}
        :return: full path of augmented spectrogram
        :rtype: str
        '''
        cls.log.info(f"Adding masks to {file_name}")
        # print(file_name)
        orig_spectrogram = np.asarray(Image.open(os.path.join(in_dir, file_name)))
        freq_masked, freq_name = cls.freq_mask(orig_spectrogram, num_masks=2)
        masked_spectrogram, time_name = cls.time_mask(freq_masked, num_masks=2)
        img = Image.fromarray(masked_spectrogram)
        fpath = Path(file_name)
        new_file_name = f"{fpath.stem}-{freq_name}-{time_name}.png"
        if out_dir is None:
            # Write masked spectrogram to the same dir as original:
            outpath = Path.joinpath(fpath.parent, new_file_name)
        else:
            outpath = Path.joinpath(out_dir, new_file_name)
        img.save(outpath)
        return outpath

    #------------------------------------
    # add_noise
    #-------------------

    @classmethod
    def add_noise(cls, spectrogram, std=1.0):
        '''
        Reads a spectrogram from a file, adds uniform noise 'jitter' to it,
        and writes the result back to a file.

        :param spectrogram: the spectrogram to modify
        :type spectrogram: np.array
        :param std: standard deviation of the noise; default: 1.0
        :type std: {int | float}
        :return: full path of augmented spectrogram
        :rtype: str
        '''
        
        if std < 0:
            raise ValueError(f"Standard deviation must be non-negative; was {std}")
        
        cls.log.info(f"Adding uniform noise to spectrogram")
        new_spectro = spectrogram.copy()
        spectro_noised = cls.random_noise(new_spectro, std=std)
        return spectro_noised 

    #------------------------------------
    # freq_mask 
    #-------------------

    # Functions below are from https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb
    # Functions edited to support np arrays
    @classmethod
    def freq_mask(cls, spec, max_height=40, num_masks=1, replace_with_zero=False):
        '''
        Takes a spectrogram array, and returns a new
        spectrogram array with a frequency mask added. 
        Also returns a suggested file name based on
        the randomly chosen mask parameters.
        
        :param spec: the spectrogram
        :type spec: np.array
        :param max_height: max height of horizontal stripes
        :type max_height: int
        :param num_masks: how many masks to add
        :type num_masks: int
        :param replace_with_zero: if True, replaces the existing
            values with zero. Else replaces them with the mean
            of the entire image
        :type replace_with_zero: bool
        :returns a new array, and a suggested file name
        :rtype (np.array, str)
        '''
        cloned = spec.copy()
        num_mel_channels = cloned.shape[0]
        if max_height >= num_mel_channels:
            # Ensure that random choice of 
            # mask height is at least 1 below:
            max_height = num_mel_channels - 1

        for _i in range(0, num_masks):
            # Choose random stripe height within given limit,
            # but at least 1:
            mask_height = random.randrange(1, max_height)
            # Random choice of where to place the stripe:
            f_zero = random.randrange(0, num_mel_channels - mask_height)
    
            # avoids randrange error if values are equal and range is empty
            #if (f_zero == f_zero + mask_height): continue
    
            mask_end = f_zero + mask_height
            if (replace_with_zero): 
                cloned[f_zero:mask_end,:] = 0
            else:
                cloned[f_zero:mask_end,:] = cloned.mean()
        return cloned, f"-fmask{int(f_zero)}_{int(mask_end)}"

    #------------------------------------
    # time_mask 
    #-------------------

    @classmethod
    def time_mask(cls, spec, max_width=20, num_masks=1, replace_with_zero=False):
        '''
        Takes a spectrogram array, and returns a new
        spectrogram array with a time mask added. 
        Also returns a suggested file name based on
        the randomly chosen time mask parameters.
        
        :param spec: the spectrogram
        :type spec: np.array
        :param max_width: width of vertical stripes
        :type max_width: int
        :param num_masks: how many masks to add
        :type num_masks: int
        :param replace_with_zero: if True, replaces the existing
            values with zero. Else replaces them with the mean
            of the entire image
        :type replace_with_zero: bool
        :returns a new array, and a suggested file name
        :rtype (np.array, str)
        '''
        cloned = spec.copy()
        min_width   = 1 # Time slice
        len_spectro = cloned.shape[1]
        # Is width of stripe ge width of spectro?
        if max_width >= len_spectro:
            max_width = len_spectro - 1 
            
        for _i in range(0, num_masks):
            # Random stripe width:
            mask_width = random.randrange(min_width, max_width)
            # Random stripe placement: up to case 
            # when stripe is at the edge of the spectro:
            t_zero = random.randrange(0, len_spectro - mask_width)
    
            # avoids randrange error if values are equal and range is empty
            #if (t_zero == t_zero + t):
            #    mask_end = 0
            #    continue
    
            mask_end = t_zero + mask_width
            # cls.log.info(f"Time masked width: [{t_zero} : {mask_end}]")
            if (replace_with_zero): 
                cloned[:,t_zero:mask_end] = 0
            else:
                spectro_mean = cloned.mean()
                #cls.log.info(f"Mean inserted is {spectro_mean}")
                cloned[:,t_zero:mask_end] = spectro_mean

        return cloned, f"-tmask{int(t_zero)}_{int(mask_end)}"

    #------------------------------------
    # random_noise
    #-------------------
    
    @classmethod
    def random_noise(cls, spectrogram, noise_type='uniform', std=1.0):
        '''
        Adds Gaussian or uniform noise to a numpy array, and returns
        the result. The std arg controls with width of the
        (standard) distribution. 
        
           Assumes that std is a positive number
           Assumes that spectrogram is normalized: 0 to 255
        
        :param spectrogram: the spectrogram to modify
        :type spectrogram: np.array
        :param noise_type: whether to add normal (gaussian) or uniform noise;
            must be 'uniform' or 'normal'
        :type noise_type: str
        :param std: standard deviation of the noise
        :type std: float
        :return: spectrogram with random noise added
        :rtype: np.array
        '''
    
        if noise_type not in ('uniform', 'normal'):
            raise ValueError(f"Noise type must be 'uniform', or 'normal'")
        
        clone = spectrogram.copy()
        if noise_type == 'uniform':
            noise = np.random.Generator.normal(np.random.default_rng(),
                                               loc=0.0, 
                                               scale=std, 
                                               size=clone.shape)
        else:
            noise = np.random.Generator.uniform(np.random.default_rng(),
                                                low=0, 
                                                high=1.0, 
                                                size=clone.shape)
        clone_noised  = clone + np.uint8(np.round(noise))

        # We might get out of bounds due to noise addition.
        # Since this method is intended for images, the
        # required value range is 0-255
        clone_clipped = np.clip(clone_noised, 0, 255)
        return clone_clipped, f"-noise{noise.mean()}"

    # ----------------- Utilities --------------

    #------------------------------------
    # noise_multiplier
    #-------------------

    @classmethod
    def noise_multiplier(cls, orig_recording, noise):
        '''
        Return a random noise multiplier between
        3 and 30 dB

        :param orig_recording: audio recording
        :type orig_recording: np.ndarray 
        :param noise: noise recording
        :type noise: np.ndarray
        :return multiplier to apply to noise so that
            adding the result to orig_recording 
            results in the mix
        :rtype float
        '''
        MIN_SNR, MAX_SNR = 3, 30  # min and max sound to noise ratio (in dB)
        snr = random.uniform(MIN_SNR, MAX_SNR)
        noise_rms = np.sqrt(np.mean(noise**2))
        orig_rms  = np.sqrt(np.mean(orig_recording**2))
        desired_rms = orig_rms / (10 ** (float(snr) / 20))
        return desired_rms / noise_rms

    #------------------------------------
    # energy_highlights
    #-------------------
    
    @classmethod
    def energy_highlights(cls, audio_sample, sr):
        '''
        Given an audio data numpy array, return
        an array of Interval instances. Each instance
        contains the low frequency and high frequency
        among the frequency bands of higher intensity
        then 2 standard deviations above the mean intensity
        of the overall audio.
        
        Takes a short-time fast Fourier transform (STFT).
        takes the mean intensity in each frequency band,
        and and find those with higher than 2 sdev mean
        of intensity across the band.
        
        :param audio_sample: 1D array of audio signal
        :type audio_sample: np array
        :return: a list of frequency intervals that each
            contain more than 2 sdev higher intensity than
            the overall signal 
        :rtype: [Interval]
        '''
        
        spectro_complex = librosa.stft(audio_sample)
        spectro = np.abs(spectro_complex)
        
        energy_centroids_by_frame = librosa.fft_frequencies(sr)
        
        #num_freq_bands, num_time_slices = spectro.shape
        freqbands_width = energy_centroids_by_frame[1] - energy_centroids_by_frame[0] 
        
        # Get mean intensity across each freq band:
        intensity_means = np.mean(spectro, axis=1)
        
        # The mean and stdev among those mean intensities:
        intensity_overall_mean = np.mean(spectro)
        intensity_std_among_bands = np.std(intensity_means)
        
        # The bands with higher intensity mean than 
        # 2 stdevs of the bands:
        high_intensity_bands_mask = \
            abs((intensity_means - intensity_overall_mean)) > 2*intensity_std_among_bands
            
        # Use the mask to pull out an array
        # of center frequencies whose band contains
        # more than 2Sdev's worth of intensity. Result
        # will be like:
        #     
        #    array([   0.        ,   10.76660156,   21.53320312, 1679.58984375,
        #           1690.35644531, 1701.12304688, 1711.88964844, 1722.65625   ,        
        #               ...

        high_intensity_bands = np.compress(high_intensity_bands_mask, 
                                           energy_centroids_by_frame)
        
        # Form frequency intervals of high intensity
        # by combining adjacent high-intensity bands.
        # Each element of high_intensity_bands_mask is
        # adjacent to its predecessor, if the two center
        # frequencies are freqbands_width apart:
        
        low_band = 0.
        ref_band = 0.
        intervals = []
        
        for band in high_intensity_bands[1:]:
            if band - ref_band > freqbands_width:
                intervals.append(Interval(low_band, ref_band+1))
                low_band = band
            ref_band = band
        
        # The last interval:
        intervals.append(Interval(low_band, ref_band+1))
        
        return intervals

    #------------------------------------
    # set_random_seed 
    #-------------------

    @classmethod
    def set_random_seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)

    #------------------------------------
    # define_bandpass 
    #-------------------

    @classmethod
    def define_bandpass(cls, lowcut, highcut, sr, order = 2):
        """
        The defintion and implementation of the bandpass filter
    
        :param highcut: the highcut frequency for the bandpass filter
        :type highcut: int
        :param lowcut: the lowcut frequency for the bandpass filter
        :type lowcut: int
        :param sr: the sample rate of the audio
        :type sr: int
        :param order: the order of the filter
        :type order: int
        :returns b, a: Numerator (b) and denominator (a) polynomials of the IIR filter. Only returned if output='ba'.
        """
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype = 'band')
        return b, a

    #------------------------------------
    # filter_bird 
    #-------------------

    @classmethod
    def filter_bird(cls, audio, sr):
        """
        Filters the given audio using a bandpass filter.
    
        :param audio: recording
        :type audio: np.array
        :param sr: the sample rate of the audio
        :type sr: int
        :returns output: the filtered recording audio time series
        """
        #bandpass
        b, a = cls.define_bandpass(500, 8000, sr) # filters out anything not between 0.5 and 8khz
        output = lfilter(b, a, audio)
    
        # noise reduction - easier to listen to for a human, harder for the model to classify
        # select section of data that is noise
        #noisy_part = output[0:1000]
        # perform noise reduction
        #output =  nr.reduce_noise(audio_clip=output, noise_clip=noisy_part, verbose=True, n_grad_freq=0, n_std_thresh=2)
    
        # normalize the volume
        return output / np.max(output)

    #------------------------------------
    # scale_minmax
    #-------------------
    
    @classmethod
    def scale_minmax(cls, X, min_val=0.0, max_val=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max_val - min_val) + min_val
        return X_scaled

    #------------------------------------
    # load_audio 
    #-------------------
    
    @classmethod
    def load_audio(cls, fname, offset=0.0, duration=None):
        '''
        Loads a .wav or mp3 audio file,
        and returns a numpy array, and
        the recording's sample rate.

        :param fname: audio file to load
        :type fname: str
        :param offset: where to start load: seconds into recording 
        :type offset: float
        :param duration: how many seconds to load
        :type duration: {None | float|
        :returns the recording and the associated sample rate
        :rtype: (np.array, float)
        :raises FileNotFoundError, AudioLoadException
        '''
        
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File {fname} does not exist.")
        
        if os.path.getsize(fname) == 0:
            raise AudioLoadException(f"Audio file to load is empty: {fname}", fname)

        # Hide the UserWarning: PySoundFile failed. Trying audioread instead.
        # Happens when loading mp3 files:
        warnings.filterwarnings(action="ignore",
                                message="PySoundFile failed. Trying audioread instead.",
                                category=UserWarning, 
                                module='', 
                                lineno=0)

        try:
            # The following may raise a warning like:
            #   ...audioread/__init__.py:86: ResourceWarning: unclosed file <_io.BufferedReader name=16>
            # this is discussed on the Web, and is due
            # to some subprocess call in audioread; nothing
            # I'll do on this one:

            recording, sample_rate = librosa.load(fname, offset=offset, duration=duration)
        except Exception as e:
            raise AudioLoadException(f"Could not load {fname}", fname, other_msg=repr(e)) from e

        return recording, sample_rate

    #------------------------------------
    # load_spectrogram 
    #-------------------
    
    @classmethod
    def load_spectrogram(cls, fname, to_nparray=True):
        '''
        Loads a .png spectrogram file,
        and returns a numpy array

        :param fname: file to load
        :type fname: str
        :param to_nparray: if True, convert torchvision. Image
            instance to a numpy array, and return that as result
        :type to_nparray: bool
        :returns tuple: the image and the .png file's possibly empty metadata dict
        :rtype: ({np.array|torchvision.Image}, {str : str})
        :raises FileNotFoundError
        '''
        
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File {fname} does not exist.")
        
        try:
            png_img = PngImageFile(fname)
        except Exception as e:
            raise RuntimeError(f"Error opening {fname}: {repr(e)}")
        try:
            info = png_img.text
        except Exception as e:
            cls.log.info(f"No available info in .png file: {repr(e)}")
            info = None
        
        img_obj = Image.open(fname)
        if to_nparray:
            res = np.asarray(img_obj)
        else:
            res = img_obj 
        return (res, info)

    #------------------------------------
    #  soundfile_metadata
    #-------------------

    @classmethod
    def soundfile_metadata(cls, fname):
        '''
        Given the path to an mp3 file, return a dict
           duration     : <datetime.timedelta>
           sample_rate  : <integer in Hz>
        
        :param fname: full path to mp3 file
        :type fname: str
        :return excerpt from mp3 metadata
        :rtype {str : float | int}
        '''

        if not os.path.exists(fname):
            raise FileNotFoundError(f"File {fname} does not exist.")
        
        # If ffmpeg is unavailable, we cannot 
        # find the play lengths of the mp3 recordings.
        # Give up:
        
        if os.environ['PATH'].find(':/usr/local/bin:') == -1:
            os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/bin:'
        if shutil.which('ffprobe') is None:
            raise NotImplementedError("Must install ffmpeg: sudo apt-get ffmpeg")
        
        # Get metadata as a string, like:
        #     Input #0, mp3, from '/tmp/bird.mp3':
        #       Metadata:
        #         title           : Silver-throated Tanager (Tangara icterocephala)
        #         genre           : Thraupidae
        #         artist          : Thore Noernberg
        #         album           : xeno-canto
        #         TIT1            : wbaakn
        #         copyright       : 2012 Thore Noernberg
        #       Duration: 00:00:18.88, start: 0.000000, bitrate: 193 kb/s
        #         Stream #0:0: Audio: mp3, 44100 Hz, stereo, s16p, 192 kb/s

        proc = subprocess.run(['ffprobe', fname], capture_output=True)
        
        # Even if no error, ffprobe writes to stderr,
        # and subprocess returns byte str:
        
        # Output is a byte string, and we need to leave
        # it that way, because some mp3 metadata has bad
        # unicode. We'll deal with those lines in the
        # line-by-line loop below:
        output = proc.stderr.split(b'\n')
        
        # Find duration and sample rates
        duration    = None
        sample_rate = None
        
        for byte_line in output:
            if duration is not None and sample_rate is not None:
                # All done:
                break

            if duration is None:
                match = cls.DURATION_PATTERN.search(byte_line)
                if match is not None:
                    # Get duration as regular string, not byte str:
                    duration = match.group(1).decode('utf8')
                    continue
            if sample_rate is None:
                match = cls.STREAM_PATTERN.search(byte_line)
                if match is not None:
                    sample_rate = int(match.group(1))

        if duration is None or sample_rate is None:
            msg = f"File {fname}: Problem finding: "
            if duration is None:
                msg += 'duration.'
            if sample_rate is None:
                msg += "Problem finding sample rate."
            raise ValueError(f"{msg} ffprobe output: \n {output}")
            
        # Now have like:
        #    duration: '00:00:18.88'
        #    sample_rate: 44100

        # Get a reasonable representation of the duration
        # from something like: '00:00:54.19':
        hrs, mins, secs = duration.split(':')
        duration_obj = datetime.timedelta(hours=int(hrs), 
                                          minutes=int(mins), 
                                          seconds=float(secs))
        return {'duration': duration_obj,
                'sample_rate' : sample_rate
                }
        

    #------------------------------------
    # save_image 
    #-------------------
    
    @classmethod
    def save_image(cls, img, outfile, info=None):
        '''
        Given an image and an optional metadata
        dictionary, write the image as .png, include
        the metadata. 
        
        :param img: image to save
        :type img: np_array
        :param outfile: destination path
        :type outfile: str
        :param info: metadata to add
        :type info: {str : str}
        '''
        
        # Create metadata to include in the 
        # spectrogram .png file:
        if info is not None:
            metadata = PngInfo()
            for key, val in info.items():
                metadata.add_text(key, str(val))

        skimage.io.imsave(outfile, img, pnginfo=metadata)

    #------------------------------------
    # save_img_array 
    #-------------------
    
    @classmethod
    def save_img_array(cls, img_arr, dst_path):
        
        dst_dir = os.path.dirname(dst_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            
        img = Image.fromarray(img_arr)
        img.save(dst_path)

    #------------------------------------
    # gzip_file
    #-------------------
    
    @classmethod
    def gzip_file(cls, file_path, delete=False):
        '''
        reads contents of given file, and writes
        back a gzip compressed copy to <file_path>.gz
    
        If delete is True, the original file is deleted
        from the file system. Else two files will exist
        after the call: the original, and the .gz version.
                  
        :param file_path: path to file to compress
        :type file_path: str
        :param delete: if True, delete original file
            after saving a compressed version
        :type delete: bool
        :return path to compressed file
        :rtype str
        '''

        with open(file_path, 'r') as fd:
            content = fd.read()
        
        compressed_content = gzip.compress(content.encode('utf8'))
        dest_fpath = file_path + '.gz'
        with open(dest_fpath, 'bw') as fd:
            fd.write(compressed_content)
        
        if delete:
            os.remove(file_path)
        
        return dest_fpath

    #------------------------------------
    # read_gzipped_file
    #-------------------
    
    @classmethod
    def read_gzipped_file(cls, fname):
        
        with gzip.open(fname, 'rb') as fd:
            content = fd.read().decode('utf8')
        return content 

