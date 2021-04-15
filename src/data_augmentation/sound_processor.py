import math
import os, sys
import random
import warnings

from PIL import Image
import librosa.display
from logging_service import LoggingService
from scipy.signal import butter
from scipy.signal import lfilter
from matplotlib import MatplotlibDeprecationWarning

import data_augmentation.utils as utils
import matplotlib.pyplot as plt
import numpy as np


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

    #------------------------------------
    # Constructor Stub 
    #-------------------
    
    def __init__(self):
        raise NotImplementedError("Class SoundProcessor is not intended for instantiation")

    #------------------------------------
    # set_random_seed 
    #-------------------

    @classmethod
    def set_random_seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)

    #------------------------------------
    # add_background
    #-------------------

    @classmethod
    def add_background(cls, file_name, noise_path, in_dir, out_dir, len_noise_to_add=5.0):
        """
        Combines the wav recording in the species subdirectory with a random 5s clip of audio from a random recording in
        data_augmentation/lib.
    
        :param species: the directory names of the species to modify the wav files of. If species=None, all subdirectories will be used.
        :type species: str
        """
        len_noise_to_add = float(len_noise_to_add)  # need length to be float
        backgrounds = os.listdir(noise_path)
    
        background_name = backgrounds[random.randint(0, len(backgrounds)-1)]
        # Check if augmented file exists
        output_path = os.path.join(out_dir, file_name[:-len(".wav")]) + "-" + background_name
        if os.path.exists(output_path):
            ans = input(f"Do you want to rewrite {output_path}? [y/N]  ")
            if ans not in ["y", "yes"]:
                return file_name
    
        cls.log.info(f"Adding {background_name} to {file_name}.")
        # We will be working with 1 second as the smallest unit of time
        # load all of both wav files and determine the length of each
        noise, noise_sr = librosa.load(os.path.join(noise_path, background_name))  # type(noise) = np.ndarray
        orig_recording, orig_sr = librosa.load(os.path.join(in_dir, file_name))
    
        new_sr = math.gcd(noise_sr, orig_sr)
        if noise_sr != orig_sr:
            # Resample both noise and orig records so that they have same sample rate
            cls.log.info(f"Resampling: {background_name} and {file_name}")
            noise = librosa.resample(noise, noise_sr, new_sr)
            orig_recording = librosa.resample(orig_recording, orig_sr, new_sr)
            input("ready?")
    
        noise_duration = librosa.get_duration(noise, noise_sr)
        if noise_duration < len_noise_to_add:
            cls.log.info(f"Duration:{noise_duration} < len_noise_to_add:{len_noise_to_add}. Will only add {noise_duration}s of noise")
            samples_per_segment = len(noise)
        elif noise_duration > len_noise_to_add:  # randomly choose noise segment
            samples_per_segment = int(new_sr * len_noise_to_add)  # this is the number of samples per 5 seconds
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
    
        assert(len(during_noise) == len(noise))
    
        segment_with_noise = during_noise + utils.noise_multiplier(orig_recording, noise) * noise
        first_half = np.concatenate((before_noise, segment_with_noise))
        new_record = np.concatenate((first_half, after_noise)) # what i think it should be
        new_duration = librosa.get_duration(new_record, float(new_sr))
    
        assert(new_duration == orig_duration)
        # output the new wav data to a file
        aug_sample_name = file_name[:-len(".wav")] + "-" + background_name[:-len(".wav")] + "_bgd" + str(int(noise_start_loc/new_sr * 1000)) + "ms.wav"
        output_path = os.path.join(out_dir, aug_sample_name)
        librosa.output.write_wav(output_path, new_record, new_sr)
        return aug_sample_name

    #------------------------------------
    # change_volume 
    #-------------------

    @classmethod
    def change_volume(in_dir, out_dir, species=None):
        """
        Adjusts the volume of all the wav files in the species directory.
    
        :param in_dir: the path to the directory to fetch samples from
        :type in_dir: str
        :param out_dir: the path to the directory to save the new files to
        :type out_dir: str
        :param species: the directory names of the species to modify the wav files of. If species=None, all subdirectories will be used.
        :type species: str
        """
        for file_name in os.listdir(in_dir):
            if species is None or species in file_name:
                y0, sample_rate0 = librosa.load(os.path.join(in_dir, file_name))
    
                # adjust the volume
                factor = random.randrange(-12, 12, 1)
                y1 = y0 * (10 ** (factor / 20))
    
                # output the new wav data to a file
                librosa.output.write_wav(os.path.join(out_dir, file_name[:len(file_name) - 4])
                                         + "-volume" + str(factor) + ".wav", y1, sample_rate0)
    #------------------------------------
    # time_shift 
    #-------------------

    @classmethod
    def time_shift(cls, file_name, in_dir, out_dir):
        """
        Performs a time shift on all the wav files in the species directory. The shift is 'rolling' such that
        no information is lost.
    
        :param in_dir: the path to the directory to fetch samples from
        :type in_dir: str
        :param out_dir: the path to the directory to save the new files to
        :type out_dir: str
        :param species: the directory names of the species to modify the wav files of. If species=None, all subdirectories will be used.
        :type species: str
        """
        y, sample_rate = librosa.load(os.path.join(in_dir, file_name))
        length = librosa.get_duration(y, sample_rate)  # returns length in seconds
        # shifts the recording by a random amount between 0 and length of recording by a multiple of 10 ms
        amount = random.randrange(0, int(length)*10, 1)/10  # shift is in seconds
    
        # create two seperate sections of the audio
        y0, sample_rate0 = librosa.load(os.path.join(in_dir, file_name), offset=amount)
        y1, _sample_rate1 = librosa.load(os.path.join(in_dir, file_name), duration=amount)
    
        # combine the wav data
        y2 = np.append(y0, y1)
        # print(f"Amount: {amount}ms")
        assert(len(y) == len(y2))
        # output the new wav data to a file
        aug_sample_name = file_name[:len(file_name) - 4] + "-shift" + str(int(amount * 1000)) + "ms.wav"
        librosa.output.write_wav(os.path.join(out_dir, aug_sample_name), y2, sample_rate0)
        return aug_sample_name

    #------------------------------------
    # warp_spectrogram 
    #-------------------

    @classmethod
    def warp_spectrogram(cls, file_name, in_dir, out_dir):
        """
        Performs Frequency and Time Masking for Spectrograms
    
        :param in_dir: the path to the directory containing spectrograms
        :type in_dir: str
        :param out_dir: the path to the directory to save the augmented spectrograms
        :type out_dir: str
        :param species: the directory names of the species to modify the wav files of. If species=None, all subdirectories will be used.
        :type species: str
        """
        cls.log.info(f"Adding masks to {file_name}")
        print(file_name)
        orig_spectrogram = np.asarray(Image.open(os.path.join(in_dir, file_name)))
        freq_masked, freq_name = cls.freq_mask(orig_spectrogram, num_masks=2)
        masked_spectrogram, time_name = cls.time_mask(freq_masked, num_masks=2)
        plt.imshow(masked_spectrogram);
        plt.axis('off')
        aug_sample_name = file_name[:-len(".png")] + "-" + freq_name + "-" + time_name +".png"
        plt.savefig(os.path.join(out_dir, aug_sample_name), dpi=100, bbox_inches='tight', pad_inches=0, format='png', facecolor='none')
        return aug_sample_name

    #------------------------------------
    # create_spectrogram 
    #-------------------

    """ Below from src/recording_download.py """
    @classmethod
    def create_spectrogram(cls, birdname, in_dir, out_dir, n_mels=128):
        """
        Filters audio and converts it to a spectrogram which is saved.
    
        :param birdname: the bird's scientific name + recording id
        :type birdname: str
        :param instance: which chronological 5s segment of the orginal recording recording this clip is from
        :type instance: int
        :param audio: the librosa audio data of the 5s recording to be filtered
        :type audio: audio time
        :param sr: the sample rate of the audio
        :type sr: int
        :param out_dir: the relative or absolute file path to a directory to put the output files
        :type out_dir: str
        :param n_mels: the number of mel bands in the spectrogram
        :type n_mels: int
        """
        # create a mel spectrogram
        spectrogramfile = os.path.join(out_dir, birdname[:-len(".wav")]) + '.png'
        cls.log.info(f"Creating spectrogram: {spectrogramfile}")
        audio, sr = librosa.load(os.path.join(in_dir, birdname))
        cls.log.info(f"Audio len: {len(audio)/sr}")
        # Use bandpass filter for audio before converting to spectrogram
        audio = cls.filter_bird(audio, sr)
        mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels)
        # create a logarithmic mel spectrogram
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # create an image of the spectrogram and save it as file
        fig = plt.figure(figsize=(6.07, 2.02))
        
        # Don't show the annoying deprecation
        # librosa.display() warnings about renaming
        # 'basey' to 'base' to match matplotlib: 
        warnings.simplefilter("ignore", category=MatplotlibDeprecationWarning)
        # Same for UserWarning: PySoundFile failed. Trying audioread instead:
        warnings.filterwarnings(action="ignore",
                                message="PySoundFile failed. Trying audioread instead.",
                                category=UserWarning, 
                                module='', 
                                lineno=0)

        librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', cmap='gray_r')
            
        plt.tight_layout()
        plt.axis('off')
        # Workaround for "Done(renderer)" exception in Tkinter callback        
        fig.canvas.start_event_loop(sys.float_info.min) 
        plt.savefig(spectrogramfile, dpi=100, bbox_inches='tight', pad_inches=0, format='png', facecolor='none')
        plt.close()
        return birdname[:-len(".wav")] + '.png'

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
        Opens a specifc recording of a bird, filters the audio and converts it to a spectrogram which is saved.
    
        :param birdname: the bird's scientific name + recording id
        :type birdname: str
        :param audio: the librosa audio data of the 5s recording to be filtered
        :type audio: audio time
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
    # freq_mask 
    #-------------------

    # Functions below are from https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb
    # Functions edited to support np arrays
    @classmethod
    def freq_mask(cls, spec, F=30, num_masks=1, replace_with_zero=False):
        cloned = spec.copy()
        num_mel_channels = cloned.shape[0]
        cls.log.info(f"num_mel_channels is {num_mel_channels}")
    
        for _i in range(0, num_masks):
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)
    
            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): continue
    
            mask_end = random.randrange(f_zero, f_zero + f)
            cls.log.info(f"Masked freq is [{f_zero} : {mask_end}]")
            cls.log.info(f"Mean inserted is {cloned.mean()}")
            if (replace_with_zero): cloned[f_zero:mask_end] = 0
            else: cloned[f_zero:mask_end] = cloned.mean()
        return cloned, f"fmask{int(f_zero)}_{int(mask_end)}"

    #------------------------------------
    # time_mask 
    #-------------------

    @classmethod
    def time_mask(cls, spec, T=40, num_masks=1, replace_with_zero=False):
        cloned = spec.copy()
        len_spectro = cloned.shape[1]
        for _i in range(0, num_masks):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)
    
            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t):
                mask_end = 0
                continue
    
            mask_end = random.randrange(t_zero, t_zero + t)
            cls.log.info(f"Masked time is [{t_zero} : {mask_end}]")
            cls.log.info(f"Mean inserted is {cloned.mean()}")
            if (replace_with_zero): cloned[:,t_zero:mask_end] = 0
            else: cloned[:,t_zero:mask_end] = cloned.mean()
        return cloned, f"tmask{int(t_zero)}_{int(mask_end)}"
    
