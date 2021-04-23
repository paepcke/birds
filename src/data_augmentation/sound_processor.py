import math
import os
from pathlib import Path
import random

import soundfile
import librosa
from PIL import Image
from logging_service import LoggingService
from scipy.signal import butter
from scipy.signal import lfilter
import skimage.io

from data_augmentation.utils import Utils
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
    def add_background(cls, file_name, noise_path, out_dir, len_noise_to_add=5.0):
        '''
        Takes an absolute file path, and the path to a
        directory that contains noise to overlay onto the 
        given sound file (wind, rain, etc.).
        
        Returns a numpy structure corresponding to the
        original audio with the noise overlaid, plus the
        sample rate of the new sample. A file name is suggested
        for the sample. It is composed of elements such 
        as the nature and duration of the noise. Client
        may choose to ignore or use.

        :param file_name: absolute path to sound file
        :type file_name: str
        :param noise_path: absolute path to directory
            with noise files
        :type noise_path: str
        :param out_dir: destination directory of new audio file
        :type out_dir: str
        :param len_noise_to_add: how much of a noise snippet
            to overlay (seconds)
        :type len_noise_to_add: float
        :return: full path of new audio file
        :rtype: str
        '''
        
        len_noise_to_add = float(len_noise_to_add)
        backgrounds = os.listdir(noise_path)
    
        # Pick a random noise file:
        background_name = backgrounds[random.randint(0, len(backgrounds)-1)]
    
        cls.log.info(f"Adding {background_name} to {file_name}.")
        
        # We will be working with 1 second as the smallest unit of time
        # load all of both wav files and determine the length of each
        noise, noise_sr = librosa.load(os.path.join(noise_path, background_name))  # type(noise) = np.ndarray
        orig_recording, orig_sr = librosa.load(file_name)
    
        new_sr = math.gcd(noise_sr, orig_sr)
        if noise_sr != orig_sr:
            # Resample both noise and orig records so that they have same sample rate
            cls.log.info(f"Resampling: {background_name} and {file_name}")
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
    
        assert(len(during_noise) == len(noise))
    
        segment_with_noise = during_noise + Utils.noise_multiplier(orig_recording, noise) * noise
        first_half   = np.concatenate((before_noise, segment_with_noise))
        new_sample   = np.concatenate((first_half, after_noise)) # what i think it should be
        new_duration = librosa.get_duration(new_sample, float(new_sr))
    
        assert(new_duration == orig_duration)
        # File name w/o extension:
        sample_file_stem = Path(file_name).stem
        noise_file_stem  = Path(background_name).stem
        noise_dur = str(int(noise_start_loc/new_sr * 1000))
        file_name= f"{sample_file_stem}-{noise_file_stem}_bgd{noise_dur}ms.wav"
        out_path = os.path.join(out_dir, file_name)
        
        soundfile.write(out_path, new_sample, new_sr)
        return out_path

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
        y0, sample_rate0 = librosa.load(sample_path)

        # Adjust the volume
        factor = random.randrange(-12, 12, 1)
        
        cls.log.info(f"Changing volume of {sample_path} by factor {factor}.")

        y1 = y0 * (10 ** (factor / 20))

        # Output the new wav data to a file
        # Just the foofile part of /home/me/foofile.mp3:
        sample_root = Path(sample_path).stem
        new_sample_fname = f"{sample_root}-volume{factor}.wav"
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
        """
        y, sample_rate = librosa.load(file_name)
        length = librosa.get_duration(y, sample_rate)  # returns length in seconds
        # shifts the recording by a random amount between 0 and length of recording by a multiple of 10 ms
        amount = random.randrange(0, int(length)*10, 1)/10  # shift is in seconds
    
        # Create two seperate sections of the audio
        # Snippet after the shift amount:
        y0, sample_rate0 = librosa.load(file_name, offset=amount)
        # Snippet before the shift amount:
        y1, _sample_rate1 = librosa.load(file_name, duration=amount)
    
        # Append the before-snippet to the 
        # end of the after-snippet: 
        y2 = np.append(y0, y1)
        # print(f"Amount: {amount}ms")
        assert(len(y) == len(y2))
        # Output the new wav data to a file
        # Get just the 'foo' part of '/blue/red/foo.mp3':
        file_stem = Path(file_name).stem
        aug_sample_name = f"{file_stem}-shift{str(int(amount * 1000))}ms.wav"
        out_path = os.path.join(out_dir, aug_sample_name)
        soundfile.write(out_path, y2, sample_rate0)
        return out_path

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
        # print(file_name)
        orig_spectrogram = np.asarray(Image.open(os.path.join(in_dir, file_name)))
        freq_masked, freq_name = cls.freq_mask(orig_spectrogram, num_masks=2)
        masked_spectrogram, time_name = cls.time_mask(freq_masked, num_masks=2)
        img = Image.fromarray(masked_spectrogram)
        fpath = Path(file_name)
        new_file_name = f"{fpath.stem}-{freq_name}-{time_name}.png"
        outpath = Path.joinpath(fpath.parent, new_file_name)
        img.save(outpath)

    #------------------------------------
    # create_spectrogram 
    #-------------------

    @classmethod
    def create_spectrogram(cls, audio_sample, sr, outfile, n_mels=128):
        '''
        Create and save a spectrogram from an audio 
        sample. Bandpass filter is applied, Mel scale is used,
        and power is converted to decibels.
         
        :param audio_sample: audio
        :type audio_sample: np.array
        :param sr: sample rate
        :type sr: int
        :param outfile: where to store the result
        :type outfile: str
        :param n_mels: number of mel scale bands 
        :type n_mels: int
        '''
        # Use bandpass filter for audio before converting to spectrogram
        audio = SoundProcessor.filter_bird(audio_sample, sr)
        mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels)
        # create a logarithmic mel spectrogram
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # Create an image of the spectrogram and save it as file
        img = cls.scale_minmax(log_mel, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image
        img = 255-img # invert. make black==more energy

        # Save as PNG
        skimage.io.imsave(outfile, img)

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
    

    #------------------------------------
    # scale_minmax
    #-------------------
    
    @classmethod
    def scale_minmax(cls, X, min_val=0.0, max_val=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max_val - min_val) + min_val
        return X_scaled

