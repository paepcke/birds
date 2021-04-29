import math
import os, sys
from pathlib import Path
import random
import warnings

from PIL import Image
from PIL.PngImagePlugin import PngImageFile, PngInfo
import librosa
from logging_service import LoggingService
from scipy.signal import butter
from scipy.signal import lfilter
import skimage.io
import soundfile

from data_augmentation.utils import Utils
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



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

    # ------------------ Operations on Sound Files --------------

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
        noise, noise_sr = SoundProcessor.load_audio(os.path.join(noise_path, background_name))  # type(noise) = np.ndarray
        orig_recording, orig_sr = SoundProcessor.load_audio(file_name)
    
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
    
        assert len(during_noise) == len(noise)
    
        segment_with_noise = during_noise + Utils.noise_multiplier(orig_recording, noise) * noise
        first_half   = np.concatenate((before_noise, segment_with_noise))
        new_sample   = np.concatenate((first_half, after_noise)) # what i think it should be
        new_duration = librosa.get_duration(new_sample, float(new_sr))
    
        assert new_duration == orig_duration
        # File name w/o extension:
        sample_file_stem = Path(file_name).stem
        noise_file_stem  = Path(background_name).stem
        noise_dur = str(int(noise_start_loc/new_sr * 1000))
        file_name= f"{sample_file_stem}-{noise_file_stem}_bgd{noise_dur}ms.wav"
        
        # Ensure that the fname doesn't exist:
        uniq_fname = Utils.unique_fname(out_dir, file_name)
        out_path = os.path.join(out_dir, uniq_fname)
        
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
        y0, sample_rate0 = SoundProcessor.load_audio(sample_path)

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
        aug_sample_name = f"{file_stem}-shift{str(int(amount * 1000))}ms.wav"
        out_path = os.path.join(out_dir, aug_sample_name)
        soundfile.write(out_path, y2, sample_rate0)
        return out_path

    # --------------- Operations on Spectrograms Files --------------

    #------------------------------------
    # create_spectrogram 
    #-------------------

    @classmethod
    def create_spectrogram(cls, audio_sample, sr, outfile, n_mels=128, info=None):
        '''
        Create and save a spectrogram from an audio 
        sample. Bandpass filter is applied, Mel scale is used,
        and power is converted to decibels.
        
        The sampling rate (sr) and time duration in fractional
        seconds is added as metadata under keys "sr" and "duration".
        Additional info may be included in the .png if info is 
        a dict of key/value pairs.
        
        Retrieving the metadata can be done via SoundProcessor.load_spectrogram(),
        or any other PNG reading software that handles the PNG specification.
         
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
        
        # Use bandpass filter for audio before converting to spectrogram
        audio = SoundProcessor.filter_bird(audio_sample, sr)
        mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels)
        # create a logarithmic mel spectrogram
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
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
        :raises FileNotFoundError
        '''
        
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File {fname} does not exist.")

        # Hide the UserWarning: PySoundFile failed. Trying audioread instead.
        # Happens when loading mp3 files:
        warnings.filterwarnings(action="ignore",
                                message="PySoundFile failed. Trying audioread instead.",
                                category=UserWarning, 
                                module='', 
                                lineno=0)

        recording, sample_rate = librosa.load(fname, offset=offset, duration=duration)
        return recording, sample_rate

    #------------------------------------
    # load_spectrogram 
    #-------------------
    
    @classmethod
    def load_spectrogram(cls, fname):
        '''
        Loads a .png spectrogram file,
        and returns a numpy array

        :param fname: file to load
        :type fname: str
        :returns tuple: the image and the .png file's possibly empty metadata dict
        :rtype: (np.array, {str : str}
        :raises FileNotFoundError
        '''
        
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File {fname} does not exist.")

        png_img = PngImageFile(fname)
        try:
            info = png_img.text
        except Exception as e:
            cls.log.info(f"No available info in .png file: {repr(e)}")
            info = None
        
        img = np.asarray(Image.open(fname))
        return (img, info)

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

            if info is not None:
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

