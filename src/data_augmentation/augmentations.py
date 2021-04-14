import librosa
import librosa.display
import random
import os
import numpy as np
import math
import data_augmentation.utils as utils
import logging as log
import data_augmentation.specaugment as sa
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy.signal import lfilter

class SoundProcessor:

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
    
        log.info(f"Adding {background_name} to {file_name}.")
        # We will be working with 1 second as the smallest unit of time
        # load all of both wav files and determine the length of each
        noise, noise_sr = librosa.load(os.path.join(noise_path, background_name))  # type(noise) = np.ndarray
        orig_recording, orig_sr = librosa.load(os.path.join(in_dir, file_name))
    
        new_sr = math.gcd(noise_sr, orig_sr)
        if noise_sr != orig_sr:
            # Resample both noise and orig records so that they have same sample rate
            log.info(f"Resampling: {background_name} and {file_name}")
            noise = librosa.resample(noise, noise_sr, new_sr)
            orig_recording = librosa.resample(orig_recording, orig_sr, new_sr)
            input("ready?")
    
        noise_duration = librosa.get_duration(noise, noise_sr)
        if noise_duration < len_noise_to_add:
            log.info(f"Duration:{noise_duration} < len_noise_to_add:{len_noise_to_add}. Will only add {noise_duration}s of noise")
            samples_per_segment = len(noise)
        elif noise_duration > len_noise_to_add:  # randomly choose noise segment
            samples_per_segment = int(new_sr * len_noise_to_add)  # this is the number of samples per 5 seconds
            subsegment_start = random.randint(0, len(noise) - samples_per_segment)
            noise = noise[subsegment_start: subsegment_start + samples_per_segment]
        log.info(f"len(noise) after random segment: {len(noise)}; noise duration: {len(noise)/new_sr}")
    
    
        orig_duration = librosa.core.get_duration(orig_recording, orig_sr)
        # if orig_recording is shorter than the noise we want to add, just add 5% noise
        if orig_duration < len_noise_to_add:
            log.info(f"Recording: {file_name} was shorter than len_noise_to_add. Adding 5% of recording len worth of noise.")
            new_noise_len = orig_duration * 0.05
            noise = noise[:int(new_noise_len * new_sr)]
        noise_start_loc = random.randint(0, len(orig_recording) - samples_per_segment)
        log.info(f"Inserting noise starting at {noise_start_loc/new_sr} seconds.")
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
        log.info(f"Adding masks to {file_name}")
        print(file_name)
        orig_spectrogram = np.asarray(Image.open(os.path.join(in_dir, file_name)))
        freq_masked, freq_name = sa.freq_mask(orig_spectrogram, log, num_masks=2)
        masked_spectrogram, time_name = sa.time_mask(freq_masked, log, num_masks=2)
        plt.imshow(masked_spectrogram);
        plt.axis('off')
        aug_sample_name = file_name[:-len(".png")] + "-" + freq_name + "-" + time_name +".png"
        plt.savefig(os.path.join(out_dir, aug_sample_name), dpi=100, bbox_inches='tight', pad_inches=0, format='png', facecolor='none')
        return aug_sample_name

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
        log.info(f"Creating spectrogram: {spectrogramfile}")
        audio, sr = librosa.load(os.path.join(in_dir, birdname))
        log.info(f"Audio len: {len(audio)/sr}")
        # Use bandpass filter for audio before converting to spectrogram
        audio = cls.filter_bird(audio, sr)
        mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels)
        # create a logarithmic mel spectrogram
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # create an image of the spectrogram and save it as file
        plt.figure(figsize=(6.07, 2.02))
        librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', cmap='gray_r')
        plt.tight_layout()
        plt.axis('off')
    
        plt.savefig(spectrogramfile, dpi=100, bbox_inches='tight', pad_inches=0, format='png', facecolor='none')
        plt.close()
        return birdname[:-len(".wav")] + '.png'

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
