"""
A collection of functions for modifying groups of .wav files in order to increase the number and variety of samples.
"""
import librosa
import random
import os
import numpy as np
import math
import sys

# BACKGROUND_NOISE_PATH = "/Users/lglik/Code/birds/data_augmentation/lib/default_background"
BACKGROUND_NOISE_PATH = "data_augmentation/lib/default_background/"
DESTINATION_PATH = "/home/data/birds/NEW_BIRDSONG/EXTRA_FILTERED/CALL_FILTERED_FULL_AUG"
SOURCE_FOLDER_PATH = "/home/data/birds/NEW_BIRDSONG/EXTRA_FILTERED/CALL_FILTERED_TIMESHIFT"
SPECIES = ""


def time_shift(in_dir, out_dir, species=None):
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
    for file_name in os.listdir(in_dir):
        if species is None or species in file_name:
            y, sample_rate = librosa.load(os.path.join(in_dir, file_name))
            length = 95 * librosa.get_duration(y, sample_rate)
            # shifts the recording by a random amount between 0 and 5 seconds by a multiple of 10 ms
            amount = random.randrange(1, int(length), 1) * 0.01

            # create two seperate sections of the audio
            y0, sample_rate0 = librosa.load(os.path.join(in_dir, file_name), offset=amount)
            y1, sample_rate1 = librosa.load(os.path.join(in_dir, file_name), duration=amount)

            # combine the wav data
            y2 = np.append(y0, y1)

            # output the new wav data to a file
            librosa.output.write_wav(os.path.join(out_dir, file_name[:len(file_name) - 4])
                                     + "_shift" + str(amount) + ".wav", y2, sample_rate0)


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
                                     + "_volume" + str(factor) + ".wav", y1, sample_rate0)


def add_background(in_dir, out_dir, species=None, len_noise_to_add=5.0):
    """
    Combines the wav recording in the species subdirectory with a random 5s clip of audio from a random recording in
    data_augmentation/lib.

    :param species: the directory names of the species to modify the wav files of. If species=None, all subdirectories will be used.
    :type species: str
    """
    len_noise_to_add = float(len_noise_to_add)  # need length to be float
    for file_name in os.listdir(in_dir):
        if species is None or species in file_name:
            for background_name in os.listdir(BACKGROUND_NOISE_PATH):
                # We will be working with 1 second as the smallest unit of time
                # load all of both wav files and determine the length of each
                noise, noise_sr = librosa.load(os.path.join(BACKGROUND_NOISE_PATH, background_name))  # type(noise) = np.ndarray
                orig_recording, orig_sr = librosa.load(os.path.join(in_dir, file_name))

                # Returns ndarray with new sample rate
                def resample(orig, orig_sr, new_sr):
                    return np.delete(orig, np.arrange(0, len(orig), orig_sr/new_sr))

                new_sr = math.gcd(noise_sr, orig_sr)
                if noise_sr != orig_sr:
                    # Resample both noise and orig records so that they have same sample rate
                    print(f"Resampling: {background_name} and {file_name}")
                    noise = resample(noise, noise_sr, new_sr)
                    orig_recording = resample(orig_recording, orig_sr, new_sr)
                    input("ready?")

                noise_duration = librosa.get_duration(noise, noise_sr)
                if noise_duration < len_noise_to_add:
                    print(f"Cannot add noise with file: {background_name}. Duration:{noise_duration} < len_noise_to_add:{len_noise_to_add}")
                elif noise_duration > len_noise_to_add:  # randomly choose noise segment
                    samples_per_segment = int(new_sr * len_noise_to_add)  # this is the number of samples per 5 seconds
                    subsegment_start = random.randint(0, len(noise) - samples_per_segment)
                    noise = noise[subsegment_start: subsegment_start + samples_per_segment]
                print(f"len(noise) after random segment: {len(noise)}; noise duration: {len(noise)/new_sr}")


                orig_duration = librosa.core.get_duration(orig_recording, orig_sr)
                # if orig_recording is shorter than the noise we want to add, just add 5% noise
                if orig_duration < len_noise_to_add:
                    print(f"Recording: {file_name} was shorter than len_noise_to_add. Adding 5% of recording len worth of noise.")
                    new_noise_len = orig_duration * 0.05
                    noise = noise[:int(new_noise_len * new_sr)]
                noise_start_loc = random.randint(0, len(orig_recording) - samples_per_segment)
                print(f"Inserting noise starting at {noise_start_loc/new_sr} seconds.")
                # split original into three parts: before_noise, during_noise, after_noise
                before_noise = orig_recording[:noise_start_loc]
                during_noise = orig_recording[noise_start_loc: noise_start_loc + samples_per_segment]
                after_noise = orig_recording[noise_start_loc + samples_per_segment:]

                assert(len(during_noise) == len(noise))

                def noise_multiplier(orig_recording, noise):
                    MIN_SNR, MAX_SNR = 3, 30  # min and max sound to noise ratio (in dB)
                    snr = random.uniform(MIN_SNR, MAX_SNR)
                    noise_rms = np.sqrt(np.mean(noise**2))
                    orig_rms  = np.sqrt(np.mean(orig_recording**2))
                    desired_rms = orig_rms / (10 ** (float(snr) / 20))
                    return desired_rms / noise_rms

                segment_with_noise = during_noise + noise_multiplier(orig_recording, noise) * noise
                first_half = np.concatenate((before_noise, segment_with_noise))
                new_record = np.concatenate((first_half, after_noise)) # what i think it should be
                new_duration = librosa.get_duration(new_record, float(new_sr))

                assert(new_duration == orig_duration)
                # output the new wav data to a file
                librosa.output.write_wav(os.path.join(out_dir, file_name[:-len(".wav")])
                                         + "_" + background_name[:len(file_name) - 4], new_record, new_sr)


if __name__ == '__main__':
    """The main method. Parses command-line input and calls appropriate functions:"""
    if len(sys.argv) >= 4:
        in_dir = sys.argv[2]
        out_dir = sys.argv[3]
        if sys.argv[1] == "-ts":
            time_shift(in_dir, out_dir)
        elif sys.argv[1] == "-cv":
            change_volume(in_dir, out_dir)
        elif sys.argv[1] == "-ab":
            add_background(in_dir, out_dir,
                           species= sys.argv[4] if len(sys.argv) == 5 else None)
        else:
            print("ERROR: invlaid parameter flag")
    else:
        print("ERROR: invalid arguements")
        print(len(sys.argv))
