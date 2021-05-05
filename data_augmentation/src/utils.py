import random
import numpy as np
import pandas as pd
import os
import librosa

def noise_multiplier(orig_recording, noise):
    MIN_SNR, MAX_SNR = 3, 30  # min and max sound to noise ratio (in dB)
    snr = random.uniform(MIN_SNR, MAX_SNR)
    noise_rms = np.sqrt(np.mean(noise**2))
    orig_rms  = np.sqrt(np.mean(orig_recording**2))
    desired_rms = orig_rms / (10 ** (float(snr) / 20))
    return desired_rms / noise_rms

def create_folder(dir_path):
    if os.path.exists(dir_path):
        ans = input(f"{dir_path} already exists. Do you want to rewrite {dir_path}? [y/N]  ")
        return ans in ["y", "yes"]
    else:
        os.mkdir(dir_path)
        return True

def sample_compositions_by_species(path, augmented):
    num_samples_in = {} # initialize dict - usage num_samples_in['CORALT_S'] = 64
    for species in os.listdir(path):
        if augmented:
            aug_type_dict = {"add_bg":0, "time_shift":0, "mask":0, "original":0}
            for sample_name in os.listdir(os.path.join(path, species)):
                if "_bgd" in sample_name: aug_type_dict["add_bg"] += 1
                elif "-shift" in sample_name: aug_type_dict["time_shift"] += 1
                elif "fmask" in sample_name or "tmask" in sample_name: aug_type_dict["mask"] += 1
                else: aug_type_dict["original"] += 1
            num_samples_in[species]= aug_type_dict
        else:
            num_samples_in[species] = {"num_samples":len(os.listdir(os.path.join(path, species)))}
    return pd.DataFrame.from_dict(num_samples_in, orient='index').sort_index()

def find_total_recording_length(species_dir_path):
    total_duration = 0
    for recording in os.listdir(species_dir_path):
        y, sr = librosa.load(os.path.join(species_dir_path, recording))
        total_duration += librosa.get_duration(y, sr)
    return total_duration

def recording_lengths_by_species(path):
    num_samples_in = {} # initialize dict - usage num_samples_in['CORALT_S'] = 64
    for species in os.listdir(path):
        num_samples_in[species] = {"total_recording_length": find_total_recording_length(os.path.join(path, species))}
    return pd.DataFrame.from_dict(num_samples_in, orient='index').sort_index()


def count_max_augs(distribution):
    count = 0
    for val in distribution:
        if val != 0:
            count += 1
    return count
