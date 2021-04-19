import random
import numpy as np
import pandas as pd
import os
import librosa
import shutil
# File name matching with bash wildcard patterns:
from fnmatch import fnmatch

def noise_multiplier(orig_recording, noise):
    MIN_SNR, MAX_SNR = 3, 30  # min and max sound to noise ratio (in dB)
    snr = random.uniform(MIN_SNR, MAX_SNR)
    noise_rms = np.sqrt(np.mean(noise**2))
    orig_rms  = np.sqrt(np.mean(orig_recording**2))
    desired_rms = orig_rms / (10 ** (float(snr) / 20))
    return desired_rms / noise_rms

def create_folder(dir_path, overwrite_freely=False):
    '''
    Given the path to a directory:
        o If the dir does not exist, create it
        o Ask for permission to replace the dir
            - if permission denied, existing dir remains
            - else tree under dir is cleared; empty dir remains
          If overwrite_freely is True, skip the request
          for permission. 
    
    Returns True if a new, empty directory was created.
    Return of False means the dir existed, and was left
    untouched. 
            
    :param dir_path: path of directory to create 
    :type dir_path: str
    :return True is empty dir was created, else False
    :rtype: bool
    
    '''
    if os.path.exists(dir_path):
        if overwrite_freely:
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            return True
        # Need to ask client for permission to replace:
        ans = input(f"{dir_path} already exists. Replace? [y/N]:  ")
        if ans.lower() in ["y", "yes"]:
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            return True
        # Did not create a new dir
        return False
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

def find_in_dir_tree(root_dir, pattern='*', entry_type='file'):
    '''
    Similates part of the bash shell find command.
    Only bash patters are supported, not Python regex.
    For the find -type values, only file and directory 
    are supported as file types. Though abbreviations
    'f' and 'd' are fine.
    
    Case of strings are case-normalized using os.path.normcase().
    Change fnmatch to fnmatchcase() in the function to
    perform case-sensitive searches.
    
    Examples:
       
       # Return list of files in /tmp
          find_in_dir_tree('/tmp', entry_type='file')
          
       # Return list of text files in /tmp
          find_in_dir_tree('/tmp', pattern="*.txt")
          
       # Return list of directories in /tmp that start 
       # with 'Kfold'
          find_in_dir_tree('/tmp', pattern="Kfold", entry_type='directory')
          find_in_dir_tree('/tmp', pattern="Kfold", entry_type='d')
          
    
    :param root_dir: directory in which to start recursive descent
    :type root_dir: str
    :param pattern: pattern that file or dir must match
    :type pattern: str
    :param entry_type: whether file or directories are included
    :type entry_type: str
    :return: list of file or directory names
    :rtype: [str]
    '''

    file_list = []
    if pattern is None and entry_type is None:
        raise ValueError("At least one of pattern and/or type must be non-None")
    if entry_type not in [None, 'file', 'f', 'directory', 'd']:
        raise ValueError(f"Entry type must be one of {[None, 'file', 'f', 'directory', 'd']}, not {entry_type}")

    if entry_type in [None, 'f', 'file']:
        entry_type = 'file'
    else:
        entry_type = 'directory'

    # Walk through directory
    for dName, _sdName, fList in os.walk(root_dir):

        if entry_type == 'directory' and fnmatch(dName, pattern):
            file_list.append(dName)
            continue
    
        for fname in fList:
            if entry_type == 'directory' and os.path.isdir(fname):
                file_list.append(fname)
                continue
            # Match search string?
            if fnmatch(fname, pattern): 
                file_list.append(os.path.join(dName, fname))

    return file_list
