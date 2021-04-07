import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter
from scipy.signal import lfilter
import soundfile as sf
import librosa
import librosa.display
import noisereduce as nr
import os
import sys
import argparse
import soundfile as sf
sys.path.append('data_augmentation/src')
import augmentations as aug

def create_spectrogram(sample_name, sample, sr, out_dir, n_mels=128):
    # Use bandpass filter for audio before converting to spectrogram
    audio = aug.filter_bird(sample, sr)
    mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels)
    # create a logarithmic mel spectrogram
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # create an image of the spectrogram and save it as file
    plt.figure(figsize=(6.07, 2.02))
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', cmap='gray_r')
    plt.tight_layout()
    plt.axis('off')
    spectrogramfile = os.path.join(out_dir, sample_name + '.png')
    plt.savefig(spectrogramfile, dpi=100, bbox_inches='tight', pad_inches=0, format='png', facecolor='none')
    plt.close()

def sliding_window(in_dir, species, sample_name, out_dir, window_len = 5):
    """
    Performs a time shift on all the wav files in the species directory. The shift is 'rolling' such that
    no information is lost.

    :param file_name: the path to the original.wav file to split using sliding window
    :type file_name: str
    :param out_dir: the path to the directory to save the new files to
    :type out_dir: str
    :param species: the directory names of the species to modify the wav files of. If species=None, all subdirectories will be used.
    :type species: str
    """
    orig, sample_rate = librosa.load(os.path.join(in_dir, species, sample_name))
    length = int(librosa.get_duration(orig, sample_rate))
    for start_time in range(length - window_len):
        window, sr = librosa.load(os.path.join(in_dir, species, sample_name),
                                  offset=start_time, duration=window_len)
        window_name = sample_name[:-len(".wav")] + '_sw-start' + str(start_time)
        create_spectrogram(window_name, window, sr, os.path.join(out_dir, 'spectrograms/', species))
        sf.write(os.path.join(out_dir, 'wav-files', species, window_name + '.wav'), window, sr)

def create_folder(dir_path):
    if os.path.exists(dir_path):
        ans = input(f"{dir_path} already exists. Do you want to rewrite {dir_path}? [y/N]  ")
        return ans in ["y", "yes"]
    else:
        os.mkdir(dir_path)
        return True

def main(in_dir, out_dir, specific_species=None):
    species_list = os.listdir(in_dir)
    create_folder(out_dir)
    spectrogram_dir_path = os.path.join(out_dir,'spectrograms/')
    wav_dir_path = os.path.join(out_dir,'wav-files/')
    create_folder(spectrogram_dir_path)
    create_folder(wav_dir_path)
    if specific_species == None:
        for species in os.listdir(in_dir):
            if (create_folder(os.path.join(spectrogram_dir_path, species))
            and create_folder(os.path.join(wav_dir_path, species))):
                for sample_name in os.listdir(os.path.join(in_dir, species)):
                    sliding_window(in_dir, species, sample_name, out_dir)
            else:
                print(f"Skipping sliding window for {species}")
    else:
        if specific_species in species_list:
            if (create_folder(os.path.join(spectrogram_dir_path, specific_species))
            and create_folder(os.path.join(wav_dir_path, specific_species))):
                for sample_name in os.listdir(os.path.join(in_dir, specific_species)):
                    sliding_window(in_dir, specific_species, sample_name, out_dir)
            else:
                print(f"Skipping sliding window for {species}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    parser.add_argument('input_path',
                           metavar='IN_DIR',
                           type=str,
                           help='the path to original directory with .wav files')
    parser.add_argument('output_path',
                           metavar='OUT_DIR',
                           type=str,
                           help='the path to output directory to write new .wav/.png files')
    parser.add_argument('-s', '--species',
                           metavar='S',
                           type=str,
                           help='specific species to use sliding window on',
                           default=None)
    # Execute the parse_args() method
    args = parser.parse_args()
    main(args.input_path, args.output_path, specific_species=args.species)
