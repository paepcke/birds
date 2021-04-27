import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import sound_processor as aug
import specaugment as sa
import os
import numpy as np
from PIL import Image
import logging as log


def test_add_background():
    file_name, noise_path, in_dir, out_dir = "Corapipoaltera13180.wav", "data_augmentation/lib/Noise_Recordings/", "raw_wav_files/", "wav_augmented/"
    sample_name = aug.add_background(file_name, noise_path, in_dir, out_dir, len_noise_to_add=5.0)
    print(f"Sample name: {sample_name}")

def test_time_shift():
    file_name, in_dir, out_dir = "Corapipoaltera13180.wav", "raw_wav_files/", "wav_augmented/"
    sample_name= aug.time_shift(file_name, in_dir, out_dir)
    print(f"Sample name: {sample_name}")

def test_create_spectrogram():
    file_name, in_dir, out_dir = "Corapipoaltera13180.wav", "raw_wav_files/", "spectrogram_augmented/"
    spectrogram_name = aug.create_spectrograms(file_name, in_dir, out_dir, n_mels=128)
    print(f"Spectrogram name: {spectrogram_name}")

def test_warp_spectrogram():
    file_name, in_dir, out_dir = "Corapipoaltera13180.png", "spectrogram_augmented/", "spectrogram_augmented/"
    sample_name = aug.warp_spectrogram(file_name, in_dir, out_dir)
    print(f"Sample name: {sample_name}")

def test_time_mask():
    in_dir = "../TAKAO_BIRD_WAV_feb20/AMADEC_C/"
    spectrogram_dir = "../TAKAO_BIRD_WAV_feb20_augmented_samples_0.00n_0.00ts_1.00w/spectrograms_augmented/AMADEC_C/"
    for spectrogram in os.listdir(in_dir):
        spectrogram = spectrogram[:-len(".wav")] + ".png"
        try:
             orig_spectrogram = np.asarray(Image.open(os.path.join(spectrogram_dir, spectrogram)))
        except FileNotFoundError as e:
            print(e)
            continue
        masked_spectrogram, time_name = sa.time_mask(orig_spectrogram, log, num_masks=2)
        print(time_name)

if __name__ == '__main__':
    # test_add_background()
    # test_time_shift()
    # test_create_spectrogram()
    # test_warp_spectrogram()
    test_time_mask()
