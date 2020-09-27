import librosa
import random
import os
import numpy as np

# BACKGROUND_NOISE_PATH = "/home/lglik/Code/birds/data_augmentation/lib/default_background"
# DESTINATION_PATH = "/home/data/birds/Birdsong_Filtered_Augmented/"
# SOURCE_FOLDER_PATH = "/home/data/birds/Birdsong_Filtered/"
# SPECIES = ""

BACKGROUND_NOISE_PATH = "/Users/lglik/Code/birds/data_augmentation/lib/default_background"
DESTINATION_PATH = "/Users/LeoGl/PycharmProjects/bird/wav_test/experimental/dest/"
SOURCE_FOLDER_PATH = "/Users/LeoGl/PycharmProjects/bird/wav_test/experimental/source/"
SPECIES = ""


# this does not work that well
def time_shift(species=None):
    for file_name in os.listdir(SOURCE_FOLDER_PATH):
        if species is None or species in file_name:
            amount = random.randrange(0, 500, 1) * 0.01
            y0, sample_rate0 = librosa.load(SOURCE_FOLDER_PATH + "/" + file_name, offset=amount)
            y1, sample_rate1 = librosa.load(SOURCE_FOLDER_PATH + "/" + file_name, duration=amount)

            # combine the wav data
            y2 = np.append(y0, y1)

            # output the new wav data to a file
            librosa.output.write_wav(DESTINATION_PATH + "/" + file_name[:len(file_name) - 4]
                                     + "_shift" + str(amount) + ".wav", y2, sample_rate0)


def change_volume(species=None):
    for file_name in os.listdir(SOURCE_FOLDER_PATH):
        if species is None or species in file_name:
            y0, sample_rate0 = librosa.load(SOURCE_FOLDER_PATH + "/" + file_name)

            # adjust the volume
            factor = random.randrange(-12, 12, 1)
            y1 = y0 * (10 ** (factor / 20))

            # output the new wav data to a file
            librosa.output.write_wav(DESTINATION_PATH + "/" + file_name[:len(file_name) - 4]
                                     + "_volume" + str(factor) + ".wav", y1, sample_rate0)


def add_background(species=None):
    for file_name in os.listdir(SOURCE_FOLDER_PATH):
        if species is None or species in file_name:
            for background_name in os.listdir(BACKGROUND_NOISE_PATH):
                # load the entire background wav file and determine length
                y0, sample_rate0 = librosa.load(BACKGROUND_NOISE_PATH + "/" + background_name)
                duration = int(librosa.get_duration(y0, sample_rate0))
                start_loc = random.randrange(0, duration - 5, 1)

                # load the source wav file and 5 seconds of the background file
                y1, sample_rate1 = librosa.load(SOURCE_FOLDER_PATH + "/" + file_name, duration=5)
                y2, sample_rate2 = librosa.load(BACKGROUND_NOISE_PATH + "/" + background_name, duration=5,
                                                offset=start_loc)

                # combine the wav data
                y3 = y1 * (10 ** (-3 / 20)) + (y2 * (10 ** (-3 / 20)))
                sr = int((sample_rate1 + sample_rate2) / 2)

                # output the new wav data to a file
                librosa.output.write_wav(DESTINATION_PATH + "/" + file_name[:len(file_name) - 4]
                                         + "_" + background_name[:len(file_name) - 4], y3, sr)


if __name__ == '__main__':
    time_shift()

