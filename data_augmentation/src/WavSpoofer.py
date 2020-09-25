import librosa
import random
import os
import math

BACKGROUND_NOISE_PATH = "/home/lglik/Code/birds/data_augmentation/lib/default_background"
DESTINATION_PATH = "/home/data/birds/NEW_BIRDSONG/CALL_FILTERED_AUGMENTED"
SOURCE_FOLDER_PATH = "/home/data/birds/NEW_BIRDSONG/CALL_FILTERED"
SPECIES = "Euphoniaimitans"


# this does not work that well
def time_shift():
    for file_name in os.listdir(SOURCE_FOLDER_PATH):
        for shift in range(10):
            y1, sample_rate1 = librosa.load(SOURCE_FOLDER_PATH + "/" + file_name)
            amount = random.randrange(1, 4, 1)
            y1 *= 0
            y2, sample_rate2 = librosa.load(SOURCE_FOLDER_PATH + "/" + file_name, offset=amount)

            # combine the wav data
            y3 = y1 * (10 ** (-3 / 20)) + (y2 * (10 ** (-3 / 20)))
            sr = int((sample_rate1 + sample_rate2) / 2)

            # output the new wav data to a file
            librosa.output.write_wav(DESTINATION_PATH + "/" + file_name[:len(file_name) - 4]
                                     + "_shift" + amount, y3, sr)


def add_background(species):
    for file_name in os.listdir(SOURCE_FOLDER_PATH):
        if species is None or species in file_name:
            for background_name in os.listdir(BACKGROUND_NOISE_PATH):
                # load all of both wav files and determine length
                y0, sample_rate0 = librosa.load(BACKGROUND_NOISE_PATH + "/" + background_name)
                duration = int(librosa.get_duration(y0, sample_rate0))
                start_loc = random.randrange(0, duration - 5, 1)
                y1, sample_rate1 = librosa.load(SOURCE_FOLDER_PATH + "/" + file_name, duration=5)
                recording_length = librosa.core.get_duration(y=y1, sr=sample_rate1)
                
                # load the source wav file and 5 seconds of the background file
                
                dur_limit = 5.0
                if recording_length < 5:
                    dur_limit = math.floor(recording_length / 0.05) * 0.05
                
                y2, sample_rate2 = librosa.load(BACKGROUND_NOISE_PATH + "/" + background_name, duration=dur_limit,
                                                offset=start_loc)
                y3, sample_rate3 = librosa.load(SOURCE_FOLDER_PATH + "/" + file_name, duration=dur_limit)
                
                
                print("recording_length", recording_length)
                print("dur_limit", dur_limit)
                print("background duration", librosa.core.get_duration(y=y2, sr=sample_rate2))

                # combine the wav data
                y4 = y2 * (10 ** (-3 / 20)) + (y3 * (10 ** (-3 / 20)))
                sr = int((sample_rate2 + sample_rate3) / 2)

                # output the new wav data to a file
                librosa.output.write_wav(DESTINATION_PATH + "/" + file_name[:len(file_name) - 4]
                                         + "_" + background_name[:len(file_name) - 4], y4, sr)


if __name__ == '__main__':
    add_background(SPECIES)

