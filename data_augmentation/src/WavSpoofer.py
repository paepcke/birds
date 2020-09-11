import librosa
import random
import os

BACKGROUND_NOISE_PATH = "/Users/LeoGl/PycharmProjects/bird/wav_test/experimental/background"
DESTINATION_PATH = "/Users/LeoGl/PycharmProjects/bird/wav_test/experimental/dest"
SOURCE_FOLDER_PATH = "/Users/LeoGl/PycharmProjects/bird/wav_test/experimental/source"


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


def add_background():
    for file_name in os.listdir(SOURCE_FOLDER_PATH):
        for background_name in os.listdir(BACKGROUND_NOISE_PATH):
            # load the entire background wav file and determine length
            y0, sample_rate0 = librosa.load(BACKGROUND_NOISE_PATH + "/" + background_name)
            duration = int(librosa.get_duration(y0, sample_rate0))
            start_loc = random.randrange(0, duration - 5, 1)

            # load the source wav file and 5 seconds of the background file
            y1, sample_rate1 = librosa.load(SOURCE_FOLDER_PATH + "/" + file_name, duration=5)
            y2, sample_rate2 = librosa.load(BACKGROUND_NOISE_PATH + "/" + background_name, duration=5, offset=start_loc)

            # combine the wav data
            y3 = y1 * (10 ** (-3 / 20)) + (y2 * (10 ** (-3 / 20)))
            sr = int((sample_rate1+sample_rate2)/2)

            # output the new wav data to a file
            librosa.output.write_wav(DESTINATION_PATH + "/" + file_name[:len(file_name) - 4]
                                     + "_" + background_name[:len(file_name) - 4], y3, sr)


if __name__ == '__main__':
    add_background()

