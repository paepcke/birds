"""
Functions involved in downloading birds from xeno-canto and afiltering, augmenting, and sorting those downloads.
"""
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

# A list of the scientific names for the bird species in this study


birds = ['Tangara+gyrola', 'Amazilia+decora', 'Hylophilus+decurtatus', 'Arremon+aurantiirostris',
         'Dysithamnus+mentalis', 'Lophotriccus+pileatus', 'Euphonia+imitans', 'Tangara+icterocephala',
         'Catharus+ustulatus', 'Parula+pitiayumi', 'Henicorhina+leucosticta', 'Corapipo+altera',
         'Empidonax+flaviventris']


def download_bird(bird, out_dir):
    """
    Downloads all recording of a specifc bird species from zeno-canto

    :param bird: the scientific name of the bird to be downloaded
    :type bird: str
    :param out_dir: the relative or absolute file path to a directory to put the output files
    :type out_dir: str
    :returns num: returns the number of records downloaded for the bird
    """
    response = requests.get("https://www.xeno-canto.org/api/2/recordings?query=" + birdname + "+len:5-60+q_gt:C")
    data = response.json()

    num_downloaded = 0
    for record in data['recordings']: # each bird may have multiple records
        #download file
        url = 'http:' + record['url'] + '/download'
        try:
            birdsong = requests.get(url)
            num_downloaded += 1
        except:
            print(f"Could not download url: {url}")
            continue
        #write to file
        record_name = record['gen'] + record['sp'] + record['id']
        filepath = os.path.join(out_dir, record_name) + '.wav'
        with open(filepath, 'wb') as f:
            f.write(birdsong.content)
    return num_downloaded


def get_data(birdname, in_dir, out_dir):
    """
    Opens a specifc recording of a bird, filters the audio and saves the new version.

    :param birdname: the bird's scientific name + recording id
    :type birdname: str
    :param in_dir: the relative or absolute file path to a directory to fetch the files from
    :type in_dir: str
    :param out_dir: the relative or absolute file path to a directory to put the output files
    :type out_dir: str
    """
    # read back from file
    filepath = os.path.join(in_dir, birdname)

    audiotest, sr = librosa.load(filepath, sr=None)
    dur = librosa.get_duration(audiotest, sr)

    # split the audio and filter
    for i in range(0, int(dur/5)):
        audio, sr = librosa.load(filepath, offset=5.0 * i, duration=5.0, sr=None)
        # filter the bird audio
        audio = filter_bird(birdname, audio, sr)
        # output the filtered audio
        outfile = os.path.join(out_dir, birdname[:len(birdname) - 4]) + str(i) + ".wav"
        librosa.output.write_wav(outfile, audio, sr)


def filter_bird(birdname, audio, sr):
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
    b, a = define_bandpass(500, 8000, sr) # filters out anything not between 0.5 and 8khz
    output = lfilter(b, a, audio)

    # noise reduction - easier to listen to for a human, harder for the model to classify
    # select section of data that is noise
    #noisy_part = output[0:1000]
    # perform noise reduction
    #output =  nr.reduce_noise(audio_clip=output, noise_clip=noisy_part, verbose=True, n_grad_freq=0, n_std_thresh=2)

    # normalize the volume
    return output / np.max(output)


def define_bandpass(lowcut, highcut, sr, order = 2):
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


def create_spectrogram(birdname, instance, audio, sr, out_dir, in_dir, n_mels=128):
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
    spectrogramfile = os.path.join(out_dir, birdname[:len(birdname) - 4]) + '.png'
    audio, sr = librosa.load(os.path.join(in_dir, birdname))
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


if __name__ == '__main__':
    """
    Parses the command line parameters and calls the appropriate functions..
    """
    if len(sys.argv) == 4:
        filepath_out = str(sys.argv[3])
        if sys.argv[1] in ["-d", "-da"]:  # user wants to download, second argument is the bird name, third is directory to save download

            if not os.path.isdir(filepath_out):
                os.mkdir(filepath_out)

            # if -d download only for input bird, otherwise download all birds in birds list
            birds = [sys.argv[2]] if sys.argv[1] == "-d" else birds

            for birdname in birds:
                # download samples from zeno-canto
                num_records = download_bird(birdname, filepath_out)
                print(f"Downloaded {num_records} records for {birdname}.")
        elif sys.argv[1] in ["-f", "-s"]:  # these flags can only be used after sample(s) are downloaded
            filepath_in = str(sys.argv[2])

            for sample in os.listdir(filepath_in):
                audio, sr = librosa.load(os.path.join(filepath_in, sample))
                instance = ""
                if sys.argv[1] == "-f": # filter and split the data
                    get_data(sample, filepath_in, filepath_out)
                elif sys.argv[1] == "-s": # convert to a spectrogram
                    create_spectrogram(sample, instance, audio, sr, filepath_out, filepath_in)
        else:
            print("ERROR: invalid flag argument. \n Accepted flags: -d, -f, -s")
    else:
        print(f"ERROR: invlaid number of arguments: Given {len(sys.argv)} arguments")
