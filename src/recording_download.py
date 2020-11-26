"""
Functions involved in downloading birds from xeno-canto
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

# A list of the scientific names for the bird species in this study


birds = ['Tangara+gyrola', 'Amazilia+decora', 'Hylophilus+decurtatus', 'Arremon+aurantiirostris',
         'Dysithamnus+mentalis', 'Lophotriccus+pileatus', 'Euphonia+imitans', 'Tangara+icterocephala', 
         'Catharus+ustulatus', 'Parula+pitiayumi', 'Henicorhina+leucosticta', 'Corapipo+altera', 
         'Empidonax+flaviventris']


def download_bird(bird):
    """
    Downloads all recording of a specifc bird species from zeno-canto

    :param bird: the scientific name of the bird to be downloaded
    :type bird: str
    :returns birdname: returns the bird's scientific name + recording id
    """
    #download file
    url = 'http:' + bird['url'] + '/download'
    birdsong = requests.get(url)

    #write to file
    birdname = bird['gen'] + bird['sp'] + bird['id'] 
    filepath = './Birdsong/' + birdname + '.wav' 
    with open(filepath, 'wb') as f:
        f.write(birdsong.content)           
    return birdname


def get_data(birdname, out_dir):
    """
    Opens a specifc recording of a bird, filters the audio and converts it to a spectrogram which is saved.

    :param birdname: the bird's scientific name + recording id
    :type birdname: str
    """
    # read back from file
    filepath = './Birdsong/' + birdname + '.wav'
    
    audiotest, sr = librosa.load(filepath, sr=None)      # fix this
    dur = librosa.get_duration(audiotest, sr)
    
    for i in range(0, int(dur/5)):
        audio, sr = librosa.load(filepath, offset=5.0 * i, duration=5.0, sr=None)
        # filter the bird audio
        audio = filter_bird(birdname, str(i), audio, sr, out_dir)
    
        # create and save spectrogram
        create_spectrogram(birdname, str(i), audio, sr)
    
    
def filter_bird(birdname, instance, audio, sr, out_dir):
    """
    Opens a specifc recording of a bird, filters the audio and converts it to a spectrogram which is saved.

    :param birdname: the bird's scientific name + recording id
    :type birdname: str
    :param instance: which chronological 5s segment of the orginal recording recording this clip is from
    :type instance: int
    :param audio: the librosa audio data of the 5s recording to be filtered
    :type audio: audio time
    :param sr: the sample rate of the audio
    :type sr: int
    :returns output: the filtered recording audio time series
    """
    #bandpass
    b, a = define_bandpass(2000, 3500, sr)
    # below 2khz
    output = lfilter(b, a, audio)

    # noise reduction
    # select section of data that is noise
    noisy_part = output[0:1000]
    # perform noise reduction
    reduced_noise = nr.reduce_noise(audio_clip=output, noise_clip=noisy_part, verbose=True)

    # write filtered file
    outfile = out_dir + birdname + instance + '.wav'
    sf.write(outfile, reduced_noise, sr)
    return output


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


def create_spectrogram(birdname, instance, audio, sr, n_mels = 128):
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
    :param n_mels: the number of mel bands in the spectrogram
    :type n_mels: int
    """
    # create a mel spectrogram
    spectrogramfile = './Birdsong_Spectrograms/' + birdname + instance + '.jpg'
    mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels)
    # create a logarithmic mel spectrogram
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # create an image of the spectrogram and save it as file
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.savefig(spectrogramfile, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()


if __name__ == '__main__':
    """
    Downloads all the recordings for all the species in the study and converts them to filtered logarithmic mel
    spectrograms.
    """
    # for i in range(0, 13):
    #     birdname = birds[i]
    #     response = requests.get("https://www.xeno-canto.org/api/2/recordings?query=" + birdname + "+len:5-60+q_gt:C")
    #     data = response.json()
    #
    #     for bird in data['recordings']:
    #         birdname = download_bird(bird)   #downloads the bird and returns the important part of filename
    #         get_data(birdname)

    filepath_in = "/Users/LeoGl/Documents/Filtering/BIRDSONG/"
    filepath_out = "/Users/LeoGl/Documents/Filtering/Birdsong_Filtered/"
    for species in os.listdir(filepath_in):
        for sample in os.listdir(filepath_in + species):
            audio, sr = librosa.load(filepath_in + species + "/" + sample, sr=None)
            instance = ""
            birdname = "filt_" + sample
            filter_bird(birdname, instance, audio, sr, filepath_out + species + "/")
