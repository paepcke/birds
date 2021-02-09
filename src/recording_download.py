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
    :returns birdname: returns the bird's scientific name + recording id
    """
    #download file
    url = 'http:' + bird['url'] + '/download'
    birdsong = requests.get(url)

    #write to file
    birdname = bird['gen'] + bird['sp'] + bird['id'] 
    filepath = os.path.join(out_dir, birdname) + '.wav' 
    with open(filepath, 'wb') as f:
        f.write(birdsong.content)           
    return birdname


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


def create_spectrogram(birdname, in_dir, out_dir, n_mels=128):
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
    spectrogramfile = os.path.join(out_dir, birdname[:-len(".wav")]) + '.png'
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
        filepath_in = str(sys.argv[2])
        filepath_out = str(sys.argv[3])
        download = False
        for sample in os.listdir(filepath_in):
            # instance = ""
            if sys.argv[1] == "-f": # filter and split the data
                get_data(sample, filepath_in, filepath_out)
            elif sys.argv[1] == "-s": # convert to a spectrogram
                create_spectrogram(sample, filepath_in, filepath_out)
            elif sys.argv[1] == "-d": # download samples from zeno-canto
                download = True
                break
            else:
                print("ERROR: invalid flag argument")
        if download:
            download_bird(sys.arv[2], sys.argv[3])
    else:
        print("ERROR: invlaid number of arguments")
        
