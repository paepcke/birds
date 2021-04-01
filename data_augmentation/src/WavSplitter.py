import requests
import json
import torch
# import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import librosa.display
import soundfile as sf
import math
import os
import random


birds = ['Tangara+gyrola', 'Amazilia+decora', 'Hylophilus+decurtatus', 'Arremon+aurantiirostris',
         'Dysithamnus+mentalis', 'Lophotriccus+pileatus', 'Euphonia+imitans', 'Tangara+icterocephala',
         'Henicorhina+leucosticta', 'Corapipo+altera']


def download_bird(bird):
    # download file
    url = 'http:' + bird['url'] + '/download'
    birdsong = requests.get(url)

    # write to file
    birdname = bird['gen'] + bird['sp'] + bird['id']
    filepath = './Birdsong/' + birdname + '.wav'
    with open(filepath, 'wb') as f:
        f.write(birdsong.content)
    return birdname


def get_data(birdname):
    # read back from file
    filepath = os.fspath('/home/data/birds/NEW_BIRDSONG/CALL/' + birdname + '.wav')

    audiotest, sr = librosa.load(filepath, sr=None)  # fix this
    dur = librosa.get_duration(audiotest, sr)

    for i in range(0, int(dur / 5)):

        rand = random.randrange(0, 5, 1)
        audio, sr = librosa.load(filepath, offset=5.0 * i + rand, duration=5.0, sr=None)
        # filter the bird audio
        audio = filter_bird(birdname, str(i), audio, sr)


def filter_bird(birdname, instance, audio, sr):
    # bandpass
    b, a = define_bandpass(2000, 3500, sr)
    output = signal.lfilter(b, a, audio)

    # output = output - np.mean(output)

    # write filtered file
    outfile = '/home/data/birds/NEW_BIRDSONG/CALL_FILTERED/' + birdname + instance + '.wav'
    librosa.output.write_wav(outfile, output, sr)
    return output


# definition and implementation of the bandpass filter
def define_bandpass(lowcut, highcut, sr, order=2):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


#for i in range(0, 10):
#    birdname = birds[i]
#    response = requests.get("https://www.xeno-canto.org/api/2/recordings?query=" + birdname + "+len:5-60+q_gt:C")
#    data = response.json()

#    for bird in data['recordings']:
#        birdname = download_bird(bird)  # downloads the bird and returns the important part of filename
#        get_data(birdname)

for recording in os.listdir('/home/data/birds/NEW_BIRDSONG/CALL'):
    if "Euphoniaimitans" in recording:
        get_data(recording[:len(recording)-4])
