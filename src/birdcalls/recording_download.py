import requests 
import json
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import librosa.display
import soundfile as sf
import math


"""
Functions involved in downloading birds from xeno-canto
"""


birds = ['Tangara+gyrola', 'Amazilia+decora', 'Hylophilus+decurtatus', 'Arremon+aurantiirostris', 
         'Dysithamnus+mentalis', 'Lophotriccus+pileatus', 'Euphonia+imitans', 'Tangara+icterocephala', 
         'Catharus+ustulatus', 'Parula+pitiayumi', 'Henicorhina+leucosticta', 'Corapipo+altera', 
         'Empidonax+flaviventris']



def download_bird(bird):
    #download file
    url = 'http:' + bird['url'] + '/download'
    birdsong = requests.get(url)

    #write to file
    birdname = bird['gen'] + bird['sp'] + bird['id'] 
    filepath = './Birdsong/' + birdname + '.wav' 
    with open(filepath, 'wb') as f:
        f.write(birdsong.content)           
    return birdname
    
def get_data(birdname):
    #read back from file
    filepath = './Birdsong/' + birdname + '.wav'
    
    audiotest, sr = librosa.load(filepath, sr = None)      #fix this
    dur = librosa.get_duration(audiotest, sr)
    
    for i in range (0, int(dur/5)):
        audio, sr = librosa.load(filepath, offset = 5.0 * i, duration = 5.0, sr = None)
        #filter the bird audio
        audio = filter_bird(birdname, str(i), audio, sr)
    
        #create and save spectrogram
        create_spectrogram(birdname, str(i), audio, sr)
    
    
def filter_bird(birdname, instance, audio, sr):
    #bandpass
    b, a = define_bandpass(2000, 3500, sr)
    output = lfilter(b, a, audio)
    
    #output = output - np.mean(output)
    
    #write filtered file
    outfile = './Birdsong_Filtered/' + birdname + instance + '.wav' 
    librosa.output.write_wav(outfile, output, sr)
    return output
    
#definition and implementation of the bandpass filter
def define_bandpass(lowcut, highcut, sr, order = 2):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = 'band')
    return b, a
    
def create_spectrogram(birdname, instance, audio, sr, n_mels = 128):
    spectrogramfile = './Birdsong_Spectrograms/' + birdname + instance + '.jpg' 
    mel = librosa.feature.melspectrogram(audio, sr = sr, n_mels = n_mels)
    log_mel = librosa.power_to_db(mel, ref = np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.savefig(spectrogramfile, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()   

for i in range(11, 13):
    birdname = birds[i]
    response = requests.get("https://www.xeno-canto.org/api/2/recordings?query=" + birdname + "+len:5-60+q_gt:C")
    data = response.json()

    for bird in data['recordings']:
        birdname = download_bird(bird)   #downloads the bird and returns the important part of filename
        get_data(birdname)