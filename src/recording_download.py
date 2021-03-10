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
import logging as log

# A list of the scientific names for the bird species in this study

# List of bird species to download. List of tuples (species_name, split call and song?)
# Not in list but was previously in this set: 'Catharus+ustulatus',
BIRD_LIST = [('Amazilia+decora', True), ('Arremon+aurantiirostris', True),
             ('Corapipo+altera', True),
             ('Dysithamnus+mentalis', True),
             ('Empidonax+flaviventris', False), ('Euphonia+imitans', False),
             ('Henicorhina+leucosticta', True), ('Hylophilus+decurtatus', False),
             ('Lophotriccus+pileatus', True),
             ('Parula+pitiayumi', True),
             ('Tangara+gyrola', False), ('Tangara+icterocephala', False)
            ]

def folder_prefix(birdname):
    name_list = birdname.split('+')
    #Folder prefix is first three letters of first and second part of birdname
    folder_prefix = name_list[0][:3] + name_list[1][:3]
    return folder_prefix.upper()

def download_bird_samples(birds, out_dir, num_samples_per_species):
    """
    Wrapper function for downloading birds. Iterates through list birds and calls download function for each
    species.
    """
    for birdname, split_call_song in birds:
        # download samples from zeno-canto
        num_records = download_bird(birdname, out_dir, folder_prefix(birdname), split_call_song, to_download=num_samples_per_species)
        print(f"Downloaded {num_records} records for {birdname}.")

def download_bird(birdname, out_dir, folder_prefix, split_call_song, to_download = -1):
    """
    Downloads all recording of a specifc bird species from zeno-canto

    :param bird: the scientific name of the bird to be downloaded
    :type bird: str
    :param out_dir: the relative or absolute file path to a directory to put the output files
    :type out_dir: str
    :returns num: returns the number of records downloaded for the bird
    """
    # Create folders for samples to go in
    # TODO: once merged with takao branch, use utils.make_folder
    if split_call_song:
        # Make folder for call and song
        if not os.path.exists(os.path.join(out_dir, folder_prefix + "_S")):
            os.mkdir(os.path.join(out_dir, folder_prefix + "_S"))
        if not os.path.exists(os.path.join(out_dir, folder_prefix + "_C")):
            os.mkdir(os.path.join(out_dir, folder_prefix + "_C"))
    else:
        # Make one folder for the species
        if not os.path.exists(os.path.join(out_dir, folder_prefix)):
            os.mkdir(os.path.join(out_dir, folder_prefix))

    response = requests.get("https://www.xeno-canto.org/api/2/recordings?query=" + birdname + "+len:5-60+q_gt:C")
    data = response.json()

    num_downloaded = 0
    to_download = len(data['recordings']) if to_download == -1 else to_download
    print(f"Request for {birdname} returned {len(data['recordings'])} recordings. ")
    for record in data['recordings']: # each bird may have multiple records
        #download file
        url = 'http:' + record['url'] + '/download'
        try:
            birdsong = requests.get(url)
        except:
            print(f"Could not download url: {url}")
            continue
        #write to file
        record_name = record['gen'] + record['sp'] + record['id']
        if split_call_song:
            if isinstance(record['type'], str):
                type_list = record['type'].lower()
            else:
                type_list = [type.lower() for type in record['type']]
            if 'song' in type_list:
                folder_name = folder_prefix + "_S"
            elif 'call' in type_list:
                folder_name = folder_prefix + "_C"
            else:
                folder_name = folder_prefix
                print(f"Unknown record type: type_list = {type_list}. Skipping {record_name}...")
                continue
            filepath = os.path.join(out_dir, folder_name, record_name) + '.wav'
        else:
            filepath = os.path.join(out_dir, folder_prefix, record_name) + '.wav'

        with open(filepath, 'wb') as f:
            f.write(birdsong.content)
        num_downloaded += 1
        if num_downloaded == to_download: break
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
    log.info(f"Creating spectrogram: {spectrogramfile}")
    audio, sr = librosa.load(os.path.join(in_dir, birdname))
    log.info(f"Audio len: {len(audio)/sr}")
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
    if len(sys.argv) <= 5:

        if sys.argv[1] in ["-d", "-da"]:  # user wants to download, second argument is the bird name, third is directory to save download
            # if -d download only for input bird, otherwise download all birds in birds list
            if sys.argv[1] == "-d":
                filepath_out = str(sys.argv[3])
                input_birdname = sys.argv[2].split(',')
                if len(input_birdname) == 2:
                    birds = [(input_birdname[0], input_birdname[1] == "True")]
                else:
                    print("ERROR: specify birdname in format: birdn_name,True/False")
                # if there is a fifth argument, it specifies samples per species. -1 to download all samples
                num_samples_per_species = int(sys.argv[4]) if len(sys.argv) == 5 else -1
            else:
                birds = BIRD_LIST
                # if there is a fourth argument, it specifies samples per species. -1 to download all samples
                num_samples_per_species = int(sys.argv[3]) if len(sys.argv) == 4 else -1
                # for -da flag, file_out path is specified afer flag
                filepath_out = str(sys.argv[2])

            download_bird_samples(birds, filepath_out, num_samples_per_species)

        elif sys.argv[1] in ["-f", "-s"]:  # these flags can only be used after sample(s) are downloaded
            filepath_in = str(sys.argv[2])
            filepath_out = str(sys.argv[3])
            for sample in os.listdir(filepath_in):
                audio, sr = librosa.load(os.path.join(filepath_in, sample))
                instance = ""
                if sys.argv[1] == "-f": # filter and split the data
                    get_data(sample, filepath_in, filepath_out)
                elif sys.argv[1] == "-s": # convert to a spectrogram
                    create_spectrogram(sample, instance, audio, sr, filepath_out, filepath_in)
        else:
            print("ERROR: invalid flag argument. \n Accepted flags: -d, -da, -f, -s")
    else:
        print(f"ERROR: invlaid number of arguments: Given {len(sys.argv)} arguments")
