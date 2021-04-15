#!/usr/bin/env python3

import argparse
import os
import warnings

import librosa.display
from logging_service.logging_service import LoggingService
from matplotlib import MatplotlibDeprecationWarning

from data_augmentation import utils
from data_augmentation.sound_processor import SoundProcessor as aug
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


class SoundChopper:
    '''
    Processes directories of .wav or .mp3 files,
    chopping them into window_len second snippets.
    Each audio snippet is saved, and spectrograms
    are created for each.
    
    Assumes:

                        self.in_dir
                        
          Species1        Species2   ...     Speciesn
           smpl1_1.mp3      smpl2_1.mp3         smpln_1.mp3
           smpl1_2.mp3      smpl2_2.mp3         smpln_2mp3
                            ...
                            
    Chops each .mp3 (or .wav) file into window_len snippets.
    Saves those snippets in a new directory. Creates a spectrogram 
    for each snippet, and saves those in a different, new directory.
        
        Resulting directories under self.out_dir will be:
         
                         self.out_dir
            spectrograms               wav-files

    '''
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, 
                 input_dir, 
                 output_dir, 
                 specific_species=None,
                 overwrite_freely=False
                 ):
        self.in_dir       = input_dir
        self.out_dir      = output_dir
        self.specific_species = specific_species
        self.overwrite_freely = overwrite_freely
        
        self.log = LoggingService()
        
        self.num_chopped = 0

        # Don't show the annoying deprecation
        # librosa.display() warnings about renaming
        # 'basey' to 'base' to match matplotlib: 
        warnings.simplefilter("ignore", category=MatplotlibDeprecationWarning)
        
        # Hide the UserWarning: PySoundFile failed. Trying audioread instead.
        warnings.filterwarnings(action="ignore",
                                message="PySoundFile failed. Trying audioread instead.",
                                category=UserWarning, 
                                module='', 
                                lineno=0)
        (num_audios, num_spectros) = self.chop_all()
        assert(num_audios == num_spectros)
        self.log.info(f"Created {num_audios} audio snippets and {num_spectros} spectrogram snippets")

    #------------------------------------
    # chop_all
    #-------------------

    def chop_all(self):
        '''
        Workhorse: Assuming self.in_dir is root of all
        species audio samples:
        
                        self.in_dir
                        
          Species1        Species2   ...     Speciesn
           smpl1_1.mp3      smpl2_1.mp3         smpln_1.mp3
           smpl1_2.mp3      smpl2_2.mp3         smpln_2mp3
                            ...
                            
        Chops each .mp3 (or .wav) file into window_len snippets.
        Saves those snippets in a new directory. Creates a spectrogram 
        for each snippet, and saves those in a different, new directory.
        
        Resulting directories under self.out_dir will be:
         
                         self.out_dir
            spectrograms               wav-files
            
        If self.specific_species is None, audio files under all
        species are chopped. Else, self.specific_species is 
        expected to be a list of species names that correspond
        to the names of species directories above: Species1, Species2, etc.
        
        Returns a 2-tuple: (number of created .wav audio snippet files,
                            number of created .png spectrogram snippet files,
        
        '''
        species_list = os.listdir(self.in_dir)
        # Root dir of the two dirs that will hold new 
        # audio snippet and spectrogram files
        utils.create_folder(self.out_dir, overwrite_freely=self.overwrite_freely)

        # Below the rootP
        spectrogram_dir_path = os.path.join(self.out_dir,'spectrograms/')
        wav_dir_path = os.path.join(self.out_dir,'wav-files/')
        utils.create_folder(spectrogram_dir_path, overwrite_freely=self.overwrite_freely)
        utils.create_folder(wav_dir_path, overwrite_freely=self.overwrite_freely)
        
        if self.specific_species == None:
            # Chop them all:
            for species in os.listdir(self.in_dir):
                # One dir each for the audio and spectrogram
                # snippets of one species:
                if (utils.create_folder(os.path.join(spectrogram_dir_path, species),
                                        overwrite_freely=self.overwrite_freely
                                        )
                and utils.create_folder(os.path.join(wav_dir_path, species),
                                        overwrite_freely=self.overwrite_freely)):
                    audio_files = os.listdir(os.path.join(self.in_dir, species))
                    num_files   = len(audio_files)
                    for i, sample_name in enumerate(audio_files):
                        # Chop one audio file:
                        self.log.info(f"Chopping {species} audio {i}/{num_files}")
                        self.chop_one_audio_file(self.in_dir, species, sample_name, self.out_dir)
                    self.num_chopped += num_files
                    
                else:
                    print(f"Skipping audio chopping for {species}")
        else:
            if self.specific_species in species_list:
                
                # Creation of spectrogram and/or audio output
                # directories may fail if one of the dirs already
                # exists, and user forbade replacement in the
                # dialog question:
                
                created_spectrogram_dir = utils.create_folder(os.path.join(spectrogram_dir_path, self.specific_species),
                                                              overwrite_freely=self.overwrite_freely
                                                              )
                created_audio_dir       = utils.create_folder(os.path.join(wav_dir_path, self.specific_species),
                                            overwrite_freely=self.overwrite_freely)
                 
                if created_spectrogram_dir and created_audio_dir:
                    for sample_name in os.listdir(os.path.join(self.in_dir, self.specific_species)):
                        self.chop_one_audio_file(self.in_dir, species, sample_name, self.out_dir)
                else:
                    print(f"Skipping audio chopping for {species}")
                    
        num_spectros = utils.find_in_dir_tree(spectrogram_dir_path, pattern='*.png')
        num_audios   = utils.find_in_dir_tree(wav_dir_path, pattern='*.wav')
        return (num_audios, num_spectros)
        
    #------------------------------------
    # create_spectrogram 
    #-------------------

    def create_spectrogram(self, sample_name, sample, sr, out_dir, n_mels=128):
        # Use bandpass filter for audio before converting to spectrogram
        audio = aug.filter_bird(sample, sr)
        mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels)
        # create a logarithmic mel spectrogram
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # create an image of the spectrogram and save it as file
        fig = plt.figure(figsize=(6.07, 2.02))
        librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', cmap='gray_r')
        plt.tight_layout()
        plt.axis('off')
        spectrogramfile = os.path.join(out_dir, sample_name + '.png')
        #*****
        #workaround for Exception in Tkinter callback
        import sys
        fig.canvas.start_event_loop(sys.float_info.min) 
        #*****
        plt.savefig(spectrogramfile, dpi=100, bbox_inches='tight', pad_inches=0, format='png', facecolor='none')
        plt.close()

    #------------------------------------
    # chop_one_audio_file 
    #-------------------

    def chop_one_audio_file(self, in_dir, species, sample_name, out_dir, window_len = 5):
        """
        Generates window_len second sound file snippets
        and associated spectrograms from sound files of
        arbitrary length. 
        
        Performs a time shift on all the wav files in the 
        species directories. The shift is 'rolling' such that
        no information is lost.
    
        :param in_dir: directory of the audio file to chop 
        :type file_name: str
        :param species: the directory names of the species to 
            modify the wav files of. If species=None, all 
            subdirectories will be processed.
        :type species: {None | [str]}
        :param sample_name: basefile name of audio file to chop
        :type sample_name: str
        :param out_dir: root directory under which spectrogram
            and audio snippets will be saved (in different subdirs)
        :type out_dir: str
        """

        orig, sample_rate = librosa.load(os.path.join(in_dir, species, sample_name))
        length = int(librosa.get_duration(orig, sample_rate))
        for start_time in range(length - window_len):
            window, sr = librosa.load(os.path.join(in_dir, species, sample_name),
                                      offset=start_time, duration=window_len)
            window_name = sample_name[:-len(".wav")] + '_sw-start' + str(start_time)
            self.create_spectrogram(window_name, 
                                    window, 
                                    sr, 
                                    os.path.join(out_dir, 'spectrograms/', species)
                                    )
            sf.write(os.path.join(out_dir, 'wav-files', species, window_name + '.wav'), window, sr)


# ------------------------ Main -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    parser.add_argument('input_dir',
                           metavar='IN_DIR',
                           type=str,
                           help='the path to original directory with .wav files')
    parser.add_argument('output_dir',
                           metavar='OUT_DIR',
                           type=str,
                           help='the path to output directory to write new .wav/.png files')
    parser.add_argument('-s', '--species',
                           type=str,
                           nargs='+',
                           help='repeatable: specific species to use sliding window on',
                           default=None)
    parser.add_argument('-y', '--overwrite_freely',
                        help='if set, overwrite existing out directories without asking; default: False',
                        action='store_true',
                        default=False
                        )

    # Execute the parse_args() method
    args = parser.parse_args()
    
    sound_chopper = SoundChopper(args.input_dir, 
                                 args.output_dir, 
                                 specific_species=args.species,
                                 overwrite_freely=args.overwrite_freely
                                 )
    sound_chopper.log.info("Done")