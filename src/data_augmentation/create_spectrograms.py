#!/usr/bin/env python3
'''
Created on Apr 26, 2021

@author: paepcke
'''

import argparse
import os
from pathlib import Path
import sys

from logging_service import LoggingService

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils


class Spectrogrammer:
    '''
    Spectrogram creation and manipulation
    '''
    
    # Get a Python logger that is
    # common to all modules in this
    # package:

    log = LoggingService()

    @classmethod
    def create_spectrograms(cls, in_dir, out_dir, num=None):
        '''
        Given a directory with sound files, create
        a spectrogram for each, and deposit them,
        in the out_dir
        '''
        
        dirs_filled = []
        
        # Is in_dir the root of subdirectories, each holding
        # audio files of one species? Or does in_dir hold the
        
        dir_content = Utils.listdir_abs(in_dir)
        dirs_to_do = [candidate
                      for candidate
                      in dir_content
                      if os.path.isdir(candidate)
                      ]
        if len(dirs_to_do) == 0:
            # Given dir directly contains the audio files:
            audio_dirs = [in_dir]
        else:
            audio_dirs = dirs_to_do
            
        # Go through the absolute paths of the director(y/ies):
        for one_dir in audio_dirs:
            # At the destination, create a directory
            # named the same as one_dir, which we are about
            # to process:
            dst_dir = os.path.join(out_dir, Path(one_dir).stem)
            
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
                
            for i, aud_file in enumerate(Utils.listdir_abs(one_dir)):
                if not Utils.is_audio_file(aud_file):
                    continue
                sound, sr = SoundProcessor.load(aud_file)
                # Get parts of the file: root, fname, suffix
                path_el_dict  = Utils.path_elements(aud_file)
                new_fname = path_el_dict['fname'] + '.png'
                
                dst_path = os.path.join(dst_dir, new_fname)
                
                cls.log.info(f"Creating spectrogram for {os.path.basename(dst_path)}")
                SoundProcessor.create_spectrograms(sound, sr, dst_path)
                if num is not None and i >= num-1:
                    break
            dirs_filled.append(dst_dir)
        
        return dirs_filled 

# ------------------------ Main ------------
if __name__ == '__main__':
    
        parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         description="Here is what this package does."
                                         )
    
        parser.add_argument('-n', '--num',
                            type=int,
                            help='limit processing to first n audio files',
                            default=None)
        
        parser.add_argument('indir',
                            help='directories of audio files whose spectrograms to create',
                            default=None)
        parser.add_argument('outdir',
                            help='destination of spectrogram files',
                            default=None)
    
        args = parser.parse_args()

        Spectrogrammer.create_spectrograms(args.in_dir, 
                                          args.outdir,
                                          num=args.num)