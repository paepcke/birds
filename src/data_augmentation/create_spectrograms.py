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
    def create_spectrogram(cls, in_dir, out_dir, num=None):
        '''
        Given a directory with sound files, create
        a spectrogram for each, and deposit them,
        in the out_dir
        '''
        # Go through the absolute paths of the in_dir:
        for i, aud_file in enumerate(Utils.listdir_abs(in_dir)):
            sound, sr = SoundProcessor.load(aud_file)
            # Get parts of the file: root, fname, suffix
            path_el_dict  = Utils.path_elements(aud_file)
            # Make dest 
            dest_path = Path(out_dir).joinpath(path_el_dict['fname'] + '.png')
            
            cls.log.info(f"Creating spectrogram for {os.path.basename(dest_path)}")
            SoundProcessor.create_spectrogram(sound, sr, dest_path)
            if num is not None and i >= num:
                break 

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

        Spectrogrammer.create_spectrogram(args.in_dir, 
                                          args.outdir,
                                          num=args.num)