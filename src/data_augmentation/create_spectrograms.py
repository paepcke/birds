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
from data_augmentation.utils import Utils, WhenAlreadyDone


class SpectrogramCreator:
    '''
    Spectrogram creation and manipulation
    '''
    
    # Get a Python logger that is
    # common to all modules in this
    # package:

    log = LoggingService()

    @classmethod
    def create_spectrograms(cls, 
                            in_dir, 
                            out_dir, 
                            num=None,
                            overwrite_policy=False
                            ):
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
        
        cls.log.info(f"Number of audio file dirs for which to create spectrograms: {len(audio_dirs)}")
        # Go through the absolute paths of the director(y/ies):
        for one_dir in audio_dirs:
            # At the destination, create a directory
            # named the same as one_dir, which we are about
            # to process:
            dst_dir = os.path.join(out_dir, Path(one_dir).stem)
            
            if not os.path.exists(dst_dir):
                cls.log.info(f"Creating dest dir {dst_dir}")
                os.makedirs(dst_dir)
                
            # Check overwrite policy: if the destination
            # directory is not empty, follow the policy:
            existing_files = Utils.listdir_abs(dst_dir)
            
            # If any files already exist at the destination,
            # check the overwrite policy. If OVERWRITE, remove
            # the files right now: 
            if len(existing_files) > 0:
                if overwrite_policy == WhenAlreadyDone.OVERWRITE:
                    cls.log.info(f"Removing any existing files in {dst_dir}")
                    for fname in existing_files:
                        os.remove(fname)

            for i, aud_file in enumerate(Utils.listdir_abs(one_dir)):
                #cls.log.info(f"Creating spectros for audio in {one_dir}")
                if not Utils.is_audio_file(aud_file):
                    continue
                sound, sr = SoundProcessor.load_audio(aud_file)
                # Get parts of the file: root, fname, suffix
                path_el_dict  = Utils.path_elements(aud_file)
                new_fname = path_el_dict['fname'] + '.png'
                
                dst_path = os.path.join(dst_dir, new_fname)
                # If dst exists, ask whether to overwrite;
                # skip file if not:
                if os.path.exists(dst_path) and overwrite_policy == WhenAlreadyDone.ASK:
                    if not Utils.user_confirm(f"Overwrite {dst_path}?", False):
                        # Not allowed to overwrite
                        continue
                    else:
                        os.remove(dst_path)
                
                #cls.log.info(f"Creating spectrogram for {os.path.basename(dst_path)}")
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

    parser.add_argument('-y', '--overwrite_policy',
                    help='if set, overwrite existing out directories without asking; default: False',
                    action='store_true',
                    default=False
                    )

    parser.add_argument('indir',
                        help='directories of audio files whose spectrograms to create',
                        default=None)
    parser.add_argument('outdir',
                        help='destination of spectrogram files',
                        default=None)

    args = parser.parse_args()

    # Turn the overwrite_policy arg into the
    # proper enum mem
    if args.overwrite_policy:
        overwrite_policy = WhenAlreadyDone.OVERWRITE
    else:
        overwrite_policy = WhenAlreadyDone.ASK
        
    SpectrogramCreator.create_spectrograms(args.indir, 
                                           args.outdir,
                                           num=args.num,
                                           overwrite_policy=overwrite_policy
                                           )
        