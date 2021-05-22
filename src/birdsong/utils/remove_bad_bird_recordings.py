#!/usr/bin/env python3
'''
Created on May 22, 2021

@author: paepcke
'''
import argparse
import os
from pathlib import Path
import sys

from data_augmentation.utils import Utils


class BirdRecordingCuller:
    

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, dir_root, fextension):
        
        # Good recording code ranges:
        good_rec_id_rngs = [range(50000, 70001),
                            range(170000, 180001),
                            ] 
        
        # Get all audio file paths relative
        # to dir_root:
        
        pattern = f'*.{fextension}' 
        wav_paths = Utils.find_in_dir_tree(dir_root, 
                                           pattern=pattern)
                                           
        
        #*********
        # wav_paths = ['/foo/bar/AM01_20190711_049999.wav', # no
        #   		   '/foo/bar/AM01_20190711_050000.wav', # yes
        #   		   '/foo/bar/AM01_20190711_070000.wav', # yes
        #   		   '/foo/bar/AM01_20190711_070001.wav', # no
        #   		   '/foo/bar/AM01_20190711_169999.wav', # no
        #   		   '/foo/bar/AM01_20190711_170000.wav', # yes
        #   		   '/foo/bar/AM01_20190711_170001.wav', # no
        # ]
        # #********* 
        # Get just the filename without parents
        # and extension:
        to_delete = []
        for aud_path in wav_paths:
            ser_num = self.extract_ser_num(aud_path)
            if   ser_num in good_rec_id_rngs[0] \
              or ser_num in good_rec_id_rngs[1]:
                continue
            else:
                to_delete.append(aud_path)

        print(f"Examined {len(wav_paths)} {pattern} files...")
        if len(to_delete) > 0:
            if Utils.user_confirm(f"Delete {len(to_delete)} aud files? (N/y):", default='n'):
                num_deleted = 0
                for fname in to_delete:
                    try:
                        os.remove(fname)
                    except Exception as e:
                        print(f"Could not delete {fname}: {repr(e)}")
                    else:
                        num_deleted += 1
                        
                print(f"Removed {num_deleted} files")
            else:
                print('Canceling')
        else:
            print('No files are out of good recorder serial number ranges')

    #------------------------------------
    # extract_ser_num 
    #-------------------
    
    def extract_ser_num(self, fpath):
        fname_no_ext = Path(fpath).stem
        ser_num = int(fname_no_ext.split('_')[-1])
        return ser_num

# -------------------- Main -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Remove .wav or .png files outside Kelley's acceptable recorder serial numbers"
                                     )

    parser.add_argument('-r', '--rootdir',
                        help='fully qualified path to root of files to cull')
    parser.add_argument('-e', '--extension',
                        choices=['.wav', '.png'],
                        help='file extension among which to cull')

    args = parser.parse_args()

    if args.rootdir is None \
        or not os.path.exists(args.rootdir) \
        or not os.path.isdir(args.rootdir):
        print(f"Argument rootdir must be an existing dir, not {args.rootdir}")
        sys.exit(1)

    if args.extension not in ['.wav', '.mp3', '.png']:
        print("File extension must be '.wav', '.mp3', or '.png'")
        sys.exit(1)
    
    BirdRecordingCuller(args.rootdir, args.extension)
    
    #BirdRecordingCuller('/Users/paepcke/EclipseWorkspacesNew/birds/src/data_augmentation/tests/sound_data/')
