#!/usr/bin/env python3
'''
Created on Apr 29, 2021

@author: paepcke
'''

import argparse
import os
import sys

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



class PNGMetadataManipulator(object):
    '''
    classdocs
    '''
    
    #------------------------------------
    # list_metadata
    #-------------------

    @classmethod
    def extract_metadata(self, png_fname, show=False, printout=False):
        if show:
            # To save startup time, only load
            # matplotlib if needed:
            import matplotlib.pyplot as plt
        
        img, metadata = SoundProcessor.load_spectrogram(png_fname)

        if printout: 
            try:
                if len(metadata) == 0:
                    print("No metadata")
                for key, val in metadata.items():
                    print(f"{key} : {val}")
            except Exception as _e:
                print("No metadata available")

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_title(os.path.basename(png_fname))
            ax.set_axis_off()
            plt.imshow(img, cmap='gray')
            plt.show()

        return metadata
        
    #------------------------------------
    # set_metadata
    #-------------------

    @classmethod
    def set_metadata(self, png_fpath, info_to_set, outfile=None, setting=False):
        '''
        Modify metadata in a .png file. Distinguishes between
        replacing existing metadata (set == True), and adding
        to the existing info (set == False). Either way, takes
        a dict of metadata in info_to_set. 
        
        If outfile is None (or same as the input file png_fpath),
        the modification is in-place.
        
        :param png_fpath: input png file
        :type png_fpath: str
        :param info_to_set: dict of metadata information
        :type info_to_set: {str : str}
        :param outfile: if provided, create a new png file with the 
            provided metadata
        :type outfile: {None | str}
        :param setting: whether or not to replace existing metadata
            with info_to_set, or to add. Replacing only for common
            keys
        :type setting: bool
        '''
        
        if type(info_to_set) != dict:
            raise TypeError(f"info_to_set must be a dict, not {type(dict)}")
        
        img, metadata = SoundProcessor.load_spectrogram(png_fpath)
        if outfile is None:
            outfile = png_fpath
            
        if setting:
            metadata = info_to_set
        else:
            metadata.update(info_to_set)
            
        SoundProcessor.save_image(img, outfile, metadata)

# ------------------------ Main ------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Show PNG metadata, and image"
                                     )

    parser.add_argument('fname',
                        help='fully qualified path to .png file'
                        )
    parser.add_argument('-s', '--show',
                        help='show png image in addition to listing the metadata',
                        action='store_true')

    parser.add_argument('-p', '--printout',
                        help='if set, print metadate to console; default: True',
                        action='store_true',
                        default=True
                        )

    parser.add_argument('--set_info',
                        type=str,
                        nargs='+',
                        help='set metadata; repeatable; mutually exclusive with --add_info: key and value; ex.: --set_info foo 10 bar 20 -- fname.png',
                        default=None
                        )

    parser.add_argument('--add_info',
                        type=str,
                        nargs='+',
                        help='add metadata; repeatable; mutually exclusive with --set_info: key and value; ex.: --add_info foo 10 bar 20 -- fname.png',
                        default=None
                        )
    
    parser.add_argument('-o', '--outfile',
                        help='full path to where new png is to be written. Only for --add_info'
                        )

    parser.add_argument('-f', '--force',
                        help="if set, don't ask confirmation when replacing metadata (rather than adding): do confirm",
                        action='store_true',
                        default=False
                        )

    args = parser.parse_args()


    # All the checks below should be simplified.
    # Summary:
    #    o If args say to just show metadata, do that and quit
    #    o When asked to modify metadata in a .png file, 
    #         distinguish between *adding* to existing, vs.
    #         *replacing*. Provide warning if replacing

    # Sanity Checks:
    
    # File exists?
    if not os.path.exists(args.fname):
        print(f"File {args.fname} not found")
        sys.exit(1)

    setting = args.set_info is not None and len(args.set_info) > 0
    adding  = args.add_info is not None and len(args.add_info) > 0
    
    if args.printout and not setting and not adding:
        # Just want to see metadata:
        PNGMetadataManipulator.extract_metadata(args.fname, show=args.show, printout=args.printout)
        sys.exit(0)

    # Enforce --set_info and --add_info mutually exclusive:
    if adding and setting:
        print("Cannot simultaneously set metadata and add to existing metadata")
        sys.exit(1) 
    
    # For convenience:
    info_to_add = args.add_info
    info_to_set = args.set_info
    
    # Enforce args to set_info or add_info being 
    # equal length, i.e. having 'names' and 'values'
    # as pairs:
    if (setting and len(info_to_set) % 2 != 0) \
       or (adding and len(info_to_add) % 2 != 0):
        print("Info entries must be pairs of keys and values; length is odd numbered here")
        sys.exit(1)

    # Safety precaution just for setting
    # (and thereby overwriting) metadata:
    
    if setting and not args.force:
        if not Utils.user_confirm("Really want to overwrite png file metadata? (N/y)", default='n'):
            print("Canceling")
            sys.exit(0)

    if args.printout:
        print("Metadata before:")
        PNGMetadataManipulator.extract_metadata(args.fname, show=args.show, printout=args.printout)
        print("")

    # Setting info_to_set:
    if args.outfile is None:
        # Overwrite the input file,
        # i.e. add metadata in place:
        outfile = args.fname
    else:
        outfile = args.outfile

    # Key/val pairs for setting metadata 
    # came in as a sequence of keys and values
    # on the cmd line. Turn that sequence
    # into a dict, as needed by set_metadata: 
    info_dict = {}
    # Grab the CLI metadata modification info 
    # (i.e. key/vals to set or key/vals to add):
    cli_dict_spec = info_to_set if info_to_set is not None else info_to_add

    # Grab pairs from the CLI (each being a key and a val):
    for idx in range(0, len(cli_dict_spec)-1, 2):
        info_dict[cli_dict_spec[idx]] = cli_dict_spec[idx+1]

    if setting:
        PNGMetadataManipulator.set_metadata(args.fname, info_dict, outfile, setting=True)
    else:
        # Just adding to metadata
        PNGMetadataManipulator.set_metadata(args.fname, info_dict, outfile, setting=False)

    # If printing to console, show metadata 
    # after this update:
    
    if args.printout:
        print("Metadata after:")
        PNGMetadataManipulator.extract_metadata(args.fname, show=False, printout=True)
