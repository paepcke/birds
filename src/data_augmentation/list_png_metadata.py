#!/usr/bin/env python3
'''
Created on Apr 29, 2021

@author: paepcke
'''

import argparse
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from data_augmentation.sound_processor import SoundProcessor

class PNGMetadataLister(object):
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
    # add_metadata
    #-------------------

    @classmethod
    def add_metadata(self, png_fpath, info, outfile=None):
        
        if type(info) != dict:
            raise TypeError(f"info must be a dict, not {type(dict)}")
        
        img, _metadata = SoundProcessor.load_spectrogram(png_fpath)
        if outfile is None:
            outfile = png_fpath
        SoundProcessor.save_image(img, outfile, info)

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

    parser.add_argument('-p-', '--printout',
                        help='if set, print metadate to console; default: True',
                        default=True
                        )

    parser.add_argument('--info',
                        type=str,
                        nargs='+',
                        help='set metadata; repeatable: key and value; ex.: --info foo 10 bar 20',
                        default=None
                        )

    parser.add_argument('-o', '--outfile',
                        help='full path to where new png is to be written. Only for --add_info'
                        )

    args = parser.parse_args()

    # File exists?
    if not os.path.exists(args.fname):
        print(f"File {args.fname} not found")
        sys.exit(1)

    # User wants to set metadata in image file?
    info = args.info
    if info is not None and len(info) % 2 != 0:
        print("Info entries must be pairs of keys and values; this entry's length is odd numbered")
        sys.exit(1)

    if info is not None and args.printout:
        print("Metadata before:")

    PNGMetadataLister.extract_metadata(args.fname, show=args.show, printout=args.printout)

    if info is None:
        # All done:
        sys.exit(0)

    # Setting info:
    if args.outfile is None:
        # Overwrite the input file,
        # i.e. add metadata in place:
        outfile = args.fname
    else:
        outfile = args.outfile

    PNGMetadataLister.add_metadata(args.fname, info, outfile)

    # If printing to console, show metadata 
    # after this update:
    
    if args.printout:
        PNGMetadataLister.extract_metadata(args.fname, show=False, printout=True)

