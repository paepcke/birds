#!/usr/bin/env python3
'''
Created on Apr 29, 2021

@author: paepcke
'''

import argparse
import os
from pathlib import Path
import sys
import pprint

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



class PNGMetadataManipulator(object):
    '''
    View, set, or clear the metadata contained in
    .png files. Capabilities:
    
        o for each .png file, return metadata to 
          caller, plus optionally print metadata to
          stdout and optionally show the .png
        o operate on individual files, or entire
          directories
        o retrieve either (list of) metadata, or 
          obtain a generator yielding tuples (fname, metadata)
    '''
    
    #------------------------------------
    # list_metadata
    #-------------------

    @classmethod
    def extract_metadata(cls, png_src, show=False, printout=False):
        '''
        Given a .png file path, or a directory:
        
            o extract metadata
            o print the metadata if requested (printout == True)
            o show the .png file if requested (show == True)
            o return: if png_src is a dir: a list of metadata dicts
                      else just one metadata dict
        
        If png_src is a directory, all .png files below the
        directory will be found recursively, and printed/shown/returned
        as requested. 
        
        :param png_src: .png file or directory tree root containing 
            .png file
        :type png_src: src
        :param show: whether or not to display the .png file
        :type show: bool
        :param printout: whether or not to print metadata
            to stdout
        :type printout: bool
        :return: list of metadata dicts, if png_src is a dir, 
            else one metadata dict
        :rtype {{src : src} | [{src : src}]}
        '''
        if show:
            # To save startup time, only load
            # matplotlib if needed:
            import matplotlib.pyplot as plt

        if os.path.isfile(png_src):
            img, metadata = SoundProcessor.load_spectrogram(png_src)
            snip_path_md_gen = iter([(png_src, metadata)])
        else:
            snip_path_md_gen = cls.metadata_list(png_src)
            
        md_list = []
        for snippet_path, metadata in snip_path_md_gen:
            md_list.append(metadata)
    
            if printout: 
                try:
                    print(f"{Path(snippet_path).name}---", end='')
                    # Very inefficient: if using metadata_list()
                    # above, we lost the img array. Could fix,
                    # but not worth it for now:
                    
                    try:
                        print(f"Shape: {img.shape}")
                    except UnboundLocalError:
                        img, metadata = SoundProcessor.load_spectrogram(snippet_path)
                        print(f"Shape: {img.shape}")

                    if len(metadata) == 0:
                        print("No metadata")
                    pprint.pprint(metadata, 
                                  sort_dicts=False,
                                  indent=4)

                    #for key, val in metadata.items():
                    #    print(f"{key} : {val}")
                except Exception as _e:
                    print("No metadata available")
    
            if show:
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.set_title(os.path.basename(snippet_path))
                ax.set_axis_off()
                plt.imshow(img, cmap='gray')
                plt.show()

        return md_list[0] if os.path.isfile(png_src) else md_list
        
    #------------------------------------
    # set_metadata
    #-------------------

    @classmethod
    def set_metadata(cls, png_fpath, info_to_set, outfile=None, setting=False):
        '''
        Modify metadata in a .png file. Distinguishes between
        replacing existing metadata (setting == True), and adding
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

    #------------------------------------
    # clear_metadata
    #-------------------

    @classmethod
    def clear_metadata(cls, png_fpath, outfile=None):
        '''
        Clear all metadata from given png file. If outfile
        is none, clears in place.
        
        :param png_fpath: png file whose metadata to clear
        :type png_fpath: src
        :param outfile: if given, copy of png_fpath file with
            metadata cleared
        :type outfile: {None | str}
        '''
        cls.set_metadata(png_fpath, {}, outfile, setting=True)

    #------------------------------------
    # metadata_list
    #-------------------
    
    @classmethod
    def metadata_list(cls, png_src):
        '''
        Generator that takes the path to a
        directory tree containing .png files.
        Walks the tree, and for each .png file
        returns a tuple (<absolute-file-path, <metadata-dict>) 

        :param png_src: root of directorty tree
        :type png_src: str
        :returns successively with each call: a tuple
            of file path and metadata
        :rtype (str : {str : str}
        :raise FileNotFoundError if given dir does not
            exist, or is not a directory
        '''
        
        if not os.path.isdir(png_src):
            raise FileNotFoundError(f"Must provide an existing directory of png files, not {png_src}")
        if os.path.isfile(png_src):
            metadata = cls.extract_metadata(png_src)
            yield (png_src, metadata)
            return
            
        for root, _dirs, files in os.walk(png_src):
            for fname in files:
                if Path(fname).suffix != '.png':
                    continue 
                fpath = os.path.join(root, fname)
                metadata = cls.extract_metadata(fpath)
                yield (fpath, metadata)

    #------------------------------------
    # species_distribution
    #-------------------
    
    @classmethod
    def snippet_distribution(cls, png_root_dir):
        '''
        Given a root directory with subdirectories that
        contain .png files, return a dict mapping 
        species to number of samples for that species.
        
        PNG files contained in png_root_dir itself are
        considered in species the name of png_root_dir.
        
        Assumption: the name of each subdirectory is 
            the name of a species.

        :param png_root_dir: root of directories that contain
            png snippets
        :type png_root_dir: src
        :return: mapping of species name to number of snippets
        :rtype: {str : int}
        '''
        snip_distrib = {}
        for root, _dirs, files in os.walk(png_root_dir):
            png_files = list(filter(lambda fname: fname.endswith('.png'), files))
            if len(png_files) > 0:
                # Species name is the current dir:
                snip_distrib[Path(root).name] = len(png_files)            
        return snip_distrib

    #------------------------------------
    # handle_metadata_modification 
    #-------------------
    
    @classmethod
    def handle_metadata_modification(cls, args):

        setting = args.set_info is not None and len(args.set_info) > 0
        adding  = args.add_info is not None and len(args.add_info) > 0
        
        # Is snippet_src a directory for which we are
        # to list all the metadata?
        if os.path.isdir(args.snippet_src):
            # Cannot also have set_info or add_info
            if setting or adding:
                print("Cannot set or add info which snippet source is a directory")
                sys.exit(1)
    
        if args.printout and not setting and not adding:
            # Just want to see metadata:
            PNGMetadataManipulator.extract_metadata(args.snippet_src, show=args.show, printout=args.printout)
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
            PNGMetadataManipulator.extract_metadata(args.snippet_src, show=args.show, printout=args.printout)
            print("")
    
        # Setting info_to_set:
        if args.outfile is None:
            # Overwrite the input file,
            # i.e. add metadata in place:
            outfile = args.snippet_src
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
            PNGMetadataManipulator.set_metadata(args.snippet_src, info_dict, outfile, setting=True)
        else:
            # Just adding to metadata
            PNGMetadataManipulator.set_metadata(args.snippet_src, info_dict, outfile, setting=False)
    
        # If printing to console, show metadata 
        # after this update:
        
        if args.printout:
            print("Metadata after:")
            PNGMetadataManipulator.extract_metadata(args.snippet_src, show=False, printout=True)

# ------------------------ Main ------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Show PNG metadata, and image"
                                     )

    parser.add_argument('snippet_src',
                        help='fully qualified path to .png file'
                        )
    parser.add_argument('-s', '--show',
                        help='show png image in addition to listing the metadata; default: False',
                        action='store_true',
                        default=False)

    parser.add_argument('-p', '--printout',
                        help='if set, print metadate to console; default: False',
                        action='store_true',
                        default=False
                        )

    parser.add_argument('-d', '--distribution',
                        help='if set, print how many snippets are in each species subdir',
                        action='store_true',
                        default=False
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



    # Sanity Checks:

    # Check for nothing having been requested:
    if not (args.printout or
            args.distribution or
            args.set_info or
            args.add_info
            ):
        print("Nothing request; set one or more of the parameters:")
        parser.print_help()
    
    # File/dir exists?
    if not os.path.exists(args.snippet_src):
        print(f"File {args.snippet_src} not found")
        sys.exit(1)
        
    # If printout of snippet distribution by species
    # is requested, do that now:
    
    if args.distribution:
        distrib_dict = PNGMetadataManipulator.snippet_distribution(args.snippet_src)
        pprint.pprint(distrib_dict)
        
    setting = args.set_info is not None and len(args.set_info) > 0
    adding  = args.add_info is not None and len(args.add_info) > 0
    if setting or adding:
        # Add or modify metadata. If --printout
        # is set, this method will do the printing: 
        PNGMetadataManipulator.handle_metadata_modification(args)
    elif args.printout:
        PNGMetadataManipulator.extract_metadata(args.snippet_src, show=False, printout=True)
