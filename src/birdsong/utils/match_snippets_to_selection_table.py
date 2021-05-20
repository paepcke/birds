'''
Created on May 13, 2021

@author: paepcke
'''
import argparse
import csv
import os
from pathlib import Path
import sys

from birdsong.utils.utilities import FileUtils
from data_augmentation.list_png_metadata import PNGMetadataManipulator
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils


class SnippetSelectionTableMapper:
    '''
    
    Matches and labels spectrogram snippets with 
    entries in a Raven selection table. 'Matching'
    means a time overlap between the snippet's beging
    and end times, and a selection table row's begin
    and end time. 
    
    In case of a match, the row's species determination S is 
    noted. For snippets that do not match any selection table
    row, S is set to be 'noise'. 
    
    Each snippet's png metadata will receive an additional 
    key called 'confirmed_content' that will contain S.
    
    In addition, snippets that do match receive two additional
    key/val pairs: freq_low and freq_high. These values are
    taken from the matching selection table row: 

    Raven selection tables for birds are csv files:

        Selection           row number
        View                not used
        Channel             not used
        Begin Time (s)      begin of vocalization in fractional seconds
        End Time (s)        end of vocalization in fractional seconds
        Low Freq (Hz)       lowest frequency within the lassoed vocalization
        High Freq (Hz)      highest frequency within the lassoed vocalization
        species             four-letter species name
        type                {song, call, call-1, call-trill}
        number              not used
        mix                 comma separated list of other audible species 

    When a row that matches a snippet contains species in the 'mix' column,
    copies of the snippets are made, and each is labeled with
    one of those species.
    
    Snippet metadata is expected to look like this:
    
		 sr : 22050
		 duration : 115.0
		 species : batpig1
		 duration(secs) : 4.990916431166734
		 start_time(secs) : 61.8873637464675
		 end_time(secs) : 66.87828017763424
    

    Snippet metadata added in case of a match:
    
         'confirmed_content' : <species from the selection table row; may be 'noise'>
         'low_freq'          : <low bound of frequency involved in vocalization
         'high_freq'         : <high bound of frequency involved in vocalization
         'multiple_species'  : <comma-separated list of species heard simultaneously>
         'type'              : <whether Song/Call/Call-1/Call-Trill...>

    IMPORTANT: procedures rely on each spectrogram
    	snippet's metadata to include the time range
    	that the snippet covers in the larger spectrogram
    	from which it is snipped. This metadata is added
    	to the .png files created by chop_spectrograms.py

    '''


    def __init__(self, selection_tbl_csv, snippets_path, out_dir):
        '''
        Constructor
        '''
        if not os.path.exists(selection_tbl_csv) or \
           not os.path.isfile(selection_tbl_csv):
            print(f"Raven selection table .csv file {selection_tbl_csv} does not exist or is not a file")
            sys.exit(1)
            
        if not os.path.exists(snippets_path) or not os.path.isdir(snippets_path):
            print(f"Spectrogram snippets {snippets_path} does not exist or is not a directory")
            sys.exit(1)

        if os.path.exists(out_dir) and os.path.isfile(out_dir):
            print(f"Out directory {out_dir} is an existing file; quitting")
            sys.exit(1)

        # If out_dir does not exist, create it,
        # and all dirs along the path:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Read Raven selection table, obtaining
        # a start-time sorted list of dicts. Each
        # dict will contain the information of one
        # spectrogram selection:
         
        selections_sorted = Utils.read_raven_selection_table(selection_tbl_csv)
        
        # If out_dir is None, modify snippets in place.
        # Use caution; maybe make a backup of the snippets,
        # or specify an out_dir:
        
        if out_dir is None:
            out_dir = snippets_path
        
        self.match_snippets(selections_sorted, snippets_path, out_dir)

    #------------------------------------
    # match_snippets
    #-------------------
    
    def match_snippets(self, selections, snippets_path, outdir):
        '''
        For each snippet S, examines the time span covered by S.
        Finds the selection table row (if any) whose begin/end
        times overlap with S's time span.
        
        Modifies the snippet's metadata as per this class' 
        header comment.
        
        The selections are expected to be a list of dicts as
        produced by Utils.read_raven_selection_table(). This
        means:
        
           o Dicts are sorted by selection start time
           o the d['mix'] is a possibly empty list of species
                names.

        :param selections: list of dicts, each containing the 
            information of one selection table row. 
        :type selections: {str : str}
        :param snippets_path: path to directory with spectrogram
            snippets
        :type snippets_path: str
        :param outdir: directory where to write the updated
            snippets. Value is allowed to be same as snippets_path,
            but then the snippets will be updated in place
        :type outdir: str
        '''

        # One spectrogram snippet at a time:
        for snippet in os.listdir(snippets_path):
            snip_fname   = os.path.join(snippets_path, snippet)
            spectro_arr, metadata = SoundProcessor.load_spectrogram(snip_fname)
            #**********
            try:
                snippet_tstart = metadata['start_time(secs)']
            except Exception as e:
                print(e)
            #**********
            snippet_tend   = metadata['end_time(secs)']
        
            # Find the index of the select table row
            # whose time interval includes the snippet
            # start time:
            row_idx = Utils.binary_in_interval_search(
                selections,
                snippet_tstart,
                'Begin Time (s)',  # Key for low time bound in selection rows
                'End Time (s)'     # Key for high time bound in selection rows
                )
            if row_idx == -1:
                # Start of snippet is not in a selection,
                # What about the end of the snippet?
                row_idx = Utils.binary_in_interval_search(
                    selections,
                    snippet_tend,
                    'Begin Time (s)',  # Key for low time bound in selection rows
                    'End Time (s)'     # Key for high time bound in selection rows
                    )
            
            if row_idx == -1:
                # This snippet was not involved in
                # any of the human-created selection
                # rectangles:
                metadata['confirmed_content'] = 'noise'
                # Save the updated snippet:
                snip_outname = os.path.join(outdir, 'noise', snippet)
                FileUtils.ensure_directory_existence(snip_outname)
                SoundProcessor.save_image(spectro_arr, snip_outname , metadata)
                continue

            # Row index is > -1: we identified a snippet
            # that was labeled in the selection table:
            selection = selections[row_idx]
            
            low_f     = selection['Low Freq (Hz)']
            high_f    = selection['High Freq (Hz)']
            species   = selection['species']
            voc_type  = selection['type'] # Song/Call/Song-Trill, etc.
            # Get possibly empty list of species
            # names that also occur in the selection:
            multiple_species = selection['mix']

            metadata['confirmed_content'] = species
            metadata['low_freq'] = low_f
            metadata['high_freq'] = high_f
            metadata['type'] = voc_type
            metadata['multiple_species'] = multiple_species

            # Save the updated snippet:
            snip_outname = os.path.join(outdir, species, snippet)
            FileUtils.ensure_directory_existence(snip_outname)
            SoundProcessor.save_image(spectro_arr, snip_outname , metadata)

            # If the snippet matched, and contained multiple
            # overlapping calls, create a copy of the snippet
            # for each species:
            if len(multiple_species) > 0:
                # Ensure the human coder did not include
                # the primary species in the list of overlaps:
                try:
                    del multiple_species[multiple_species.index(species)]
                except (IndexError, ValueError):
                    # All good, species wasn't in the list of additionals
                    pass
                for overlap_species in multiple_species:
                    metadata['confirmed_content'] = overlap_species
                    # New name for a copy of this snippet:
                    p = Path(snip_outname)
                    new_fname = p.stem + f"_{overlap_species}" + p.suffix
                    new_snip_outname = os.path.join(outdir, overlap_species, new_fname)
                    FileUtils.ensure_directory_existence(new_snip_outname)
                    
                    SoundProcessor.save_image(spectro_arr, new_snip_outname , metadata)

# ------------------------ Main ------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Match spectrogram snippets to Raven selection table."
                                     )

    parser.add_argument('selection_table_csv',
                        help='Path to Raven selection table')

    parser.add_argument('snippets_path',
                        help='Path directory of snippets with embedded time information')


    args = parser.parse_args()

    matcher = SnippetSelectionTableMapper(args.selection_table_csv, args.snippets_path)
