#!/usr/bin/env python3
'''
Created on Aug 31, 2021

@author: paepcke
'''

import argparse
import datetime
import os
from pathlib import Path
import sys

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from result_analysis.charting import Charter


class RecordingsInventory:
    '''
    Given the root of subdirectories with species
    audio recordings do the following:
    
        o Inspect each audio file to recover its recording time
        o Generate a dataframe holding the total recording
          time for each recording:
          
                             total_recording_length (secs)
              species1              10.4
              species2               6
                      ...
                      
        o Create a manifest subdirectory of the given root's parent
          called "<Audio_Manifest_<root_dir_name>"
          
        o Save the dataframe into the manifest subdir as 'manifest.json'
        o Save an optionally command line specified message to README.txt
          in the manifest subdir
        o If requested, generate a barchart of the results, and
          save it in <manifest subdir>/audio_recording_distribution.pdf
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, species_root, message=None, chart_result=False):
        '''
        Constructor
        
        :param species_root: root of species subdirectories
        :type species_root: str
        :param message: message to include with the audio manifest
        :type message: {None | str}
        :param chart_result: whether or not to include a 
            barchart of the result
        :type chart_result: bool
        '''
        
        if not os.path.exists(species_root) or not os.path.isdir(species_root):
            print(f"Directory must exist, but given {species_root}")
            sys.exit(1)
        print("Begin recording inventory...")
        start_time = datetime.datetime.now()
        # Get df:
        #                   total_recording_length
        #    species1              10.4
        #    species2            ...
        
        df = SoundProcessor.recording_lengths_by_species(species_root)
        end_time = datetime.datetime.now()
        duration_str = Utils.time_delta_str(end_time - start_time)
        print(f"Done with recording inventory ({duration_str}).")

        species_root_path = Path(species_root)
        # Ensure existence of destination for manifest directory:
        manifest_dir_path = species_root_path.parent.joinpath(f"Audio_Manifest_{species_root_path.stem}")
        if not manifest_dir_path.exists():
            manifest_dir_path.mkdir()
            
        # Generate README.txt if requested:
        if message is not None:
            with open(manifest_dir_path.joinpath('README.txt'), 'w') as fd:
                fd.write(message)
        
        # Write df as manifest.json:
        manifest_fname = manifest_dir_path.joinpath('manifest.json')
        with open(manifest_fname, 'w') as fd:
            df.to_json(fd)

        print(df.to_string())

        if chart_result:
            fig_title = f"Recording Durations in {species_root_path.stem}"
            self.chart_results(df, manifest_dir_path, fig_title)
            
            
    #------------------------------------
    # chart_results
    #-------------------
    
    def chart_results(self, df, manifest_dir_path, fig_title):

        print(f"Preparing bar chart...")
        fig = plt.figure()
        ax = Charter.barchart(df['total_recording_length (secs)'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # Get current, raw seconds y-tick labels, like:
        #    [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]

        ytick_locs = ax.get_yticks().tolist()
        # Magic:
        ax.yaxis.set_major_locator(mticker.FixedLocator(ytick_locs))
        
        # Replace the seconds ints with 
        # hrs:mins:secs:
        
        new_ylabels = [str(datetime.timedelta(seconds=secs)) for secs in ytick_locs]
        ax.set_yticklabels(new_ylabels)
        
        # Make x-axis species labels small, b/c there may
        # be many of them:
        
        ax.set_xticklabels(ax.get_xticklabels(), fontdict={'fontsize' : 9})

        # Y Axis label:
        ax.set_ylabel('Time (hrs:mins:secs)')
        
        ax.set_title(fig_title)
        
        fig.tight_layout()
        fig.show()
        
        # Save as pdf:
        fig_fname = manifest_dir_path.joinpath('audio_recording_distribution.pdf')
        fig.savefig(fig_fname)

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Tally recording length of species tree."
                                     )

    parser.add_argument('species_root',
                        help='root of species subdirectories'
                        )
    parser.add_argument('-c', '--chart',
                        help='whether or not to show barchart of result',
                        action='store_true',
                        default=True
                        )
    parser.add_argument('-m', '--message',
                        help='message to include with the manifest',
                        default=None
                        )

    args = parser.parse_args()

    RecordingsInventory(args.species_root,
                        message=args.message,
                        chart_result=args.chart)
    
    if args.chart:
        input("Hit any key to close figure and exit...")
    