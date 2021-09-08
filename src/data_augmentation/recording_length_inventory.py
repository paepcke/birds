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
from logging_service import LoggingService


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
          
    The dataframe is available as <inst>.inventory. The manifest directory
    is available as <inst>.manifest_dir_path
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 species_root, 
                 message=None, 
                 chart_result=False, 
                 print_results=True,
                 inventory=None,
                 num_workers=None
                 ):
        '''
        Given the directory below which species recordings
        subdirectories lie, add all recording durations separately
        for each species. Save the result in file manifest.json
        in a sibling directory to species_root. That manifest directory
        will be called Audio_Manifest_<species_root-dirname>.
        
        If the recording times were already computed by the client,
        the dataframe may be passed in the inventory argument. 
        
        The message is placed in file README.txt in the manifest
        directory. If chart_results is True, a barchart in pdf
        format will also be created and placed in the manifest dir.
        
        The print_results arg controls whether results are printed
        to the console. Useful when using this file from the command
        line.
        
        Raises FileNotFoundError if no audio is found under the 
        species_root directory.
        
        :param species_root: root of species subdirectories
        :type species_root: str
        :param message: message to include with the audio manifest
        :type message: {None | str}
        :param chart_result: whether or not to include a 
            barchart of the result
        :type chart_result: bool
        :param print_results: whether or not to print the
            recording durations of each species to the console
        :type print_results: bool
        :param inventory: optionally, pre-computed inventory
        :type inventory: {None | pd.DataFrame}
        :param num_workers: max number of CPUs to use;
            Default: Utils.MAX_PERC_OF_CORES_TO_USE
        :type num_workers: {None | int}
        :raise FileNotFoundError if no audio files found.
        '''

        self.log = LoggingService()
        
        if not os.path.exists(species_root) or not os.path.isdir(species_root):
            self.log.err(f"Directory must exist, but given {species_root}")
            sys.exit(1)

        # If inventory was not provided, 
        # create it now:
        if inventory is None:
            self.log.info("Begin recording inventory...")
            start_time = datetime.datetime.now()
            # Get df:
            #                   total_recording_length       ...
            #    species1              10.4                  ...
            #    species2            ...
            
            inventory = SoundProcessor.recording_lengths_by_species(species_root, num_workers=num_workers)
            
            # Could have come out to be None
            if inventory is None:
                raise FileNotFoundError()
            
            end_time = datetime.datetime.now()
            duration_str = Utils.time_delta_str(end_time - start_time)
            self.log.info(f"Done with recording inventory ({duration_str}).")
        
        if print_results:
            print(inventory.to_string())

        manifest_dir_path = self.write_inventory(inventory, species_root, message)

        if chart_result:
            # Make bar chart of recording lengths, and
            # save it in the manifest directory:
            fig_title = f"Recording Durations in {Path(species_root).stem}"
            self.chart_results(inventory, manifest_dir_path, fig_title)

        if print_results:
            print(f"Outputs were saved in {str(manifest_dir_path)}")
            
        # Make some quantities available to the outside:
        self.manifest_dir_path = str(manifest_dir_path)
        self.inventory = inventory
        

    #------------------------------------
    # write_inventory
    #-------------------

    def write_inventory(self, inventory, species_root, message):
        '''
        Given a dataframe (inventory):

                         total_recording_length (secs)   duration (hrs:mins:secs)
            species1            10.5                        0:10:30
            species2             2.0                        0:02:00
               ...              ...
               
        of which only the 'total_recording_length (secs)' is required.
        Creates a new directory sibling to species_root:
        
                ../Audio_Manifest_<species_root_dir_name>
                
        Places into that new directory:
        
               o README.txt     : containing the content of the message argument
               o manifest.json  : the inventory saved as JSON

        :param inventory: sum of recording lengths of each species
        :type inventory: pd.DataFrame
        :param species_root: directory below which the species
            subdirectories reside
        :type species_root: str
        :param message: descriptive text for README.txt
        :type message: str
        :return: name of manifest directory
        :rtype: str
        '''

        species_root_path = Path(species_root)
        # Ensure existence of destination for manifest directory:
        manifest_dir_path = species_root_path.parent.joinpath(f"Audio_Manifest_{species_root_path.stem}")
        if not manifest_dir_path.exists():
            manifest_dir_path.mkdir()
            
        # Generate README.txt if requested:
        if message is not None:
            with open(manifest_dir_path.joinpath('README.txt'), 'w') as fd:
                fd.write(message)
        
        # Write inventory as manifest.json:
        manifest_fname = manifest_dir_path.joinpath('manifest.json')
        with open(manifest_fname, 'w') as fd:
            inventory.to_json(fd)

        return manifest_dir_path
        
    #------------------------------------
    # chart_results
    #-------------------
    
    def chart_results(self, df, manifest_dir_path, fig_title):

        self.log.info(f"Preparing bar chart...")
        fig = plt.figure()
        # Make sure the species names
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

    try:
        RecordingsInventory(args.species_root,
                            message=args.message,
                            chart_result=args.chart)
    except FileNotFoundError:
        print(f"No audio files found under {args.species_root}")
    
    if args.chart:
        input("Hit any key to close figure and exit...")
    