#!/usr/bin/env python3
'''
Created on Sep 1, 2021

@author: paepcke
'''

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from logging_service import LoggingService

from birdsong.utils.utilities import FileUtils
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils
import pandas as pd


class DurationsBalancer:
    '''
    Given the directory containing recordings
    of one species, accomplishes the following:
        
        o If the sum of recording durations is less than
          or equal to a given duration_goal in seconds, do nothing
        o If the sum is greater than the goal, recordings
          are removed from the directory until the sum drops
          to the specified duration.
          
    The process proceeds such that as few recordings as
    possible are removed to preserve variety.
    
    Clients specify whether removal means deletion, or 
    moving discarded file to a different directory.
    
    Usage: create instance, then call inst.balance()
    '''

    def __init__(self, 
                 species_recordings_dir,
                 duration_goal_seconds,
                 excess_dest_dir=None,
                 inventory=None,
                 dry_run=False
                 ):
        '''
        Get recording duration for each recording, and other
        needed computations. No action is taken as part of
        this init. The work is done when client calls the
        balance() method.
        
        The species_recordings_dir is the root of recordings
        for a single species. The dir name is assumed to 
        match the species code. 
        
        If the excess_dest_dir is None, excess files are
        deleted. Else the arg should point to a directory
        under which the excess files will be moved.
        
        If inventory is provided, the step of tallying all
        recording seconds for the species is skipped, and
        the file <species>_manifest.json.gz in the inventory
        directory is imported and turned into a dataframe
        instead. That info looks like:
        
		  recording_length_secs recording_length_hhs_mins_secs
		  wtros_bird2.mp3                  14.55                 0:00:14.550000
		  wtros_bird3.mp3                  45.53                 0:00:45.530000
		  wtros_bird1.mp3                  29.88                 0:00:29.880000
		          
        An inventory might have been created explicitly via a call
        to the recording_length_inventory.py script from a 
        terminal window. By convention manifest files are
        in directories starting with Audio_Manifest_..., or are
        included in an experiment manager.
        
        :param species_recordings_dir: root directory of
            recordings for the species to process
        :type species_recordings_dir: str
        :param duration_goal_seconds: number of recording seconds
            to remove
        :type duration_goal_seconds: {int | float}
        :param excess_dest_dir: optional directory where
            'removed' recordings are moved. If None, excess
            recordings are removed from the file system
        :type excess_dest_dir: {None | str}
        :param inventory: directory containing pre-existing
            durations.
        :type inventory: {None | str}
        :param dry_run: no removal is done, instead,
            a log of would-have actions is created in 
            self.dry_run
        :rtype dry_run: bool
        '''

        self.log = LoggingService()

        self.species = Path(species_recordings_dir).stem
        if dry_run:
            self.dry_run_log = []
        else:
            # If excess to be moved to safe location:
            if excess_dest_dir is not None:
                os.makedirs(excess_dest_dir, exist_ok=True)
                self.log.info(f"Any excess recordings will be moved to {excess_dest_dir}")
            else:
                response = input("Any excess files will be *deleted*, not moved (Y/n): ")
                if response != 'Y':
                    print("Aborting before anything was done.")

        self.dry_run = dry_run
        
        self.species_recordings_dir = species_recordings_dir
        self.excess_dest_dir = excess_dest_dir
        self.duration_goal_secs = duration_goal_seconds
        
        # Get a df with fnames as index, and columns 
        # 'recording_length_secs' and 'recording_length_hhs_mins_secs'

        if inventory is None:
            # Determine recording duration of each audio 
            # file under species_recordings_dir, and place
            # those durations into a df. Then save that df 
            # as compressed json: 
            
            # Get default manifest directory as destination
            # for the saved recordings df:
            inventory_dir = FileUtils.make_manifest_dir_name(Path(species_recordings_dir).parent)
            if not os.path.exists(inventory_dir):
                os.makedirs(inventory_dir)
            recording_durations = self._analyze_species_recordings(species_recordings_dir, 
                                                                   inventory_dir)
        else:
            # We were given a previously created inventory dir.
            # Find the manifest file for this species:
            durations_path = os.path.join(inventory, f"{self.species}_manifest.json.gz")
            if not os.path.exists(durations_path):
                # Create the info the long way after all:
                recording_durations = self._analyze_species_recordings(species_recordings_dir, 
                                                                      inventory_dir)
            else:
                # Read the pre-computed gzipped json 
                # file of recording durations into a 
                # json str, turn to dict, and frome there
                # to df:
                inventory_dict = json.loads(SoundProcessor.read_gzipped_file(durations_path)) 
                recording_durations = pd.DataFrame.from_dict(inventory_dict)
        
        self.total_duration = float(recording_durations.loc[:,:'recording_length_secs'].sum())

        # Sort by decreasing recording duration:
        self.dur_df = recording_durations.sort_values(by='recording_length_secs', 
                                                      axis='index',
                                                      ascending=False
                                                      )
    #------------------------------------
    # balance
    #-------------------

    def balance(self):
        '''
        Plan and execute the balancing. The inst var self.dur_df must
        be of the form:

                     recording_length_secs recording_length_hhs_mins_secs
            fname1           12                  0:00:12
            fname2           62                  0:01:02 

        though only the recording_length_secs is used
        and needs to be present.
        
        Assumptions: 
            o self.species_recordings_dir contains the
                directory that contains the recordings referenced
                in dur_df.
            o duration_goal_secs: number of recording seconds 
                to left as a maximum
            o dur_df: recording duration of each file
        '''
        if self.total_duration <= self.duration_goal_secs:
            # Already at goal:
            return
        
        num_files_processed = 0
        durations_removed = 0
        
        new_dur = self.total_duration
        for recording_fname, row in self.dur_df.iterrows():
            self._remove_recording(recording_fname)
            num_files_processed += 1
            recording_dur = row.loc['recording_length_secs']
            durations_removed += recording_dur
            new_dur -= recording_dur
            if new_dur <= self.duration_goal_secs:
                # Done:
                break
        
        human_readable_time = Utils.time_delta_str(durations_removed)
        if self.dry_run:
            self.log.info(f"Would have (re)moved {num_files_processed} recordings ({human_readable_time})")
        else:
            self.log.info(f"(Re)moved {num_files_processed} recordings ({human_readable_time})")

    #------------------------------------
    # _remove_recording
    #-------------------

    def _remove_recording(self, fname):
        '''
        Discards one file, either by deletion or moving,
        depending on self.excess_dest_dir. If that is None,
        delete, if it is a directory, move.
        
        If self.dry_run is True, enter log entries like:
        
            delete,<path>
            move,<path> <dest-dir>

        instead of taking action.
        
        :param fname: name of file in self.species_recordings_dir
            to process
        :type fname: str
        '''
        
        path = os.path.join(self.species_recordings_dir, fname)
        if self.excess_dest_dir is None:
            if self.dry_run:
                self.dry_run_log.append(f"delete,{path}")
                return
            os.remove(path)
        else:
            if self.dry_run:
                self.dry_run_log.append(f"mv,{path} {self.excess_dest_dir}")
                return
            shutil.move(path, self.excess_dest_dir)

    #------------------------------------
    # _analyze_species_recordings(species_recordings_dir, inventory_dir)
    #-------------------

    def _analyze_species_recordings(self, species_recordings_dir, inventory_dir):

        self.log.info("Begin recording duration inventory...")
        recording_durations = SoundProcessor.find_recording_lengths(species_recordings_dir)
        
        # File name where to put the gzipped json file with 
        # recording lengths info for this species.
        # The gzip_file call below will turn the
        # file name into .gz:
        durations_path = os.path.join(inventory_dir, f"{self.species}_manifest.json")
        # Save the df:
        recording_durations.to_json(durations_path)
        # ... and gzip it, removing the unzipped version:
        SoundProcessor.gzip_file(durations_path, delete=True)
        self.log.info(f"Done recording duration inventory")
        
        return recording_durations

# ------------------------ Main ------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description=("Move or delete enough recordings to \n"
                                                  "reach a given recording sum duration goal \n"
                                                  "within a given species"
                                                  )
                                     )

    parser.add_argument('-d', '--dryRun',
                        help='show what script would do if run normally; no culling is performed.',
                        action='store_true')

    parser.add_argument('-i', '--inventory',
                        help=('path to .json file of recording seconds available \n'
                              'for each species (by convention: manifest.json)'),
                        default=None)

    parser.add_argument('-s', '--destination',
                        help=('path to directory where excess files are to move;\n'
                              'if empty string: delete files; default: /tmp/saved_recordings'
                              ),
                        default='/tmp/saved_recordings')
    
    parser.add_argument('recording_root',
                        help='path to dir that contains recordings to cull')

    parser.add_argument('goal_duration',
                        type=int,
                        help='goal for maximum total recording duration in seconds for each species')

    args = parser.parse_args()
    
    if not os.path.exists(args.recording_root):
        print(f"Did not find {args.recording_root}")
        sys.exit(1)

    if args.inventory is not None and not os.path.exists(args.inventory):
        print(f"Did not find inventory file {args.inventory}")
        sys.exit(1)
        
    # Replace an explicitly empty string file move destination
    # with None; as stated in the add_argument call above,
    # the default if this option is skipped on command line is to 
    # save to /tmp/saved_recordings:
    if len(args.destination) == 0:
        args.destination = None

    # Go through each species subdir,
    # and cull it:
    for species_dir in Utils.listdir_abs(args.recording_root):
        if not os.path.isdir(species_dir):
            continue
        print(f"Culling {Path(species_dir).stem}")
        balancer = DurationsBalancer(
            species_dir,
            args.goal_duration,
            excess_dest_dir=args.destination,
            dry_run=args.dryRun,
            inventory=args.inventory
            )
    
        balancer.balance()
        if args.dryRun:
            print(f"Dry run would-be actions; goal: {args.goal_duration}")
            print(balancer.dry_run_log)
        else:
            if args.destination is not None:
                print(f"Any excess files were moved to {args.destination}")
