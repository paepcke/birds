#!/usr/bin/env python3
'''
Created on Sep 1, 2021

@author: paepcke
'''

import argparse
import os
import shutil
import sys

from logging_service import LoggingService

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils


class DurationsBalancer:
    '''
    Given the directory containing recordings
    of one species, accomplishes the following:
        
        o If the sum of recording durations is less than
          or equal to a given duration_goal in sec, do nothing
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
                 duration_goal,
                 excess_dest_dir=None,
                 dry_run=False
                 ):
        '''
        Get recording duration for each recording, and other
        needed computations. No action is taken as part of
        this init. The work is done when client calls the
        balance() method.
        
        :param species_recordings_dir: root directory of
            recordings for the species to process
        :type species_recordings_dir: str
        :param duration_goal: number of recording seconds
            to remove
        :type duration_goal: {int | float}
        :param excess_dest_dir: optional directory where
            'removed' recordings are moved. If None, excess
            recordings are removed from the file system
        :type excess_dest_dir: {None | str}
        :param dry_run: no removal is done, instead,
            a log of would-have actions is created in 
            self.dry_run
        :rtype dry_run: bool
        '''

        self.log = LoggingService()

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
        self.duration_goal = duration_goal
        
        # Get a df with fnames as index, and columns 
        # 'recording_length_secs' and 'recording_length_hhs_mins_secs'

        self.log.info("Begin recording duration inventory...")
        recording_durations = SoundProcessor.find_recording_lengths(species_recordings_dir)
        
        self.total_duration = float(recording_durations.loc[:,:'recording_length_secs'].sum())
        human_readable_dur  = Utils.time_delta_str(self.total_duration)
        self.log.info(f"Done recording duration inventory; total duration: {human_readable_dur}")

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
            o duration_goal: number of recording seconds to remove
            o dur_df: recording duration of each file
        '''
        num_files_processed = 0
        durations_removed = 0
        
        new_dur = self.total_duration
        for recording_fname, row in self.dur_df.iterrows():
            recording_dur = row.loc['recording_length_secs']
            maybe_new_dur = new_dur - recording_dur
            if maybe_new_dur >= self.duration_goal:
                self._remove_recording(recording_fname)
                num_files_processed += 1
                durations_removed += recording_dur
                new_dur = maybe_new_dur
        
        human_readable_time = Utils.time_delta_str(durations_removed)
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

# ------------------------ Main ------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description=("Move or delete enough recordings to \n"
                                                  "reach a given recording sum duration goal"
                                                  )
                                     )

    parser.add_argument('-d', '--dryRun',
                        help='show what script would do if run normally; no culling is performed.',
                        action='store_true')
    
    parser.add_argument('-s', '--destination',
                        help=('path to directory where excess files are to move;\n'
                              'if empty string: delete files; default: /tmp/saved_recordings'
                              ),
                        default='/tmp/saved_recordings')
    
    parser.add_argument('recording_root',
                        help='path to dir that contains recordings to cull')

    parser.add_argument('goal_duration',
                        type=int,
                        help='goal of total recording duration')

    args = parser.parse_args()
    
    if not os.path.exists(args.recording_root):
        print(f"Did not find {args.recording_root}")
        sys.exit(1)
        
    # Replace an empty string file move destination
    # with None:
    if len(args.destination) == 0:
        args.destination = None
    
    balancer = DurationsBalancer(
        args.recording_root,
        args.goal_duration,
        excess_dest_dir=args.destination,
        dry_run=args.dryRun
        )

    balancer.balance()
    if args.dryRun:
        print(f"Dry run would-be actions; goal: {args.goal_duration}")
        print(balancer.dry_run_log)
    else:
        if args.destination is not None:
            print(f"Any excess files were moved to {args.destination}")
