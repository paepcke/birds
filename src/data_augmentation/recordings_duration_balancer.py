'''
Created on Sep 1, 2021

@author: paepcke
'''

import os
import sys
import argparse
from pathlib import Path

import pandas as pd

from data_augmentation.sound_processor import SoundProcessor

class DurationsBalancer:
    '''
    classdocs
    '''


    def __init__(self, 
                 species_recordings_dir,
                 durations_goal,
                 excess_dest_dir=None,
                 ):
        '''
        Constructor
        '''
        
        species = Path(species_recordings_dir).stem
        # Get a df with fnames as index, and 'recording_length_secs'
        # as the only column: 
        recording_durations = SoundProcessor.find_recording_lengths(species_recordings_dir)
        
        
