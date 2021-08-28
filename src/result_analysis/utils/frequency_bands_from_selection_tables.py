'''
Created on Aug 25, 2021

@author: paepcke
'''
import csv
import os
import tempfile

from data_augmentation.utils import Utils
import numpy as np
from result_analysis.utils.consolidate_selection_tables import SelectionTableConsolidator


class FrequencyBandExtractor(dict):
    '''
    Given a selection table, collect the
    frequency bands of each species, and 
    create a result dict. The dict maps
    a species name to a FrequencyInfo instance.
    Those instances contain min/max frequencies,
    mean center frequency, and standard deviation
    of that mean.
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, sel_tbl_info):
        '''
        The sel_tbl_info can be:
           o the path to a selection table file
           o the path to a directory of selection tables
           
        The tables are combined into one consolidated
        table, and observations are made from the 
        "Low Freq (Hz)" and "High Freq (Hz)" columns.
        
        :param sel_tbl_info: path to selection table or directory
        :type sel_tbl_info: src
        '''
        
        if not os.path.exists(sel_tbl_info):
            raise FileNotFoundError(f"No selection table at {sel_tbl_info}")
        
        if os.path.isdir(sel_tbl_info):
            self.sel_dict_list = self._consolidate_tbls(sel_tbl_info)
        else:
            # Just a single select table:
            self.sel_dict_list = Utils.read_raven_selection_table(sel_tbl_info)

        self.freq_info = self._create_freq_infos(self.sel_dict_list)

    #------------------------------------
    # _consolidate_tbls
    #-------------------
    
    def _consolidate_tbls(self, tbl_dir):
        '''
        Given a directory path to Raven selection tables,
        consolidate all tables into one big table. Read
        the rows (i.e. each selection) into a dict, and
        return a list of those dicts.
        
        :param tbl_dir: path to directory with selection tables
        :type tbl_dir: src
        :return list of row dicts, each dict containing 
            one selection's data
        :rtype [{str : str}]
        '''
        
        tbl_paths = []
        for root, _dirs, files in os.walk(tbl_dir):
            for fname in files:
                tbl_paths.append(os.path.join(root, fname))
    
        # Consolidate into a single table:
        all_rows_dict_list = []
        for tbl_path in tbl_paths:
            all_rows_dict_list.extend(Utils.read_raven_selection_table(tbl_path))
                
        return all_rows_dict_list
    
    #------------------------------------
    # _create_freq_infos
    #-------------------
    
    def _create_freq_infos(self, selection_dicts):
        '''
        Given a list of dicts, each containing selection
        table information about a single selection (species,
        start time, frequency range, etc.), return a 
        new dict that maps species names to a FrequencyInfo
        instance. Those instances contain information about
        frequency ranges of the respective species.
        
        :param selection_dicts: list of dicts, each dict
            reflecting one row in a selection table
        :type selection_dicts: [{str : str}]]
        :return dict mapping a species name to a FrequencyInfo
            instance
        :rtype {str : FrequencyInfo}
        '''
        
        for sel_dict in selection_dicts:
            low_freq  = float(sel_dict['Low Freq (Hz)'])
            high_freq = float(sel_dict['High Freq (Hz)'])
            species   = sel_dict['species']
            
            try:
                freq_info = self[species]
            except KeyError:
                # First observation of this species:
                freq_info = FrequencyInfo(species)
                self[species] = freq_info
                
            # Add the new observation:
            freq_info.add_selection(low_freq, high_freq)

# -------------------- Class FrequencyInfo ------------

class FrequencyInfo:
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, species):
        
        self.species = species
        self.all_low_freqs  = np.array([])
        self.all_high_freqs = np.array([])
        self._curr_lowest_freq  = float('inf')
        self._curr_highest_freq = 0.0

    #------------------------------------
    # min_frequency
    #-------------------

    def min_frequency(self):
        return self._curr_lowest_freq

    #------------------------------------
    # max_frequency
    #-------------------

    def max_frequency(self):
        return self._curr_highest_freq
    
    #------------------------------------
    # center_frequency
    #-------------------
    
    def center_frequency(self):
        '''
        Return the mean of all observations' 
        center frequencies.
        '''

        return np.mean((self.all_high_freqs - self.all_low_freqs) / 2.)

    #------------------------------------
    # stdev_freq_bands
    #-------------------
    
    def stdev_freq_bands(self):
        '''
        Return the standard deviation of the
        frequencies around the mean of the species'
        center frequency.
        
        :return standard deviation spread around
            mean of center frequencies in Hz
        :rtype float
        '''
        # The delta degrees of freedom
        # for this call is 0, meaning we are
        # using degrees of freedom equal to 
        # all the frequency centers. The method
        # computes the population stdev:
        return np.std((self.all_high_freqs - self.all_low_freqs) / 2.)

    #------------------------------------
    # add_selection
    #-------------------
    
    def add_selection(self, low_freq, high_freq):
        '''
        Add another observation of this FrequencyInfo
        instance's species. Update lowest/highest observed.
        
        Checks that low_freq < high_freq; if not, switches
        them. 

        :param low_freq: the low frequency of the observation
        :type low_freq: float
        :param high_freq: hight frequency of the observation
        :type high_freq: float
        '''

        # Switch low and high frequencies
        # if they are reversed:
        
        if low_freq < high_freq:
            real_low_freq  = low_freq
            real_high_freq = high_freq
        else:
            real_low_freq  = high_freq
            real_high_freq = low_freq


        # Update the species' lowest and
        # highest observed frequency:
        if real_low_freq < self._curr_lowest_freq:
            self._curr_lowest_freq = real_low_freq
        if real_high_freq > self._curr_highest_freq:
            self._curr_highest_freq = real_high_freq

        self.all_low_freqs  = np.append(self.all_low_freqs, real_low_freq)
        self.all_high_freqs = np.append(self.all_high_freqs, real_high_freq)
