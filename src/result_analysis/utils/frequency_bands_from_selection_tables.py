'''
Created on Aug 25, 2021

@author: paepcke
'''
import csv
import os
import tempfile

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
        
        if not os.exists(sel_tbl_info):
            raise FileNotFoundError(f"No selection table at {sel_tbl_info}")
        
        tbls = []
        if os.path.isdir(sel_tbl_info):
            sel_dict_list = self._consolidate_tbls(sel_tbl_info)
        else:
            # Just a single select table:
            with open(sel_tbl_info, 'r') as fd:
                reader = csv.DictReader(fd)
                sel_dict_list = [sel_dict
                                 for sel_dict
                                 in reader 
                                 ]

        self.freq_infos = self._create_freq_infos(sel_dict_list)

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
        
        tbls = []
        for root, _dirs, files in os.walk(tbl_dir):
            for fname in files:
                tbls.append(os.path.join(root, fname))
    
        # Consolidate into a single table:
        consolidated_rows_dict_list = []
        consolidator = SelectionTableConsolidator() 
        with tempfile.NamedTemporaryFile(suffix='.txt', 
                                         prefix='sel_tbl_tmp_', 
                                         dir='/tmp', 
                                         delete=True) as fd:
            out_path = fd.name
            consolidator.consolidate_tables(tbls, out_path)
            
            fd.flush()
            reader = csv.DictReader(fd)
            for selection_dict in reader:
                consolidated_rows_dict_list.append(selection_dict)
                
        return consolidated_rows_dict_list
    
    #------------------------------------
    # _create_freq_infos
    #-------------------
    
    def _create_freq_infos(self, selection_dicts):
        '''
        Given a list of dicts, each containing selection
        table information about a single select (species,
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
            species   = sel_dict['Species']
            
            try:
                freq_info = self[species]
            except KeyError:
                # First observation of this species:
                freq_info = FrequencyInfo()
                self[species] = freq_info
                
            # Add the new observation:
            freq_info.add_selection(low_freq, high_freq)

# -------------------- Class FrequencyInfo ------------

class FrequencyInfo:
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self):
        
        self.all_low_freqs  = np.array([])
        self.all_high_freqs = np.array([])

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
        if real_low_freq < self.curr_lowest_freq:
            self.curr_lowest_freq = real_low_freq
        if real_high_freq > self.curr_high_freq:
            self.curr_highest_freq = real_high_freq

        self.all_low_freqs  = np.append(self.all_low_freqs, real_low_freq)
        self.all_high_freqs = np.append(self.all_high_freqs, real_high_freq)

    #------------------------------------
    # center_frequency
    #-------------------
    
    def center_frequency(self):
        '''
        Return the mean of all observations' 
        center frequencies.
        '''

        return np.mean(self.all_high_freqs - self.all_low_freqs) 

    #------------------------------------
    # stdev
    #-------------------
    
    def stdev(self):
        '''
        Return the standard deviation of the
        frequencies
        '''
        
        
