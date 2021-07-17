#!/usr/bin/env python3

'''
Created on Jul 16, 2021

@author: paepcke
'''
import argparse
import csv
import os
import re
import sys

from birdsong.utils.utilities import FileUtils
import matplotlib.pyplot as plt

class SelectionTableConsolidator:
    '''
    classdocs
    '''

    EXTRA_FLD_NM_PAT = re.compile(r"dict contains fields not in fieldnames: '([^']*)'.*$")
    '''Error msg for extraneous col in selection table'''
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self):
        '''
        Constructor
        '''
        pass

    #------------------------------------
    # consolidate_tables 
    #-------------------
    
    def consolidate_tables(self, tbl_paths, out_tbl_name):
        '''
        Goes through each table, cleans it, and
        writes one tab separated value table to 
        self.out_tbl_name.
        '''
        self.sel_tbl_flds = ['RecordingID',
                             'Begin Time (s)',
                             'End Time (s)',
                             'Delta Time (s)',
                             'Low Freq (Hz)',
                             'High Freq (Hz)',
                             'Species',
                             'Type', # Song/Call/Song-Trill, etc.
                             'Number',
                             'Mix'
                             ]

        with open(out_tbl_name, 'w') as fd:
            writer = csv.DictWriter(fd, fieldnames=self.sel_tbl_flds)
            writer.writeheader()
            for tbl_path in tbl_paths:
                self._add_entries(tbl_path, writer)

    #------------------------------------
    # species_histogram
    #-------------------
    
    def species_histogram(self, sel_table_path, plot=False):
        
        # Dict {<species> : num-of-observations}
        histogram = {}
        with open(sel_table_path, 'r') as fd:
            reader = csv.DictReader(fd, delimiter=',')
            for row in reader:
                species = row['Species']
                try:
                    histogram[species] += 1
                except KeyError:
                    # First time:
                    histogram[species] = 1
                mix = row['Mix']
                if len(mix) > 0:
                    # Mix is comma separated list of species:
                    species_list = mix.split(',')
                    for species in species_list:
                        try:
                            histogram[species] += 1
                        except KeyError:
                            # First time:
                            histogram[species] = 1
        if plot:
            self.plot_histogram(histogram)
        return histogram

    #------------------------------------
    # plot_histogram
    #-------------------
    
    def plot_histogram(self, histogram):
    
        species      = list(histogram.keys())
        observations = list(histogram.values())
    
        #****_fig, ax = plt.subplots(figsize =(16, 9))
        _fig, ax = plt.subplots(figsize =(16, 16))
        ax.barh(species, observations)

        # Remove axes lines and frame around fig:
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

        # More space between species names
        # and left end of bars:
        ax.xaxis.set_tick_params(pad = 5)
        ax.yaxis.set_tick_params(pad = 10)
        
        # Draw grid:
        ax.grid(b = True, color ='grey',
                linestyle ='-.', linewidth = 0.5,
                alpha = 0.2)
                
        # Label the bars
        for i in ax.patches:
            plt.text(i.get_width()+0.2, i.get_y()+0.5,
                     str(round((i.get_width()), 2)),
                     fontsize = 10, fontweight ='bold',
                     color ='grey')
        
        ax.set_title('Bird Species Observations',
                     loc ='left', )

        _fig.show()
        print('foo')

    #------------------------------------
    # _add_entries
    #-------------------
    
    def _add_entries(self, tbl_path, writer):
        
        recording_id = FileUtils.extract_recording_id(tbl_path)
        
        with open(tbl_path, 'r') as fd:
            #****reader = csv.DictReader(fd, fieldnames=self.sel_tbl_flds)
            reader = csv.DictReader(fd, delimiter='\t')
            header_row = reader.fieldnames
            #header = fd.readline().strip()
            for row in reader:
                # Remove dict entries we don't need:
                del row['Selection'] 
                del row['View']
                del row['Channel']

                # Lots of variation in the table headers
                # for species, type, number, and mix:
                
                try:
                    # The col name for species seems to be
                    # a favorite for inconsistency:
                    row['Species'] = self._try_capitalizations(row, 'SPECIE', recording_id)
                except ValueError:
                    if 'species' in header_row:
                        row['Species'] = row['species']
                        del row['species']
                    elif 'SPECIES' in header_row:
                        row['Species'] = row['SPECIES']
                        del row['SPECIES']
                    elif 'Especie' in header_row:
                        row['Species'] = row['Especie']
                        del row['Especie']
                    
                    row['Species'] = self._try_capitalizations(row, 'SPECIES', recording_id)
                    
                row['Type']   = self._try_capitalizations(row, 'TYPE', recording_id)
                row['Number'] = self._try_capitalizations(row, 'NUMBER', recording_id)
                row['Mix']    = self._try_capitalizations(row, 'MIX', recording_id)

                # Compute time delta:
                row['Delta Time (s)'] = float(row['End Time (s)']) - float(row['Begin Time (s)'])
                # Add recoring_id to each row:
                row['RecordingID'] = recording_id
                
                # Ensure that all species names are
                # upper case:
                row['Species'] = row['Species'].upper()
                if len(row['Mix']) > 0:
                    mix_species = row['Mix'].split(',')
                    clean_mix_species = [the_species.strip().upper()
                                         for the_species 
                                         in mix_species
                                         if len(the_species.strip()) > 0
                                         ]
                    row['Mix'] = ','.join(clean_mix_species)
                
                try:
                    writer.writerow(row)
                except ValueError as e:
                    # Sometimes extraneous cols are added leading to:
                    #   "dict contains fields not in fieldnames: 'Foobar'"
                    match = self.EXTRA_FLD_NM_PAT.search(e.args[0])
                    if match is not None:
                        bad_col = match.groups()[0]
                        print(f"Warning: extraneous column in tbl {recording_id}: '{bad_col}'")

    #------------------------------------
    # _try_capitalizations
    #-------------------

    def _try_capitalizations(self, row, key, recorder_id):
        val = None
        try:
            val = row[key.upper()]
            del row[key.upper()]
        except KeyError:
            pass
        try:
            val = row[key.lower()]
            del row[key.lower()]
        except KeyError:
            pass
        
        try:
            val = row[self._capitalize(key)]
            del row[self._capitalize(key)]
        except KeyError:
            pass
        
        if val is not None:
            return val
        
        raise ValueError(f"Key {key} is not in table for {recorder_id}")
        
    #------------------------------------
    # _capitalize 
    #-------------------
    
    def _capitalize(self, the_str):
        
        all_lower = the_str.lower()
        return all_lower[0].upper() + all_lower[1:]

# ------------------------ Main ------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Clean and consolidate selection tables in a directory"
                                     )

    parser.add_argument('-i', '--histogram',
                        help='print histogram of observed species'
                        )

    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help='plot histogram of observed species in bar chart; default: Not'
                        )
    
    parser.add_argument('-d', '--out_name',
                        help='name of outfile absolute, or relative to sel_tbl_all; default: sel_tbl_all_xxx.csv'
                        )
    
    parser.add_argument('-t', '--table',
                        type=str,
                        nargs='+',
                        help='repeatable: names of tables to include, or finished sel tbl for histogram; default: all .txt files in sel_tbl_dir')

    
    parser.add_argument('--sel_tbl_dir',
                        help='path to sel tbl files, if tbl consolidation requested; incompatible with --histogram'
                        )

    args = parser.parse_args()

    if args.histogram:
        
        # ------------ Do Histogram ------------
                
        if args.sel_tbl_dir is not None:
            print("Can either request consolidating sel tbls by providing sel_tbl_dir, OR histogram")
            exit(1)
        
        histogram = SelectionTableConsolidator().species_histogram(args.histogram,
                                                                   plot=args.plot
                                                                   )
        if args.plot:
            input("Hit any key to exit and close the figure: ")
        else:
            print(histogram)
        sys.exit(0)

    # ------------ Do selection table consolidation ------------

    in_dir = args.sel_tbl_dir
    
    if not os.path.exists(in_dir):
        print(f"Cannot find {in_dir}")
        sys.exit(1)
        
    if len(os.listdir(in_dir)) == 0:
        print(f"Directory {in_dir} is empty; doing nothing")
        sys.exit(1)
        
    tables = args.table
    tables_full_paths = []
    if tables is not None:
        # Ensure all tbls exist:
        for tbl_nm in tables:
            tables_full_paths.append(os.path.join(in_dir, tbl_nm))
            if not os.path.exists(tables_full_paths[-1]):
                print(f"Table {tables_full_paths[-1]} does not exist")
    else:
        for tbl_nm in os.listdir(in_dir):
            if not tbl_nm.endswith('.txt'):
                continue 
            tables_full_paths.append(os.path.join(in_dir, tbl_nm))

    out_name = args.out_name
    if out_name is None:
        out_path_root = os.path.join(in_dir, 'sel_tbl_all.txt')
        out_path = out_path_root
        distinguisher = 0
        while os.path.exists(out_path):
            distinguisher += 1
            out_path += f"_{distinguisher}"
    else:
        if not os.path.isabs(out_name):
            out_name = os.path.join(in_dir, out_name)
        # If given name exists: warn:
        if os.path.exists(out_name):
            confirmation = input(f"Table {out_name} exists; overwrite? (y/n): ")
            if confirmation != 'y':
                print("Cancelling.")
                sys.exit(0)
                
    consolidator = SelectionTableConsolidator()
    consolidator.consolidate_tables(tables_full_paths, out_name)
    
    print(f"Consolidated table in {out_name}")