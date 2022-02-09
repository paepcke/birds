'''
Created on Feb 8, 2022

@author: paepcke
'''

import argparse
import os
import sys

import texttable

from powerflock.signatures import TemplateCollection, SpectralTemplate


class TemplatePrinter:
    '''
    Prints information about one or more templates
    to the console. Purely for debugging and status
    checking.
    
    Input: file path to an individual json encoded template,
    or a json encoded TemplateCollection. Typical examples for 
    latter: templates_calibrated.json or templats.json. For former:
    template_BANAS.json. 
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, path):
        '''
        Constructor
        '''
        # Try loading as TemplateCollection:
        try:
            templ_coll = TemplateCollection.json_load(path)
        except ValueError:
            # Read as individual template:
            templ = SpectralTemplate.json_load(path)
            species = templ.signatures[0].species
            templ_coll = {species : templ}
        for species, templ in templ_coll.items():
            self.print_template(species, templ)
            
    #------------------------------------
    # print_template
    #-------------------
    
    def print_template(self, species, templ):
        
        bandpass_low  = f"{int(templ.bandpass_filter['low_val'])}kHz"
        bandpass_high = f"{int(templ.bandpass_filter['high_val'])}kHz"
        print(species)
        
        print(f"\tBandpass:\t{bandpass_low}-{bandpass_high}")
        print(f"\tNum sigs:\t{len(templ.signatures)}")
        print(f"\tRecording:\t{templ.recording_fname}")
        
        tab = texttable.Texttable()
        headings = ['sig_id', 'usable','duration']
        tab.header(headings)
        ids       = [sig.sig_id for sig in templ.signatures]
        usability = []
        durations = []
        for sig in templ.signatures:
            try:
                usable = sig.usable
            except AttributeError:
                usable = 'n/a'
            usability.append(usable)
            durations.append(f"{round(sig.duration(), 1)}sec")
        
        for row in zip(ids, usability, durations):
            tab.add_row(row)
        
        s = tab.draw()
        # Print table indented
        for line in s.split('\n'): print(f"\t{line}")

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Print table of template info, given path to template, or template collection."
                                     )

    parser.add_argument('template_path',
                        help='fully qualified path to .json template or template collection file',
                        default=None)

    args = parser.parse_args()

    if not os.path.exists(args.template_path):
        print(f"Cannot find args.template_path")
        sys.exit(1)
    TemplatePrinter(args.template_path)
    
    #TemplatePrinter('/Users/paepcke/EclipseWorkspacesNew/birds/experiments/PowerSignatures/QuadCalDev/json_files/template_BANAS.json')
    #TemplatePrinter('/Users/paepcke/EclipseWorkspacesNew/birds/experiments/PowerSignatures/QuadCalDev/json_files/templates_calibrated.json')
    