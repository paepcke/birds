'''
Created on Aug 5, 2021

@author: paepcke
'''
import csv
import json 
import os
from pathlib import Path
import shutil

import torch

from data_augmentation.utils import Utils
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn


class Experiment(dict):
    '''
    Container to hold all information about an experiment:
    the pytorch model parameters, location of model snapshot,
    location of csv files created during training and inference.
    
    An experiment instance is saved and loaded via
        o <exp-instance>.save(fp), and 
        o Experiment.load(fp)
        
    Storage format is json
    
    Methods:
        o mv                  Move all files to a new root
        o save                Write a pytorch model, csv file, or figure
        
        o add_csv            Create a CSV file writer
        o close_csv           Close a CSV file writer
        
        o close               Close all files
    
    Keys:
        o root_path
        o model_path
        o logits_path
        o probs_path
        o ir_results_path
        o tensor_board_path
        o perf_per_class_path
        o conf_matrix_img_path
        o pr_curve_img_path
    
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, root_path, initial_info=None):
        '''

        :param initial_info: optionally, a dict with already
            known facts about the experiment.
        :type initial_info:
        '''

        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if not os.path.isdir(root_path):
            raise ValueError(f"Root path arg must be a dir, not {root_path}")

        if initial_info is not None:
            # Must be a dict:
            if not type(initial_info) == dict:
                raise TypeError(f"Arg initial_info must be a dict, not {initial_info}")
            init_info_keys = list(initial_info.keys())
            # Add passed-in info to what we know:
            self.update(initial_info)
        else:
            init_info_keys = []
            
        self.root              = root_path
        self.models_path       = os.path.join(self.root, 'models')
        self.figs_path         = os.path.join(self.root, 'figs')
        self.csv_files_path    = os.path.join(self.root, 'csv_files')
        self.tensorboard_path  = os.path.join(self.root, 'tensorboard')
        
        self._create_dir_if_not_exists(self.root)
        self._create_dir_if_not_exists(self.models_path)
        self._create_dir_if_not_exists(self.figs_path)
        self._create_dir_if_not_exists(self.csv_files_path)
        self._create_dir_if_not_exists(self.tensorboard_path)
        
        
        # Add internal info so it will be saved
        # by _save_self():
        self['root_path']               = self.root
        self['_models_path']        	= self.models_path
        self['_chart_figs_path']    	= self.figs_path
        self['_csv_files_path']     	= self.csv_files_path
        self['_tensorboard_files_path'] = self.tensorboard_path

        # No csv writers yet; will be a dict
        # of CSV writer instances keyed by file
        # names (without directories):
        self.csv_writers = {}

        if 'class_names' not in init_info_keys:
            self['class_names'] = None

        # External info
        self['root_path'] = self.root

        self._save_self()

    # --------------- Public Methods --------------------

    #------------------------------------
    # save 
    #-------------------
    
    def save(self, item=None, fname=None):
        '''
        Save any of:
            o pytorch model
            o dictionaries or array-likes (incl. pandas Series), 
                like logits, probabilities, etc.
            o figures
            o this experiment
            
        If no item is provided, saves this experiment.
        Though it is automatically saved whenever a change
        is made.
        
        Saving behaviors:
            o Models: 
                 if fname exists, the name is extended
                 with '_<n>' until it is unique among this
                 experiment's already saved models. Uses
                 pytorch.save
            o Dictionaries and array-likes:
                 If a csv DictWriter for the given fname
                 exists, dicts are converted to lists using
                 list(dict.values()). The result is written as
                 a row to the csv writer. 
                 
                 If no DictWriter exists, one is created from 
                 the dict keys, or in case of array-like, using
                 range(len())
                 
            o Figures:
                 if fname exists, the name is extended
                 with '_<n>' until it is unique among this
                 experiment's already saved figures. Uses
                 plt.savefig with file format taken from extension.
                 If no extension provided in fname, default is PDF 

        :param item:
        :type item:
        :param fname:
        :type fname:
        '''
        
        if item is None:
            self._save_self()
            return
        
        if type(item) == pd.Series:
            item = list(item)
        
        # If item is given, fname must also be provided:
        if fname is None:
            raise ValueError("Must provide file name.")
        
        if type(item) == nn:
            model = item
            # A pytorch model
            dst = os.path.join(self.models_path, fname)
            if os.path.exists(dst):
                dst = Utils.unique_fname(self.models_path, fname)
            torch.save(model.state_dict(), dst)
            return dst
        
        if type(item) == dict or type(item) == list:
            # Dicts and lists are always stored as CSVs
            
            # Do we already have a csv writer for the given fname?
            dst = os.path.join(self.csv_files_path, fname)
            # Do we already have csv writer for this file:
            try:
                # Keys are the fname with '.csv':
                csv_writer_key = Path(fname).stem
                csv_writer = self.csv_writers[csv_writer_key]
            except KeyError:
                
                # No CSV writer yet:
                if type(item) == list:
                    header = range(len(item))
                else:
                    header = list(item.keys())

                # Ensure a .csv suffix:
                if not Path(dst).suffix == '.csv':
                    dst += '.csv'
                fd = open(dst, 'w')
                csv_writer = csv.DictWriter(fd, header)
                # Save the fd with the writer obj so
                # we can flush() when writing to it:
                csv_writer.fd = fd
                csv_writer.writeheader()
                self.csv_writers[csv_writer_key] = csv_writer

            # If given a row, rather than a dict
            # for the row, create a dict on the fly
            # before saving to csv:
            try:
                csv_writer.writerow(item)
            except AttributeError:
                fld_names = csv_writer.fieldnames
                if len(item) != len(fld_names):
                    raise ValueError(f"Row for this csv file must have {len(fld_names)} elements")
                tmp_dict = {k : v for k,v in zip(fld_names, item)}
                csv_writer.writerow(tmp_dict)
                
            csv_writer.fd.flush()
            return dst

        if type(item) == plt.Figure:
            fig = item
            fname_ext   = Path(fname).suffix
            # Remove the leading period if extension provided:
            file_format = 'pdf' if len(fname_ext) == 0 else fname_ext[1:] 
            dst = os.path.join(self.figs_path, fname)
            if os.path.exists(dst):
                dst = Utils.unique_fname(self.figs_path, fname)
            plt.savefig(fig, dpi=150, format=file_format)
            return dst


    #------------------------------------
    # load 
    #-------------------
    
    @classmethod
    def load(cls, path):
        '''
        Create an Experiment instance from a previously
        saved JSON export
        
        :param path: path to Experiment json export
        :type path: str
        '''
        if not os.path.exists(path):
            raise FileNotFoundError(f"Experiment {path} does not exist.")
        if not os.path.isdir(path):
            raise ValueError(f"Experiment path must be to a directory, not {path}")
        
        full_path = os.path.join(path, 'experiment.json')
        with open(full_path, 'r') as fd:
            restored_dict_contents = json.load(fd)
        
        exp_inst = Experiment(path, restored_dict_contents) 
            
        exp_inst.root              = exp_inst['root_path']
        exp_inst.models_path       = exp_inst['_models_path'] 
        exp_inst.figs_path         = exp_inst['_chart_figs_path'] 
        exp_inst.csv_files_path    = exp_inst['_csv_files_path']
        exp_inst.tensorboard_files_path = exp_inst['_tensorboard_files_path']

        # Open CSV writers if needed.
        # First, get dict CSV writer names that will
        # be the keys of the self.csv_writers dict: 
        csv_writer_names = exp_inst['csv_writer_names']
        for writer_name in csv_writer_names:
            file_path = os.path.join(self.root,
                                     exp_inst.csv_files_path,
                                     writer_name + '.csv'
                                     ) 
            with open(file_path, 'a') as fd:
                csv_writer = csv.DictWriter(fd)
                # For flush() and close() later on:
                csv_writer.fd = fd
                exp_inst.csv_writers[writer_name] = csv_writer
        
        del exp_inst['csv_writer_names']
        
        # Next, open a dict writer for each file:
        for fname in exp_inst.csv_writers.keys():
            fd = open(exp_inst.csv_writers[fname], 'a')
            exp_inst.csv_writers[fname] = csv.DictWriter(fd)
            # Add the fd to the writer, so we can 
            # flush() when needed:
            exp_inst.csv_writers[fname].fd = fd
            
        return exp_inst

    #------------------------------------
    # abspath
    #-------------------
    
    def abspath(self, fname, extension):
        '''
        Given the fname used in a previous save()
        call, and the file extension (.csv or .pth,
        .pdf, etc.): returns the current full path to the
        respective file.
        
        :param fname: name of the item to be retrieved
        :type fname: str
        :param extension: file extension, like 'jpg', 'pdf', 
            '.pth', 'csv', '.csv'
        :type extension: str
        :returns absolute path to corresponding file
        :rtype str
        '''
        if not extension.startswith('.'):
            extension = f".{extension}"
        true_fname = Path(fname).stem + extension
         
        for root, _dirs, files in os.walk(self.root):
            if true_fname in files:
                return os.path.join(root, true_fname)


    #------------------------------------
    # close 
    #-------------------
    
    def close(self):
        '''
        Close all csv writers, and release other resources
        if appropriate
        '''
        
        for csv_writer in self.csv_writers.values():
            # We previously added the fd of the file
            # to which the writer is writing in the 
            # csv.DictWriter instance. Use that now:
            csv_writer.fd.close()
            
        self._save_self()

    #------------------------------------
    # clear 
    #-------------------
    
    def clear(self, safety_str):
        '''
        Removes all results from the experiment.
        Use extreme caution. For safety, the argument
        must be "Yes, do it"
        
        :param safety_str: keyphrase "Yes, do it" to ensure
            caller thought about the call
        :type safety_str: str
        '''
        if safety_str != 'Yes, do it':
            raise ValueError("Saftey passphrase is not 'Yes, do it', so experiment not cleared")
        shutil.rmtree(self.root)


    # --------------- Private Methods --------------------


    #------------------------------------
    # _save_self 
    #-------------------
    
    def _save_self(self):
        '''
        Write json of info about this experiment
        to self.root/experiment.json
        '''
        
        # Insist on the class names to have been set:
        #try:
        #    self['class_names']
        #except KeyError:
        #    raise ValueError("Cannot save experiment without class_names having been set first")

        # CSV writers are keyed from file names
        # and map to open csv.DictWriter. Create a new 
        # dict entry to hold the csv writer names for
        # later recreation in load():
        
        self['csv_writer_names'] = list(self.csv_writers.keys()) 
        
        with open(os.path.join(self.root, 'experiment.json'), 'w') as fd:
            json.dump(self, fd)

    #------------------------------------
    # _create_dir_if_not_exists 
    #-------------------
    
    def _create_dir_if_not_exists(self, path):
        
        if not os.path.exists(path):
            os.makedirs(path)
            return
        # Make sure the existing path is a dir:
        if not os.path.isdir(path):
            raise ValueError(f"Path should be a directory, not {path}")
