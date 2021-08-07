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

import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn

from birdsong.utils.neural_net_config import NeuralNetConfig, ConfigError

'''
TODO:
   o Doc that you can store hparams individually as dict,
       or use NeuralNetConfig.
   o Turn into separate project; needs NeuralNetConfig and parts of Utils
   o When saving dataframes with index_col, use that also 
       when using pd.read_csv(fname, index_col) to get the
       index installed
'''

class ExperimentManager(dict):
    '''
    Container to hold all information about an experiment:
    the pytorch model parameters, location of model snapshot,
    location of csv files created during training and inference.
    
    An experiment instance is saved and loaded via
        o <exp-instance>.save(fp), and 
        o ExperimentManager.load(fp)
        
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
    # __new__
    #-------------------
    
    def __new__(cls, root_path, initial_info=None):
        '''
        
        :param initial_info: optionally, a dict with already
            known facts about the experiment.
        :type initial_info:
        '''

        self = super().__new__(cls)

        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if not os.path.isdir(root_path):
            raise ValueError(f"Root path arg must be a dir, not {root_path}")

        # No csv writers yet; will be a dict
        # of CSV writer instances keyed by file
        # names (without directories):
        self.csv_writers = {}

        if initial_info is not None:
            # Must be a dict:
            if not type(initial_info) == dict:
                raise TypeError(f"Arg initial_info must be a dict, not {initial_info}")
            init_info_keys = list(initial_info.keys())
            # Add passed-in info to what we know:
            self.update(initial_info)
        else:
            init_info_keys = []

        if 'class_names' not in init_info_keys:
            self['class_names'] = None

        # External info
        self['root_path'] = self.root


        self.root              = root_path
        self.models_path       = os.path.join(self.root, 'models')
        self.figs_path         = os.path.join(self.root, 'figs')
        self.csv_files_path    = os.path.join(self.root, 'csv_files')
        self.tensorboard_path  = os.path.join(self.root, 'tensorboard')
        self.hparams_path      = os.path.join(self.root, 'hparams')
        
        self._create_dir_if_not_exists(self.root)
        self._create_dir_if_not_exists(self.models_path)
        self._create_dir_if_not_exists(self.figs_path)
        self._create_dir_if_not_exists(self.csv_files_path)
        self._create_dir_if_not_exists(self.tensorboard_path)
        self._create_dir_if_not_exists(self.hparams_path)
        
        
        # Add internal info so it will be saved
        # by _save_self():
        self['root_path']               = self.root
        self['_models_path']            = self.models_path
        self['_figs_path']                = self.figs_path
        self['_csv_files_path']         = self.csv_files_path
        self['_tensorboard_files_path'] = self.tensorboard_path
        self['_hparams_path']           = self.hparams_path

        
        return self

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, _root_path):
        '''
        The __new__() method did most of the work.
        Now, if this is a real new instance, as opposed
        to one that is reconstituted via load(), save
        the experiment for the first time.
        '''

        self._save_self()

    # --------------- Public Methods --------------------

    #------------------------------------
    # add_hparams
    #-------------------
    
    def add_hparams(self, config_fname):
        '''
        Read the given config file, creating a
        NeuralNetConfig instance. Store that in
        the 'config' key. Also, write a json copy to 
        the hparams subdir. 
        
        :param config_fname: path to config file that is
            readable by the standard ConfigParser facility
        :type config_fname: src
        :return a NeuralNetConfig instance 
        :rtype NeuralNetConfig
        '''
        
        config = self._initialize_config_struct(config_fname)
        self['config'] = config
        # Save a json representation in the hparams subdir:
        config_path = os.path.join(self.hparams_path, 'config.json')
        config.to_json(config_path, check_file_exists=False)
        return config 

    #------------------------------------
    # save 
    #-------------------
    
    def save(self, item=None, fname=None, index_col=None):
        '''
        Save any of:
            o pytorch model
            o dictionaries
            o lists
            o pd.Series
            o pd.DataFrame
            o figures

            o this experiment itself
            
        If no item is provided, saves this experiment.
        Though it is automatically saved anyway whenever 
        a change is made, and when close() is called.
        
        The fname is the file name with or without .csv extension.
        The file will exist in the self.csv_files_path under the experiment
        root no matter what path is provided by fname. Any parent of 
        fname is discarded. The intended form of fname is just like:
        
            logits
            prediction_numbers
            measurement_results
            
        The index_col is only relevant when saving
        dataframes. If provided, the df index (i.e. the row labels)
        are saved in the csv file with its own column, named
        index_col. Else the index is ignored.
        
        Saving behaviors:
            o Models: 
                 if fname exists, the name is extended
                 with '_<n>' until it is unique among this
                 experiment's already saved models. Uses
                 pytorch.save
            o Dictionaries and array-likes:
                 If a csv DictWriter for the given fname
                 exists. 
                 
                 If no DictWriter exists, one is created with
                 header row from the dict keys, pd.Series.index, 
                 pd.DataFrame.columns, or for simple lists, range(len())
                 
            o Figures:
                 if fname exists, the name is extended
                 with '_<n>' until it is unique among this
                 experiment's already saved figures. Uses
                 plt.savefig with file format taken from extension.
                 If no extension provided in fname, default is PDF 

        :param item: the data to save
        :type item: {dict | list | pd.Series | pd.DataFrame | torch.nn.Module | plt.Figure}
        :param fname: key for retrieving the file path and DictWriter
        :type fname: str
        :param index_col: for dataframes only: col name for
            the index; if None, index is ignored
        :type index_col: {None | str}
        :return: path to file
        :rtype: str
        '''
        
        if item is None:
            self._save_self()
            return

        # If item is given, fname must also be provided:
        if fname is None:
            raise ValueError("Must provide file name.")

        # Fname is intended for use as the key to the 
        # csv file in self.csv_writers, and as the file
        # name inside of self.csv_files_path. So, clean
        # up what's given to us: remove parent dirs and
        # the extension:
        fname = Path(fname).stem
        
        if type(item) == nn:
            model = item
            # A pytorch model
            dst = os.path.join(self.models_path, fname)
            if os.path.exists(dst):
                dst = self._unique_fname(self.models_path, fname)
            torch.save(model.state_dict(), dst)

        elif type(item) in (dict, list, pd.Series, pd.DataFrame):
            dst = self._save_records(item, fname, index_col)

        elif type(item) == plt.Figure:
            fig = item
            fname_ext   = Path(fname).suffix
            # Remove the leading period if extension provided:
            file_format = 'pdf' if len(fname_ext) == 0 else fname_ext[1:] 
            dst = os.path.join(self.figs_path, fname)
            if os.path.exists(dst):
                dst = self._unique_fname(self.figs_path, fname)
            plt.savefig(fig, dpi=150, format=file_format)

        self.save()
        return dst

    #------------------------------------
    # load 
    #-------------------
    
    @classmethod
    def load(cls, path):
        '''
        Create an ExperimentManager instance from a previously
        saved JSON export.
        
        :param path: path to ExperimentManager json export
        :type path: str
        '''
        if not os.path.exists(path):
            raise FileNotFoundError(f"ExperimentManager {path} does not exist.")
        if not os.path.isdir(path):
            raise ValueError(f"ExperimentManager path must be to a directory, not {path}")
        
        full_path = os.path.join(path, 'experiment.json')
        with open(full_path, 'r') as fd:
            restored_dict_contents = json.load(fd)
        
        exp_inst = ExperimentManager.__new__(ExperimentManager, path, restored_dict_contents) 
            
        exp_inst.root              = exp_inst['root_path']
        exp_inst.models_path       = exp_inst['_models_path'] 
        exp_inst.figs_path         = exp_inst['_figs_path'] 
        exp_inst.csv_files_path    = exp_inst['_csv_files_path']
        exp_inst.tensorboard_files_path = exp_inst['_tensorboard_files_path']
        exp_inst.hparams_path      = exp_inst['_hparams_path']
        
        # If key 'config' contains a JSON representation
        # of a NeuralNetConfig, reconstitute it:
        try:
            config_info = exp_inst['config']
            # Little sanity check:
            if config_info.startswith('{'):
                # Hopefully it's json:
                exp_inst['config'] = NeuralNetConfig.from_json(config_info)
        except:
            # No config info (i.e. add_hparams() not used)
            pass

        # Open CSV writers if needed.
        # First, get dict CSV writer names that will
        # be the keys of the self.csv_writers dict: 
        csv_writer_names = exp_inst['csv_writer_names']
        for writer_name in csv_writer_names:
            file_path = os.path.join(exp_inst.root,
                                     exp_inst.csv_files_path,
                                     writer_name + '.csv'
                                     )
            # Get the field names (i.e. header row):
            with open(file_path, 'r') as fd:
                col_names = csv.DictReader(fd).fieldnames
            
            fd = open(file_path, 'a')
            csv_writer = csv.DictWriter(fd, col_names)
            # For flush() and close() later on:
            csv_writer.fd = fd
            exp_inst.csv_writers[writer_name] = csv_writer
        
        # Get rid of the helper info that was added by
        # the save() method:
        del exp_inst['csv_writer_names']
            
        return exp_inst

    #------------------------------------
    # refresh 
    #-------------------
    
    #def refresh(self):
    #    '''
    # Re-reads the current json reflection of
    # its state. Only need for playing tricks
    # like another experiment instance operating
    # under the same root, and making changes.
    # '''
    # self.load(self.root)

    #------------------------------------
    # __setitem__
    #-------------------
    
    def __setitem__(self, key, item):
        '''
        Save to json every time the dict is changed.
        
        :param key: key to set
        :type key: str
        :param item: value to map to
        :type item: any
        '''
        super().__setitem__(key, item)
        # **** SCHEDULE THE SAVE     MUST 
        #self.save()

    #------------------------------------
    # update
    #-------------------
    
    def update(self, *args, **kwargs):
        '''
        Save to json every time the dict is changed.
        '''
        super().update(*args, **kwargs)
        self.save()

    #------------------------------------
    # __delitem__
    #-------------------
    
    def __delitem__(self, key):
        
        # Allow KeyError to bubble up to client:
        item = self[key]
        
        # If this is a file in the experiment
        # tree, delete it:
        if self._is_experiment_file(item):
            os.remove(item)

        # If a DictWriter, close it, and delete the file:
        elif type(item) == csv.DictWriter:
            path = item.fd.name
            item.fd.close()
            os.remove(path)

        self.save()
        super().__delete__(key)

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
    #_save_records 
    #-------------------

    def _save_records(self, item, fname, index_col=None):
        '''
        Saves items of types dict, list, Pandas Series,
        and DataFrames to a csv file. Creates the csv
        file and associated csv.DictWriter if needed. 
        If DictWriter has to be created, adds it to the
        self.csv_writers dict under the fname key.
        
        When creating DictWriters, the header line (i.e. column
        names) is obtain from:
        
            o keys() if item is a dict,
            o index if item is a pd.Series
            o columns if item is a pd.DataFrame
            o range(len(item) if item is a list
        
        If DictWriter already exists, adds the record(s)

        The fname is used as a key into self.csv_writers, and
        is expected to not be a full path, or to have an extension
        such as '.csv'. Caller is responsible for the cleaning. 
            
        :param item: data to be written to csv file
        :type item: {dict | list | pd.Series | pd.DataFrame}
        :param fname: name for the csv file stem, and retrieval key
        :type fname: str
        :param index_col: for dataframes only: name of index
            column. If None, index will be ignored
        :type index_col: {str | None}
        :return full path to the csv file
        :rtype src
        '''

        # Do we already have a csv writer for the given fname?
        dst = os.path.join(self.csv_files_path, f"{fname}.csv")
        if os.path.exists(dst):
            dst = self._unique_fname(self.csv_files_path, fname)

        # Do we already have csv writer for this file:
        try:
            csv_writer = self.csv_writers[fname]
        except KeyError:
            
            # No CSV writer yet:
            if type(item) == list:
                header = list(range(len(item)))
            elif type(item) == dict:
                header = list(item.keys())
            elif type(item) == pd.Series:
                header = item.index.to_list()
                item   = list(item)
            elif type(item) == pd.DataFrame:
                header = item.columns.to_list()                # Add a column name for the row labels:
                if index_col is not None:
                    header = [index_col] + header
            else:
                raise TypeError(f"Can only store dicts and list-like, not {item}")

            fd = open(dst, 'w')
            csv_writer = csv.DictWriter(fd, header)
            # Save the fd with the writer obj so
            # we can flush() when writing to it:
            csv_writer.fd = fd
            csv_writer.writeheader()
            self.csv_writers[fname] = csv_writer


        # Now the DictWriter exists; write the data:
        # If given a dataframe, write each row:
        if type(item) == pd.DataFrame:
            if index_col is None:
                # Get to ignore the index (i.e. the row labels):
                for row_dict in item.to_dict(orient='records'):
                    csv_writer.writerow(row_dict)
            else:
                for row_dict in self._collapse_df_index_dict(item, index_col):
                    csv_writer.writerow(row_dict)
        else:
            # If given an array, rather than a dict
            # for the row, create a dict on the fly
            # before saving to csv:
            try:
                csv_writer.writerow(item)
            except AttributeError:
                # DictWriter will complain that item 
                # has no 'keys()' method:
                fld_names = csv_writer.fieldnames
                if len(item) != len(fld_names):
                    raise ValueError(f"Row for this csv file must have {len(fld_names)} elements")
                tmp_dict = {k : v for k,v in zip(fld_names, item)}
                csv_writer.writerow(tmp_dict)
            
        csv_writer.fd.flush()
        return dst

    #------------------------------------
    # _collapse_df_index_dict
    #-------------------

    def _collapse_df_index_dict(self, df, index_col):
        '''
        Given a df, return a dict that includes the
        row indices (i.e. row labels) in the column names
        index_col. Example: given dataframe:

                  foo  bar  fum
            row1    1    2    3
            row2    4    5    6
            row3    7    8    9
        
        and index_col 'row_label', return:

            [
              {'row_label' : 'row1': 'foo': 1, 'bar': 2, 'fum': 3}, 
              {'row_label' : 'row2', 'foo': 4, 'bar': 5, 'fum': 6}, 
              {'row_label' : 'row3': 'foo': 7, 'bar': 8, 'fum': 9}
            ]

        :param df: dataframe to collapse
        :type df: pd.DataFrame
        :return array of dicts, each corresponding to one
            dataframe row
        :rtype [{str: any}]
        '''
        # Now have
        df_nested_dict = df.to_dict(orient='index')
        # Now have:
        #  {'row1': {'foo': 1, 'bar': 2, 'fum': 3}, 'row2': {'foo': 4, ...
        df_dicts = []
        for row_label, row_rest_dict in df_nested_dict.items():
            df_dict = {index_col : row_label}
            df_dict.update(row_rest_dict)
            df_dicts.append(df_dict)
        return df_dicts

    #------------------------------------
    # _initialize_config_struct 
    #-------------------
    
    def _initialize_config_struct(self, config_info):
        '''
        Return a NeuralNetConfig instance, given
        either a configuration file name, or a JSON
        serialization of a configuration.

          config['Paths']       -> dict[attr : val]
          config['Training']    -> dict[attr : val]
          config['Parallelism'] -> dict[attr : val]
        
        The config read method will handle config_info
        being None. 
        
        If config_info is a string, it is assumed either 
        to be a file containing the configuration, or
        a JSON string that defines the config.
        
        :param config_info: the information needed to construct
            the NeuralNetConfig instance: file name or JSON string
        :type config_info: str
        :return a NeuralNetConfig instance with all parms
            initialized
        :rtype NeuralNetConfig
        '''

        if isinstance(config_info, str):
            # Is it a JSON str? Should have a better test!
            if config_info.startswith('{'):
                # JSON String:
                config = NeuralNetConfig.from_json(config_info)
            else: 
                config = NeuralNetConfig(config_info)
        else:
            msg = f"Error: must pass a config file name or json, not {config_info}"
            raise ConfigError(msg)
            
        return config


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
        
        # If config facility is being used, turn
        # the NeuralNetConfig instance to json:
        try:
            config = self['config']
            if isinstance(config, NeuralNetConfig):
                self['config'] = config.to_json()
        except:
            # No config
            pass
        
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

    #------------------------------------
    # _path_elements 
    #-------------------
    
    def _path_elements(self, path):
        '''
        Given a path, return a dict of its elements:
        root, fname, and suffix. The method is almost
        like Path.parts or equivalent os.path method.
        But the 'root' may be absolute, or relative.
        And fname is provided without extension.
        
          foo/bar/blue.txt ==> {'root' : 'foo/bar',
                                'fname': 'blue',
                                'suffix: '.txt'
                                }

          /foo/bar/blue.txt ==> {'root' : '/foo/bar',
                                'fname': 'blue',
                                'suffix: '.txt'
                                }

          blue.txt ==> {'root' : '',
                        'fname': 'blue',
                        'suffix: '.txt'
                        }
        
        :param path: path to dissect
        :type path: str
        :return: dict with file elements
        :rtype: {str : str}
        '''
        
        p = Path(path)
        
        f_els = {}
        
        # Separate the dir from the fname:
        # From foo/bar/blue.txt  get ('foo', 'bar', 'blue.txt')
        # From /foo/bar/blue.txt get ('/', 'foo', 'bar', 'blue.txt')
        # From blue.txt          get ('blue.txt',)
        
        elements = p.parts
        if len(elements) == 1:
            # just blue.txt
            f_els['root']   = ''
            nm              = elements[0]
            f_els['fname']  = Path(nm).stem
            f_els['suffix'] = Path(nm).suffix
        else:
            # 
            f_els['root']     = os.path.join(*list(elements[:-1]))
            f_els['fname']    = p.stem
            f_els['suffix']   = p.suffix
        
        return f_els


    #------------------------------------
    # _unique_fname 
    #-------------------
    
    def _unique_fname(self, out_dir, fname):
        '''
        Returns a file name unique in the
        given directory. I.e. NOT globally unique.
        Keeps adding '_<i>' to end of file name.

        :param out_dir: directory for which fname is to 
            be uniquified
        :type out_dir: str
        :param fname: unique fname without leading path
        :type fname: str
        :return: either new, or given file name such that
            the returned name is unique within out_dir
        :rtype: str
        '''
        
        full_path   = os.path.join(out_dir, fname)
        fname_dict  = self._path_elements(full_path)
        i = 1

        while True:
            try:
                new_path = os.path.join(fname_dict['root'], fname_dict['fname']+fname_dict['suffix'])
                with open(new_path, 'r') as _fd:
                    # Succeed in opening, so file exists.
                    fname_dict['fname'] += f'_{i}'
                    i += 1
            except:
                # Couldn't open, so doesn't exist:
                return new_path
