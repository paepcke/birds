'''
Created on Jan 21, 2021

@author: paepcke
'''

from _collections import OrderedDict
import datetime
import os
from pathlib import Path
import re
import csv

import natsort


class FileUtils(object):
    '''
    classdocs
    '''

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    #------------------------------------
    # find_class_paths
    #-------------------

    @classmethod
    def find_class_paths(cls, data_root):
        '''
        Given a root directory, return an Ordered dict
        mapping class names to lists of directories
        that contain at least one sample of that class. 
        
        Both the class names (i.e. keys), and the
        lists of directories (i.e. values) will be
        naturally sorted, and absolute.
        
        Directory names that begin with a period ('.')
        are excluded.

        The assumption is that the names of the last
        path element of the returned directories are 
        class names. This assumption is often made in 
        torchvision packages.  

        @param data_root: root directory for search 
        @type data_root: str
        @return dict mapping target classes to a list
            of directories that contain samples of that
            class. The directories will be Path objs
        @rtype Ordered{str : [Path]}
        '''
        
        # If path is relative, compute abs
        # path relative to current dir:
        if not os.path.isabs(data_root):
            data_root = os.path.join(os.path.dirname(__file__), 
                                     data_root)
        class_paths = set([])
        for root, _dirs, files in os.walk(data_root):
            
            if len(files) == 0:
                # Found only directories:
                continue
            
            # For convenience, turn the file paths
            # into Path objects:
            file_Paths = [Path(name) for name in files]
            root_Path  = Path(root)
            
            # Pick out files with an image extension:
            full_paths = []
            for file_path in file_Paths:
                if file_path.suffix in cls.IMG_EXTENSIONS \
                   and not file_path.parent.stem.startswith('.'):
                    full_paths.append(Path.joinpath(root_Path, file_path).parent)
            
            # Using union in this loop guarantees
            # uniqeness of the gathered class names:
            
            class_paths = class_paths.union(set(full_paths))
        
        # Order the paths so that all machines
        # have the same sample-id assignements later:
        
        class_paths = natsort.natsorted(list(class_paths))
         
        # Get dict {class-name : [paths-to-samples-of-that-class]}
        class_path_dict = OrderedDict()
        
        for class_path in class_paths:
            try:
                # dict[class-name] gets more
                # paths that hold samples of class-name:
                class_path_dict[class_path.stem].append(class_path)
            except KeyError:
                class_path_dict[class_path.stem] = [class_path]
                
        # Now ensure that the list of directories,
        # (i.e. the values) for each class-name's entry
        # are also sorted:

        # Use copy of class_path_dict for iteration,
        # b/c we modify class_path_dict in the loop:
        
        class_path_dict_copy = class_path_dict.copy()
        for class_name, dirs in class_path_dict_copy.items():
            class_path_dict[class_name] = natsort.natsorted(dirs)

        return class_path_dict

    #------------------------------------
    # find_class_names 
    #-------------------

    @classmethod
    def find_class_names(cls, dir_name):
        '''
        Like find_class_paths(), but only 
        the trailing directory names of the 
        paths, i.e. the class names, are
        returned.
        
        @param dir_name: root of dir to search
        @type dir_name: str
        @return: naturally sorted list of class names 
        @rtype: [str]
        '''
        class_names = set([])
        for root, _dirs, files in os.walk(dir_name):
            full_paths = [os.path.join(root, file)
                           for file in files
                            if Path(file).suffix in cls.IMG_EXTENSIONS
                            ]
            class_names = class_names.union(set([Path(full_path).parent.name
                                                     for full_path
                                                      in full_paths
                                                      ])
                                                      )              
        return natsort.natsorted(class_names)

    #------------------------------------
    # construct_filename 
    #-------------------
    
    @classmethod
    def construct_filename(cls, 
                           property_dict,
                           prefix=None,
                           suffix=None, 
                           incl_date=False):
        '''
        Given a dict of property names and
        associated values, create a filename
        that includes all the information held
        in the keys and values of the dict.
        Ex:
             {
               'lr' : 0.001,
               'bs' : 32,
               optimizer : 'Adam'
             }
             
        would return the string:
            
            'lr_0.001_bs_32_optimizer_Adam'
            
        If a prefix is provided, it will lead the
        string. Example: "Exp" would yield:
        
            'EXP_lr_0.001_bs_32_optimizer_Adam'
            
        If suffix is provided, it will be appended to the
        name: Example, suffix='.csv':
        
            'EXP_lr_0.001_bs_32_optimizer_Adam.csv'
            
        Finally, if incl_date is True, a timestamp is added
        at the start of the returned name, or right after
        the prefix
        
        @param property_dict: names and values to include
        @type property_dict: {str : Any}
        @param prefix: leading part of file name
        @type prefix: str
        @param suffix: trailing part of file name
        @type suffix: str
        @param incl_date: whether or not to include current
            data in the file name
        @type incl_date: bool
        @return: a string appropriate for use as a filename
        @rtype: str
        '''
        fname = prefix if prefix is not None else ''
        if incl_date:
            fname += f"_{cls.file_timestamp()}"
        for prop_name, prop_val in property_dict.items():
            fname += f"_{prop_name}_{str(prop_val)}"
            
        if suffix is not None:
            fname += suffix
            
        return fname


    #------------------------------------
    # user_confirm
    #-------------------
    
    @classmethod
    def user_confirm(cls, prompt_for_yes_no, default='Y'):
        resp = input(f"{prompt_for_yes_no} (default {default}): ")
        if resp in ('y','Y','yes','Yes', ''):
            return True
        else:
            return False

    #------------------------------------
    # file_timestamp
    #-------------------
    
    @classmethod
    def file_timestamp(cls):
        '''
        Finds current time, removes milliseconds,
        and replaces colons with underscores. The
        returned string is fit for inclusion in a
        filename.
        
        @return: string for inclusion in filename
        @rtype: str
        '''
        # Remove the msecs part:
        # Replace colons with underscores:
        timestamp = datetime.datetime.now().isoformat()
        timestamp = re.sub(r'[.][0-9]{6}', '', timestamp)
        timestamp = timestamp.replace(':', '_')
        return timestamp
        
# ----------------------- CSVWriterFDAccessible -------

class CSVWriterCloseable:
    '''
    Wrapper around csv writer: takes a 
    file name to use and:
    
       o Creates intermediate dirs if needed,
       o Opens the file in the given mode
       o Adds close() method which closes that 
         underlying fd.
       o Creates a writer object into an inst var.

    Only supports the writerow() method of
    csv writers.
    
    This class should inherit from a csv.Writer
    class if it existed. But as of Python 3.9 it
    does not. 
    '''
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, fname, mode='w', **kwargs):
        
        acceptable_modes = ['w','a']
        if mode not in acceptable_modes:
            raise ValueError(f"Mode must be one of {acceptable_modes}")
        
        try:
            os.makedirs(os.path.dirname(fname))
        except FileExistsError:
            # All intermediate dirs exist
            pass
        self._fname = fname
        self._fd = open(fname, mode, newline='')
        self.writer = csv.writer(self._fd, **kwargs)

    #------------------------------------
    # writerow 
    #-------------------
    
    def writerow(self, seq):
        self.writer.writerow(seq)

    #------------------------------------
    # close 
    #-------------------
    
    def close(self):
        try:
            self._fd.close()
        except Exception as e:
            raise IOError(f"Could not close CSV writer: {repr(e)}")

    # --------- Properties ----------
    
    @property
    def fd(self):
        return self._fd

    @property
    def fname(self):
        return self._fname
    