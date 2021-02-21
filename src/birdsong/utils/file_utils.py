'''
Created on Jan 21, 2021

@author: paepcke
'''

from _collections import OrderedDict
import os
from pathlib import Path

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
