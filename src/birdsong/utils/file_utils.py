'''
Created on Jan 21, 2021

@author: paepcke
'''

import os
import natsort
from pathlib import Path

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
        Given a root directory, return a list 
        of Path instances that correspond to all 
        directories under the root, which contain at 
        least one image file.
        
        The paths will be naturally sorted, and
        absolute.
        
        Directory names that begin with a period ('.')
        are excluded.

        The assumption is that the names of the last
        path element of the returned directories are 
        class names. This assumption is often made in 
        torchvision packages.  

        @param data_root: root directory for search 
        @type data_root: str
        @return a list of Path instances fo rfull-path 
            directories whose names are target classes.
        @rtype [pathlib.Path]
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
            
        # Sort by the class names (not the full paths):
        class_dict = {path.stem : path
                         for path in class_paths
                      }
        sorted_by_class = natsort.natsorted(class_dict.keys())
        class_paths_sorted = [class_dict[class_name]
                                 for class_name in sorted_by_class
                              ]
        return class_paths_sorted

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
