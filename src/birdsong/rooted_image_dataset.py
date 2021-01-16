'''
The MultiRootImageDataset class supports dataloaders
for images. 

The features are similar to torchvision.datasets.ImageFolder
in that the class:
   
    o Takes pointer(s) into a file tree holding image
      files of various formats (png, jpg, etc.)
    o Assigns integer sample_id numbers to 
      each image file in the tree
    o Assigns integer class IDs to each classification
      target
    o Returns an image tensor when given a sample_id

The class implements a dict-like pytorch dataset.

Like ImageFolder, MultiRootImageDataset expects any one 
subdirectory to contain samples for a single classification
target. The name of the subdirectory must be the name
of the corresponding target.

In contrast to ImageFolder, which requires a single file 
directory root above the class folders, MultiRootImageDataset
accepts multiple roots, merging all samples into a single
dataset. Additionally, root specifications handle '~', as
well as globbed file names:

         ~/foo/* 

NOTE: it is the client's responsibility to prevent target class
      name clashes:
      
         root1
            cars
               audi
                  audi1.png
                  audi2.png
               bmw
                  bmw1.jpg
            flowers
               rose
                  rose1.jpg
               tulip
                  tulip1.jpg
                  tulip2.jpg
        
        root 2
             cars
                 ...
                 
      will raise a DuplicateTarget exception, because
      'cars' occurs under multiple roots. 

Instances of the parent class below creates simple bookkeeping 
structures that can easily be extended with additional
targets or sample_ids. This extension is implemented
in the subclass MultiRootImageDataset.

'''

from _collections import OrderedDict
import os
from pathlib import Path

import natsort
import torch
from torchvision import transforms
from torchvision.datasets import folder

import numpy as np


# Sorting such that numbers in strings
# do the right thing: "foo2" after "foo10": 
# Should actually be 1:3 but broke the system:
SAMPLE_WIDTH  = 400 # pixels
SAMPLE_HEIGHT = 400 # pixels

class SingleRootImageDataset:
    '''
    Constructs a pytorch mapping-type dataset
    that provides:
        o Recursively finds all target-class directories under
          a single, given root. Ignores non_image files
        o Assumes that the directory names containing
          images are target class names. This assumption 
          matches ImageFolder
        o Provides map from sample_ids to absolute file
          names of corresponding images
    '''
    
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 filepath,
                 sample_width=SAMPLE_WIDTH,
                 sample_height=SAMPLE_HEIGHT
                 ):
        '''
        
        @param filepath: path to root of the target
            images, each target being a directory with
            samples for that target
        @type filepath: str
        @param sample_width: pixel width to which images are 
            to be scaled
        @type sample_height: pixel height to which images are 
            to be scaled
        @return: new instance of MultiRootImageDataset
        @rtype MultiRootImageDataset
        @raises ValueError if any of the roots is not a string.
        '''
        self.transform_img = transforms.Compose([
                transforms.Resize((sample_width, sample_height)),  # should actually be 1:3 but broke the system
                transforms.ToTensor()])

        # Build three data structures:
        #     class_name --> class id int needed for model building
        #                    No strings allowed for class_to_id
        #     sample_id  --> sample file path
        #     sample_id  --> class id
        #
        # Use OrderedDict for some of the above maps to
        # ensure that calls to keys() or values() always
        # return the same sequence.
        
        # The name of each folder in the train subdir
        # is the name of a class (i.e. species); go through this
        # folder/species sequence:
        
        # The class_name --> class-id map:
        
        # Sort the directory such that "foo2" after "foo10".
        # Good for unittesting. Dataloaders can shuffle
        # to introduce randomness later:
        
        class_paths = self.find_class_paths(filepath)
        
        self.class_to_id = OrderedDict()

        # Note: We can have images of a class
        # spread across the class paths. Only
        # assign a class id the first time we
        # encounter any one class:
         
        for class_id, class_path in enumerate(class_paths):
            class_name = class_path.stem
            
            # Add *only new* class and its ID to the class_to_id
            # dict. Skip dirs that start with dot ('.'), such
            # as ".DS_Store":
            
            if class_name not in self.class_to_id.keys() and \
                not class_name.startswith('.'):
                
                self.class_to_id[class_name] = class_id
                
        self.sample_id_to_path = OrderedDict({})
        self.sample_id_to_class = OrderedDict({})
        
        # Will be the (relatively) small list of
        # integer IDs of the class labels. Will
        # be computed first time the class_id_list()
        # method is called:
        
        self._class_id_list = None 
        
        # First sample ID:
        sample_id_start = 0
        
        # Go through the samples in each folder (i.e. class):
        
        for sample_folder in class_paths:
            class_name    = os.path.basename(sample_folder)
            class_id      = self.class_to_id[class_name]
            
            # List of full paths to each sample of current class.
            # Exclude names that start with a dot, such as
            # the .DS_Store that macos likes to spread around
            # file systems:
            folder_content  = [os.path.join(sample_folder, sample_path)
                                 for sample_path 
                                 in natsort.natsorted(os.listdir(sample_folder))
                                 if not sample_path.startswith('.')
                                 ]
            # IDs we will assign to the samples in this folder:
            sample_id_range = range(sample_id_start, sample_id_start + len(folder_content)) 
            
            # Create sample id --> filename map for just the 
            # samples in this folder:
            sample_id_map = OrderedDict({sample_id : folder_content[i]
                                            for i, sample_id in enumerate(sample_id_range)
                                         })
            
            # Append this folder's sample id --> filename to
            # our emerging final map (dict1.update(dict2) appends):
            self.sample_id_to_path.update(sample_id_map)
            
            # Build sample id --> class ID; the class ID is 
            # the same for all samples in this folder:
            self.sample_id_to_class.update({sample_id : class_id
                                            for sample_id in sample_id_range 
                                            })
            # Update where to start the IDs for the
            # next folder
            sample_id_start += len(sample_id_range)
            

    #------------------------------------
    # sample_ids
    #-------------------

    def sample_ids(self):
        '''
        Return an iterator over all 
        sample IDs. Multiple calls guaranteed
        to return IDs in the same order each time
        
        :returns List of sample identifiers
        :rtype [int]
        '''
        
        return self.sample_id_to_class.keys()

    #------------------------------------
    # sample_classes 
    #-------------------
    
    def sample_classes(self):
        '''
        Returns an np array of class ids
        of all samples. The sequence order is
        guaranteed to be the same for each call.
        Furthermore, the sequence matches that of
        the sequence returned from sample_ids()
        
        :return: Array over all class labels in
        sample_id order.
        :rtype: np.array
        '''
        
        return np.array(list(self.sample_id_to_class.values()))

    #------------------------------------
    # class_id_list 
    #-------------------
    
    def class_id_list(self):
        '''
        Return a list with the integer class ids
        between which the classifier is to discriminate.
        
        @return: list of integer class labels
        @rtype List(int) 
        '''
        if self._class_id_list is not None:
            return self._class_id_list

        self._class_id_list = list(self.class_to_id.values())
        return self._class_id_list

    #------------------------------------
    # class_names 
    #-------------------
    
    def class_names(self):
        return list(self.class_to_id.keys())

    #------------------------------------
    # file_from_sample_id 
    #-------------------
    
    def file_from_sample_id(self, sample_id):
        '''
        Given a sample_id, return the absolute
        file path of the corresponding sample
        in the file system.
        
        @param sample_id: sample ID to look up
        @type sample_id: int
        '''
        return os.path.abspath(self.sample_id_to_path[sample_id])


    #------------------------------------
    # find_class_paths
    #-------------------

    def find_class_paths(self, data_root):
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
            full_paths = [Path.joinpath(root_Path, file_Path).parent
                           for file_Path in file_Paths
                            if file_Path.suffix in self.IMG_EXTENSIONS
                           and not file_Path.parent.stem.startswith('.')
                            ]
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

    def find_class_names(self, dir_name):
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
                            if Path(file).suffix in self.IMG_EXTENSIONS
                            ]
            class_names = class_names.union(set([Path(full_path).parent.name
                                                     for full_path
                                                      in full_paths
                                                      ])
                                                      )              
        return natsort.natsorted(class_names)


    #------------------------------------
    # __len__
    #-------------------

    def __len__(self):
        return len(self.sample_id_to_path)

    #------------------------------------
    # __getitem__ 
    #-------------------

    def __getitem__(self, sample_id):
        '''
        Returns a two-tuple: image tensor, target class
        
        @param sample_id: Image sample identifier
        @type sample_id: int
        @return Image loaded as a PIL, then downsized,
            and transformed to a tensor.
        '''

        img = folder.default_loader(self.sample_id_to_path[sample_id])
        img_tensor = self.transform_img(img)
        label = self.sample_id_to_class[sample_id]

        return (img_tensor, torch.tensor(label))

# ------------------------- Class MultiRootImageDataset --------

class MultiRootImageDataset(SingleRootImageDataset):
    '''
    Combines image samples from multiple directory roots to be combined
    into a single dataset. Directory structure assumption
    under each of the roots:

     root_n:
         <class1_dir>
             sample_img1.png
             sample_img2.jpg
                 ...
         <class1_dir>
             sample_img1.png
             sample_img2.jpg
                 ...

    As in roots = ['/foo/bar/animals', '/foo/bar/cars']:
     /foo/bar/animals
         dog
             dog1.png
             dog2.jpg
         cat 
             cat1.png
             cat2.jpg
             
     /foo/bar/cars
        BMW
             bmw1.jpg
             bmw2.jpg
        Audi
             audi1.jpg
             audi2.jpg

    '''
    
    #------------------------------------
    # __new__ 
    #-------------------
    
    def __new__(cls,
                roots,
                sample_width=SAMPLE_WIDTH,
                sample_height=SAMPLE_HEIGHT
                ):
        '''
        Creates a dataset with images from multiple
        file systems roots:
        
            root1                   root2
         spam   eggs               cars  bicycles
        <imgs> <imgs>             <imgs>  <imgs>
        
        The final dataset will include all images under
        spam, eggs, cars, and bicycles. 
        
        Creates an instance of the superclass
        SingleRootImageDataset from the first of the passed-in
        root directories. This first instance will
        be the final dataset instance.
        
        Then loop through each of the other roots,
        and create a temporary superclass instance.
        Join the temporary instance's bookkeeping dictionaries 
        to those of this emerging instance. 
        
        Return this instance. 
        
        @param cls: class for which this instance will 
            manufacture an instance
        @type cls: bird_dataset.MultiRootImageDataset
        @param roots: individual, or list of paths under which
            subdirectories for classification targets are to
            be included
        @type roots: {str | [str]}
        @param sample_width: pixel width to which images are 
            to be scaled
        @type sample_height: pixel height to which images are 
            to be scaled
        @return: new instance of MultiRootImageDataset
        @rtype MultiRootImageDataset
        @raises ValueError if any of the roots is not a string.
        '''

        if type(roots) != list:
            roots = [roots]

        # Resolve wildcards, like my_root/*:
        all_roots = []

        # Filter out non-directories:
        for root in roots:
            if os.path.isdir(root):
                all_roots.append(root)

        self = SingleRootImageDataset(all_roots[0], 
                                      sample_width, 
                                      sample_height)
        self.sample_width  = sample_width
        self.sample_height = sample_height

        for one_root in all_roots[1:]:
            cls.extend_dataset(self, one_root)
            
        return self
            
    #------------------------------------
    # extend_dataset 
    #-------------------
    
    def extend_dataset(self, root):
        tmp_dataset = SingleRootImageDataset(root, 
                                             sample_width=self.sample_width,
                                             sample_height=self.sample_height
                                             )
        # Add the new sample IDs and targets to
        # this evolving instance's data structures:
        
        # Highest class id that exists already in the 
        # emerging instance:
        max_class_id = max(self.class_to_id.values())

        for (tmp_class_name, tmp_class_id) in tmp_dataset.class_to_id.items():
            self.class_to_id[tmp_class_name] = 1 + tmp_class_id + max_class_id

        # Same with the sample IDs: must shift the tmp
        # dataset's sample_ids to continue the emerging
        # instance's sample_ids:
        
        max_sample_id = max(self.sample_id_to_class.keys())
        
        for (tmp_sample_id, tmp_path) in tmp_dataset.sample_id_to_path.items():
            self.sample_id_to_path[1 + tmp_sample_id + max_sample_id] = tmp_path
        
        for (tmp_sample_id, tmp_class) in tmp_dataset.sample_id_to_class.items():
            self.sample_id_to_class[1 + tmp_sample_id + max_sample_id] = 1 + tmp_class + max_class_id
