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
import random

import natsort
import torch
from torchvision import transforms
from torchvision.datasets import folder

from birdsong.utils.utilities import FileUtils
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
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 filepath,
                 sample_width=SAMPLE_WIDTH,
                 sample_height=SAMPLE_HEIGHT,
                 to_grayscale=False,
                 percentage=None,
                 unittesting=False
                 ):
        '''
        If percentage P is provided, only P%
        of all available samples will be included
        in the set. Use if more samples are available
        than are needed. Speeds up training.
        
        Samples to include are randomly chosen after
        shuffling.
        
        :param filepath: path to root of the target
            images, each target being a directory with
            samples for that target
        :type filepath: str
        :param sample_width: pixel width to which images are 
            to be scaled
        :type sample_height: pixel height to which images are 
            to be scaled
        :param to_grayscale: do or do not convert images
            to 1-channel grayscale
        :type to_grayscale: bool
        :param percentage: if provided only a sample of 
            of all the available training samples is 
            included in the dataset.
        :type percentage: {int | float}
        :return: new instance of MultiRootImageDataset
        :rtype MultiRootImageDataset
        :raises ValueError if any of the roots is not a string.
        '''

        if unittesting:
            # Let unittests do things.
            return
        
        if percentage is not None:
            # Integrity check:
            if type(percentage) not in [int, float]:
                raise TypeError(f"Percentage must be int or float, not {type(percentage)}")
            if percentage < 1 or percentage > 100:
                raise ValueError(f"Percentage must be between 1 and 100, not {percentage}")

        img_transforms = [transforms.Resize((sample_width, sample_height)),  # should actually be 1:3 but broke the system
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                          ]
        if to_grayscale:
            img_transforms.append(transforms.Grayscale())
                          
        self.transform_img = transforms.Compose(img_transforms)

        #*******************        

        # Build three data structures:
        #
        #    self.class_to_id:
        #       class_name --> class id int needed for model building
        #                      No strings allowed for class_to_id
        #
        #    self.sample_id_to_path
        #       sample_id  --> sample file path
        #
        #    self.sample_id_to_class
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
        # to introduce randomness later.
        
        # Get list of directory paths that end with
        # a class name. NOTE: there can be multiple
        # paths for a given class:
        #     .../train/SPECIES1
        #     .../validate/SPECIES1
        
        # Get {class-name : [Path1, Path2,...]}
        # where Pathx are Path instances to directories
        # that (exclusively) contain samples of class-name.
        # No directories called '.' are included:
        
        class_directory_paths = FileUtils.find_class_paths(filepath)
        
        # Assign an int to each class in a way
        # that all replicas of this process assign
        # the same ints:
        self.class_to_id = OrderedDict()

        # Note: We can have images of a class
        # spread across the class paths. Only
        # assign a class id the first time we
        # encounter any one class:
         
        for class_id, class_name in enumerate(class_directory_paths.keys()):
            # Add *only new* class and its ID to the class_to_id
            # dict. Skip dirs that start with dot ('.'), such
            # as ".DS_Store":
            
            if class_name not in self.class_to_id.keys():
                self.class_to_id[class_name] = class_id
        
        # Generate integer sample IDs for each sample,
        # ensuring that all replica processes assign the
        # same number:
        
        self.sample_id_to_path = OrderedDict({})
        self.sample_id_to_class = OrderedDict({})
        
        # Will be the (relatively) small list of
        # integer IDs of the class labels. Will
        # be computed first time the class_id_list()
        # method is called:
        
        self._class_id_list = None 
        
        # First sample ID:
        sample_id_start = 0
        
        # Go through the samples in each of the folders
        # that hold samples of a class:
        # Result: 
        #    {class_id*****
        
        for class_name, sample_folder_list in class_directory_paths.items():
            # Get integer ID of the class name:
            class_id = self.class_to_id[class_name]

            for sample_folder in sample_folder_list:

                # List of full paths to each sample of current class.
                folder_content  = [os.path.join(sample_folder, sample_path)
                                   for sample_path 
                                   in natsort.natsorted(os.listdir(sample_folder))
                                   if Path(sample_path).suffix in FileUtils.IMG_EXTENSIONS
                                   ]

                # IDs we will assign to the samples in this folder:
                sample_id_range = range(sample_id_start, 
                                        sample_id_start + len(folder_content)) 
                
                # Create sample id --> filename map for just the 
                # samples in this folder:
                sample_id_map = OrderedDict({sample_id : folder_content[i]
                                                for i, sample_id in enumerate(sample_id_range)
                                             })
                
                # Append this folder's sample id --> filename
                # dict to our emerging final map (dict1.update(dict2) appends):
                self.sample_id_to_path.update(sample_id_map)
                
                # Build sample id --> class ID; the class ID is 
                # the same for all samples in this folder:
                self.sample_id_to_class.update({sample_id : class_id
                                                for sample_id in sample_id_range 
                                                })
                # Update where to start the IDs for the
                # next folder
                sample_id_start += len(sample_id_range)

        if percentage is None:
            # Use all samples
            return
        
        self.sample_id_to_class, self.sample_id_to_path = \
            self._cull_samples(self.sample_id_to_class, 
                               self.sample_id_to_path,
                               percentage
                               )

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
    # class_from_sample_id 
    #-------------------
    
    def class_from_sample_id(self, sample_id):
        '''
        Given a sample ID, return its class index.
        
        :param sample_id: ID to look up
        :type sample_id: int
        :return: given sample's class ID
        :rtype: int
        '''
        return self.sample_id_to_class[sample_id]

    #------------------------------------
    # class_id_list 
    #-------------------
    
    def class_id_list(self):
        '''
        Return a list with the integer class ids
        between which the classifier is to discriminate.
        
        :return: list of integer class labels
        :rtype List(int) 
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
        
        :param sample_id: sample ID to look up
        :type sample_id: int
        '''
        return os.path.abspath(self.sample_id_to_path[sample_id])

    #------------------------------------
    # sample_distribution 
    #-------------------
    
    def sample_distribution(self):
        '''
        Returns number of samples for each class
        in the dataset
        
        :return: list of tuples: (class_id, num_samples);
            one such tuple for each class_id
        :rtype: [(int, int)]
        '''
        
        class_ids, sample_counts = np.unique(self.sample_classes(), 
                                             return_counts=True
                                             )
        return list(zip(class_ids, sample_counts))

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
        
        :param sample_id: Image sample identifier
        :type sample_id: int
        :return Image loaded as a PIL, then downsized,
            and transformed to a tensor.
        '''

        img = folder.default_loader(self.sample_id_to_path[sample_id])
        img_tensor = self.transform_img(img)
        label = self.sample_id_to_class[sample_id]

        return (img_tensor, torch.tensor(label))

    #------------------------------------
    # sample_by_idx 
    #-------------------
    
    def sample_by_idx(self, idx_to_sample):
        '''
        Retrieve a sample by a list API.
        Samples IDs are stored in this dataset 
        in a fixed order. The idx_to_sample arg
        indexes into this conceptual list, and
        retrieves the corresponding sample.
        
        In the implementation, samples are stored as accessed
        through ordered dicts by the sample ids.
        
        :param idx_to_sample: index into the conceptual
            list of sample ids
        :type idx_to_sample: int
        :return: (sample, label) pair
        :rtype: (torch.Tensor, torch.Tensor)
        '''
        
        sample_id = self.sample_id_by_sample_idx(idx_to_sample)
        return self[sample_id]

    #------------------------------------
    # sample_id_by_sample_idx 
    #-------------------
    
    def sample_id_by_sample_idx(self, idx_to_sample):
        '''
        Retrieve a sample ID by a list API.
        Samples IDs are stored in this dataset 
        in a fixed order. The idx_to_sample arg
        indexes into this conceptual list, and
        retrieves the sample ID at that place in
        the list.
        
        In the implementation, samples are stored as accessed
        through ordered dicts by the sample ids.
        
        :param idx_to_sample: index into the conceptual
            list of sample ids
        :type idx_to_sample: int
        :return: sample ID
        :rtype: int
        '''
        
        # Only materialize the list of keys from 
        # self.sample_id_to_class.keys() once:
        cache_hit = False
        while not cache_hit:
            try:
                sample_id = self._sample_id_list[idx_to_sample]
                cache_hit = True
            except IndexError as e:
                raise IndexError(f"Index {idx_to_sample} is larger than the number of samples in the dataset ({len(self._sample_id_list)})") from e
            except AttributeError:
                # List has never been accessed; materialize it:
                self._sample_id_list = list(self.sample_id_to_class.keys())
                continue

        return sample_id

    #------------------------------------
    # _cull_samples 
    #-------------------

    def _cull_samples(self, 
                     sample_id_to_class,
                     sample_id_to_path, 
                     percentage):
        '''
        Given a dict mapping sample ids (sid) to 
        class ids (cid), and another dict mapping sids
        to image paths, and a percentage P, return copies
        of the dicts with random sids removed. Only 
        P% of the samples in each class will be included.

        Example inputs
          sample_id_to_class = {100 : 0, 101 : 4, 102: 4, 103 : 0}
          sample_id_to_path  = {100 : '/foo/img0, 
                                101 : /foo/img1, 
                                102 : /foo/img2, 
                                103 : /foo/img3
                                }

        :param sample_id_to_class: map sample id to class id
        :type sample_id_to_class: {int : int}
        :param sample_id_to_path: map sample id to image path
        :type sample_id_to_path: {int : str}
        :param percentage: percentage of samples to retain
            of each class
        :type percentage: {int | float}
        :return: copy of input dir, with (100-P)% of samples
            removed within each class
        '''
        
        reduced_sample_id_to_class = {}
        reduced_sample_id_to_path  = {}
        
        samples_in_classes = self._sids_by_class(sample_id_to_class)
        for cid, sids in samples_in_classes.items():
            random.shuffle(sids)
            num_sids_wanted = round(percentage * len(sids) / 100)
            sids_retained = random.sample(sids, num_sids_wanted)
            for sid in sids_retained:
                reduced_sample_id_to_class[sid] = cid
                reduced_sample_id_to_path[sid]  = sample_id_to_path[sid]

        return reduced_sample_id_to_class, reduced_sample_id_to_path

    #------------------------------------
    # _sids_by_class 
    #-------------------

    def _sids_by_class(self, sample_id_to_class):
        '''
        Given a dict mapping sample ids to class ids, 
        return a dict mapping class IDs to tuples containing
        the samples for that class.
        
        :param sample_id_to_class: map sample id to class id
        :type sample_id_to_class: {int : int}
        :return: map from class to list of sample ids 
            in that class
        :rtype: {int : [int]}
        '''
        
        sids_in_classes = OrderedDict({})
        for sid, cid in sample_id_to_class.items():
            try:
                sids_in_classes[cid].append(sid)
            except KeyError:
                sids_in_classes[cid] = [sid]
        return sids_in_classes

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
                sample_height=SAMPLE_HEIGHT,
                to_grayscale=False
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
        
        :param cls: class for which this instance will 
            manufacture an instance
        :type cls: bird_dataset.MultiRootImageDataset
        :param roots: individual, or list of paths under which
            subdirectories for classification targets are to
            be included
        :type roots: {str | [str]}
        :param sample_width: pixel width to which images are 
            to be scaled
        :type sample_height: pixel height to which images are 
            to be scaled
        :return: new instance of MultiRootImageDataset
        :rtype MultiRootImageDataset
        :raises ValueError if any of the roots is not a string.
        '''

        if type(roots) != list:
            roots = [roots]

        for root_dir in roots:
            if not os.path.isdir(root_dir):
                raise TypeError(f"Roots must be directories of target class subdirs; got {root_dir}")

        # Make a dataset from classes/samples 
        # under the first root dir: 
        self = SingleRootImageDataset(roots[0], 
                                      sample_width, 
                                      sample_height,
                                      to_grayscale=to_grayscale
                                      )
        self.sample_width  = sample_width
        self.sample_height = sample_height

        # Add to the initial dataset the classes/samples
        # under the remaining root dirs:
        for one_root in roots[1:]:
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
