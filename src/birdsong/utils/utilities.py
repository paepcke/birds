'''
Created on Jan 21, 2021

@author: paepcke
'''

from _collections import OrderedDict
import csv
import datetime
import glob
import os
from pathlib import Path
import re

import natsort
import torch
from torchvision import transforms

from birdsong.utils.learning_phase import LearningPhase


class FileUtils:
    '''
    classdocs
    '''
    
    str2bool = lambda the_str: the_str in ('1', 'y', 'Y', 'yes', 'Yes', 'True')
    
    # Elements that make up a filename
    fname_el_types = {'net'     : str,
                      'pre'     : str2bool,
                      'frz'     : int,
                      'lr'      : float,
                      'opt'     : str,
                      'bs'      : int,
                      'ks'      : int,
                      'folds'   : int,
                      'gray'    : str2bool,
                      'classes' : int
                      }

    # Translation of file element names
    # to long names used in config files,
    # and as object attributes: 
    fname_short_2_long = {'net'     : 'net_name',
                          'pre'     : 'pretrained',
                          'frz'     : 'freeze',
                          'lr'      : 'lr',
                          'opt'     : 'opt_name',
                          'bs'      : 'batch_size',
                          'ks'      : 'kernel_size',
                          'folds'   : 'num_folds',
                          'gray'    : 'to_grayscale',
                          'classes' : 'num_classes'
                       }

    fname_long_2_short = {long : short 
                          for short, long 
                          in fname_short_2_long.items()}

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    # Pattern to identify an iso
    # date near the start of a string.
    # Reconizes:
    #    o 'run_2021-03-11T10_59_02_net_resnet18...
    #    o 'model_2021-03-11T10_59_02_net_resnet18...
    #    o '2021-03-11T10_59_02_net_resnet18...
    #
    # Group 0 will be the prefix ('run', 'model',...),
    # Group 1 will be the timestamp.
    
    date_at_start_pat = \
        re.compile(r'([^_]*)[_]{0,1}([\d]{4}-[\d]{2}-[\d]{2}T[\d]{2}_[\d]{2}_[\d]{2})')


    #------------------------------------
    # ensure_directory_existence
    #-------------------
    
    @classmethod
    def ensure_directory_existence(cls, the_path):
        '''
        Ensures that all directories in the given
        path exist. If the_path 'looks like' a file, the 
        existence of all directories leading to (and including) 
        its parents is ensured. If the_path looks like a 
        directory, ensures that all paths (and including) 
        the_path exist.
        
        Whether or not the_path seems to be a file is 
        naively determined by whether or not the_path 
        has an extension. Not perfect, but will do.
        
        :param the_path: file or directory path whose
            directories are to be ensured
        :type the_path: str
        '''
        
        looks_like_file_path = len(Path(the_path).suffix) > 0

        if looks_like_file_path:
            the_dir = os.path.dirname(the_path)
        else:
            the_dir = the_path
            
        os.makedirs(the_dir, exist_ok=True) 

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

        :param data_root: root directory for search 
        :type data_root: str
        :return dict mapping target classes to a list
            of directories that contain samples of that
            class. The directories will be Path objs
        :rtype Ordered{str : [Path]}
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
        
        :param dir_name: root of dir to search
        :type dir_name: str
        :return: naturally sorted list of class names 
        :rtype: [str]
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
                           props_info,
                           prefix=None,
                           suffix=None, 
                           incl_date=False):
        '''
        NOTE: if changes are made to how filenames
              are constructed, check method parse_filename()
              for needed mods
        Given either:
        
            o a dict of property names and
              associated values, or
            o an object with instance vars 
              named all the long names in
              fname_short_2_long
              
        create a filename that includes all 
        the information held in the keys and 
        values of the dict/obj:

        Ex props_info is a dict:
             {
               'lr' : 0.001,
               'bs' : 32,
               optimizer : 'Adam'
             }
             
        would return the string:
            
            'lr_0.001_bs_32_optimizer_Adam'
            
        Ex props_info is an object:
        
             'net_resnet18_pre_False_lr_0.01_opt_SGD_bs_2_ks_7_folds_3_gray_True_classes_None
        
        I.e. all file elements in fname_short_2_long
        are included in the returned file name.
        
        If obj.num_classes is unavailable, the
        'classes' part of the fname will be 'None'
        
        If a prefix is provided, it will lead the
        string. Example: "Exp" would yield:
        
            'EXP_lr_0.001_bs_32_optimizer_Adam'
            
        If suffix is provided, it will be appended to the
        name: Example, suffix='.csv':
        
            'EXP_lr_0.001_bs_32_optimizer_Adam.csv'
            
        Finally, if incl_date is True, a timestamp is added
        at the start of the returned name, or right after
        the prefix
        
        :param props_info: names and values to include,
            or an object that provides all needed values
            as attributes (instance vars)
        :type props_info: {str : Any}
        :param prefix: leading part of file name
        :type prefix: str
        :param suffix: trailing part of file name
        :type suffix: str
        :param incl_date: whether or not to include current
            data in the file name
        :type incl_date: bool
        :return: a string appropriate for use as a filename
        :rtype: str
        '''
        fname = prefix if prefix is not None else ''
        if incl_date:
            fname += f"_{cls.file_timestamp()}"

        if not isinstance(props_info, dict):
            # An obj that promises attrs for each
            # needed value:
            property_dict = cls.make_run_props_dict(props_info)
        else:
            property_dict = props_info
            
        for prop_name, prop_val in property_dict.items():
            try:
                short_name = cls.fname_long_2_short[prop_name]
            except KeyError:
                # If property is not in the long2short dict,
                # we don't want it as part of the file name:
                continue
            fname += f"_{short_name}_{str(prop_val)}"

        if suffix is not None:
            fname += suffix
            
        return fname

    #------------------------------------
    # parse_filename 
    #-------------------
    
    @classmethod
    def parse_filename(cls, fname):
        '''
        Given a file name produced by 
        construct_filename(), return a dict
        with the constituent elements and their
        values. The keys are not the abbreviations
        in the filename, but the expanded names
        used in the rest of the code:
        
        Ex:
        From
          pred_2021-03-11T10_59_02_net_resnet18_pretrain_0_lr_0.01_opt_SGD_bs_64_ks_7_folds_0_gray_True_classes_10.csv
        return
          {net_name   : 'resnet18',
           batch_size : 64
              ...
           }
           
        :param fname: file name to parse
        :type fname: str
        :return: dict with the elements and their values
        :rtype: {str : {int|float|str}}
        '''
        prop_dict = {}
        
        # Remove the file extension:
        fname = str(Path(fname).stem)

        # Get ['lr',0.001,'bs',32,...]:
        fname_els = fname.split('_')
        
        # Find each of the file name elements
        # and their values in the element/val 
        # sequence
        for short_name, long_name in cls.fname_short_2_long.items():
            try:
                # Index into the list of fname elements:
                nm_idx = fname_els.index(short_name)
            except ValueError as _e:
                fname_el_val = 'na'
            else:
                # Value of element always follows
                # the element name in filenames:
                val_idx = nm_idx + 1
                try:
                    str_val = fname_els[val_idx]
                    # Convert to proper datatype:
                    fname_el_val = cls.fname_el_types[short_name](str_val)
                except IndexError as _e:
                    #raise IndexError(f"Element {short_name} in {fname} has no value for {short_name} ({long_name})")\
                    #    from e
                    fname_el_val = 'na'
                except ValueError:
                    fname_el_val = 'na'
                    
            prop_dict[long_name] = fname_el_val
            
        # Finally: if file name starts with a
        # timestamp, then elements 1,2,3, and 4 
        # comprise the date: 
        #  ['2021-03-11T10','59','02']  ==> '2021-03-11T10_59_02'
        
        match = cls.date_at_start_pat.search(fname)
        if match is not None:
            prop_dict['prefix'] = match[1]
            prop_dict['timestamp'] = match[2]
        
        return prop_dict


    #------------------------------------
    # expand_filename 
    #-------------------
    
    @classmethod
    def expand_filename(cls, fname, expand_dir=False):
        '''
        Resolves Unix filenames with tilde, environment vars,
        and/or wildcards, such as
          
                ~/tmp/foo.*
                $HOME/tmp/*

        :param fname: file name with Unix-like name features
        :type fname: str
        :param expand_dir: If true, and the fully expanded
            fname is a directory, return a list of that directory's
            content (non-recursively)
        :type expand_dir: bool
        :return: expanded file name(s), with tilde, env vars, and
            wildcards expanded.
        :rtype [str]
        '''
        
        # First, expand env vars and tilde. If 
        # fname includes a wildcard, it will still be there:
        #     e.g. ~/tmp/* ===> /home/johndow/*
        expanded_fname = os.path.expanduser(os.path.expandvars(fname))
        # Resolve any wildcards:
        globbed_names = glob.glob(expanded_fname)
        # If name had no wildcard, glob returned empty
        # list; in that case, insert the non-globbed name as
        # the 'collection' of name expansion results: 
        all_fnames = globbed_names if len(globbed_names) > 0 else [expanded_fname]

        # Do the one-level subdirectory processing:
        for maybe_dir in all_fnames.copy():
            if os.path.isdir(maybe_dir) and expand_dir:
                all_fnames.extend(glob.glob(f"{maybe_dir}/*"))
        return all_fnames


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
        
        :return: string for inclusion in filename
        :rtype: str
        '''
        # Remove the msecs part:
        # Replace colons with underscores:
        timestamp = datetime.datetime.now().isoformat()
        timestamp = re.sub(r'[.][0-9]{6}', '', timestamp)
        timestamp = timestamp.replace(':', '_')
        return timestamp

    #------------------------------------
    # load_preds_and_labels
    #-------------------
    
    @classmethod
    def load_preds_and_labels(cls, csv_path):
        '''
        Returns a ResultCollection. Each 
        ResultTally in the collection holds
        the outcomes of one epoch:
        
                created_at
                phase
                epoch
                num_classes
                batch_size
                preds
                labels
                mean_loss
                losses
        
        :param csv_path: path to CSV file with info
            from a past run
        :type csv_path: str
        '''

        # Deferred import to avoid import circularity
        # with modules that use both the result_tallying 
        # and utilities modules:

        from birdsong.result_tallying import ResultCollection, ResultTally
        
        coll = ResultCollection()
        csv_fname = os.path.basename(csv_path)
        
        if not os.path.exists(csv_path):
            raise ValueError(f"Path to csv file {csv_path} does not exist")

        # Get info encoded in the filename:
        prop_dict   = cls.parse_filename(csv_fname)
        
        num_classes = prop_dict['num_classes']
        batch_size  = prop_dict['batch_size']
        
        # Remove the above entries from
        # prop_dict, so we can pass the
        # rest into ResultTally as kwargs
        # with info beyond what ResultTally
        # requires as args:
        
        del prop_dict['num_classes']
        del prop_dict['batch_size']
        
        with open(csv_path, 'r') as fd:
            reader = csv.reader(fd)
            # Eat the header line:
            next(reader)
            for (epoch, train_preds, train_labels, val_preds, val_labels) in reader:
                
                # All elements are strings. 
                # Turn them into natives. The 
                # additional parms to eval() make
                # the eval safe by withholding
                # built-ins and any libs:
                
                train_preds_arr = eval(train_preds,
                                       {"__builtins__":None},    # No built-ins at all
                                       {}                        # No additional func
                                       )
                train_labels_arr = eval(train_labels,
                                       {"__builtins__":None},    # No built-ins at all
                                       {}                        # No additional func
                                       )
                val_preds_arr = eval(val_preds,
                                       {"__builtins__":None},    # No built-ins at all
                                       {}                        # No additional func
                                       )
                
                val_labels_arr = eval(val_labels,
                                       {"__builtins__":None},    # No built-ins at all
                                       {}                        # No additional func
                                       )

                epoch = int(epoch)
                
                train_tally = ResultTally(
                    epoch,
                    LearningPhase.TRAINING,
                    torch.tensor(train_preds_arr),
                    torch.tensor(train_labels_arr),
                    0.0,  # Placeholder for loss
                    num_classes,
                    batch_size,
                    prop_dict    # Additional, option  info
                    )            # from the file name
                
                val_tally = ResultTally(
                    epoch,
                    LearningPhase.VALIDATING,
                    torch.tensor(val_preds_arr),
                    torch.tensor(val_labels_arr),
                    0.0,  # Placeholder for loss
                    num_classes,
                    batch_size,
                    prop_dict    # Additional, option  info
                    )            # from the file name
                
                coll.add(train_tally, epoch)
                coll.add(val_tally, epoch)

        return coll
    
    #------------------------------------
    # get_image_transforms 
    #-------------------
    
    @classmethod
    def get_image_transforms(cls, 
                             sample_width=400, # Pixels 
                             sample_height=400,
                             to_grayscale=False
                             ):
        '''
        Returns a transformation that is applied
        to all images in the application. Always
        get the transforms from here.

        :param sample_width: desired resize width
        :type sample_width: int
        :param sample_height: desired resize height
        :type sample_height: int
        :param to_grayscale: whether or not to convert
            images to 1-channel grayscale during load
        :type to_grayscale: bool
        :return: transform composite
        :rtype: whatever torchvision.transforms.Compose returns
        '''
        
        img_transforms = [transforms.Resize((sample_width, sample_height)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                          ]
        if to_grayscale:
            img_transforms.append(transforms.Grayscale())
        
        transformation = transforms.Compose(img_transforms)
        
        return transformation

    #------------------------------------
    # to_device 
    #-------------------

    @classmethod
    def to_device(cls, item, device, gpu_id=None):
        '''
        Moves item to the specified device.
        device may be 'cpu', or 'gpu'
        
        :param cls:
        :type cls:
        :param item: tensor to move to device
        :type item: pytorch.Tensor
        :param device: one of 'cpu', or 'gpu'
        :type device: str
        :param gpu_id: if device is 'gpu', then 
            gpu_id is the ID of the GPU to use,
            zero-origin.
        :type gpu_id: int
        :return: the moved item
        :rtype: pytorch.Tensor
        '''
        
        fastest_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu' or fastest_device == 'cpu':
            return item.to('cpu')

        elif device != 'gpu':
            raise ValueError(f"Device must be 'cpu' or 'gpu'")

        # Will use a GPU; was a particular one specified?
        if gpu_id is None:
            gpu_id = 0
            torch.cuda.set_device(gpu_id)
        else:
            # Check device number:
            if type(gpu_id) != int:
                raise TypeError(f"GPU ID must be an int, not {gpu_id}")
            
            available_gpus = torch.cuda.device_count()
            if gpu_id > available_gpus - 1:
                raise ValueError(f"Asked to operate on GPU {gpu_id}, but only {available_gpus} are available")

        torch.cuda.set_device(gpu_id)
        return item.to(fastest_device)

        


    #------------------------------------
    # gpu_memory_report 
    #-------------------
    
    @classmethod
    def gpu_memory_report(cls, 
                          device=None, 
                          printout=False):
        
        subset_keys = [
            'allocation.all.current',
            'allocation.all.freed',
            'active_bytes.all.allocated',
            'active_bytes.all.current',
            'active_bytes.all.freed',
            'reserved_bytes.all.allocated',
            'reserved_bytes.all.current',
            'reserved_bytes.all.freed'
            ]
        if device is None:
            # Use current device
            device = torch.device
        mem_dict = torch.cuda.memory_stats()
        
        info_subset = {info_name : mem_dict[info_name]
                       for info_name
                       in subset_keys
                       }
        if printout:
            longest_info_name = len(max(subset_keys, key=len))
            
            print('***** GPU Mem Summary')
            for info_name, info_val in info_subset.items():
                num_spaces = longest_info_name - len(info_name)
                print(f"{' ' * num_spaces}{info_name}: {info_val}")
            print('*****')

        return info_subset


    #------------------------------------
    # make_run_props_dict 
    #-------------------
    
    @classmethod
    def make_run_props_dict(cls, inst):
        '''
        Given an instance of any class that
        guarantees the presence of instance
        variables named as specified in 
        the fname_short_2_long dict. 
        
        Return a dict mapping\
         
            file_element ---> value
            
        where value is the value of the instance
        var named file_element.
        
        Ex return:
            {'net'  : obj.net_name,
             'pre'  : obj.pretrained,
                ...
            }
         
        :param inst: instance from which to 
            draw values for each file element
        :type inst: Any
        :return: Dict mapping file elements 
            to values.
        :rtype: {str : Any}
        '''
        
        res = {}
        for el_name, attr_name in cls.fname_short_2_long.items():
            try:
                res[el_name] = inst.__getattr__(attr_name)
                if res[el_name] is None:
                    res[el_name] = 'na'
            except AttributeError:
                # Object does not have an instance
                # var named el_name:
                res[el_name] = 'na'
            
        return res
    
    #------------------------------------
    # ellipsed_file_path
    #-------------------
    
    @classmethod
    def ellipsed_file_path(cls, 
                           path,
                           acceptable_len=25):
        '''
        If a path is very long, return
        a string with some of the intermediate
        dirs replaced by ellipses. Used in 
        messages and log entries when the entire
        path is not needed. 
        
        :param path:
        :type path:
        :param acceptable_len: number of letters
            acceptable to keep in the result
        :type path: int
        :return shortened path for printing in messages
        :rtype: str
        
        '''
        if len(path) <= acceptable_len:
            return path
        
        # Get dirs and fname.
        components = path.split('/')
        # Remove any empty strings:
        components = list(filter(lambda el: el.__ne__(''),
                                 components
                                 ))
        
        if len(components) == 1:
            # Just a file name; we only
            # split dirs:
            return path
        
        # Is the filename all by itself,
        # without dirs, larger than acceptable?
        fname = components[-1]
        if len(fname) >= acceptable_len:
            # Just add the first dir, and
            # the ellipses; the string will
            # be longer than acceptable, but
            # it's the best we can do:
            
            #Add the leading '/' back if appropriate:
            res = f"/{components[0]}/...{fname}" \
                    if os.path.isabs(path) \
                    else f"{components[0]}/...{fname}" 
            
            return res 
        
        # Remove intermediate dirs
        # till acceptable len:
        while True:
            # Take out the second dir
            # (We always keep the root):
            components.remove(components[1])
            p = '/'.join(components)
            if len(p) <= acceptable_len:
                break
            
        midpoint = len(components) // 2
        components = components[:midpoint] + ['...'] + components[midpoint:]
        res = '/'.join(components)
        
        #Add the leading '/' back if appropriate:
        if os.path.isabs(path):
            res = f"/{res}"

        return res

    #------------------------------------
    # str2bool 
    #-------------------
    
    @classmethod
    def str2bool(cls, the_str):
        return the_str in ('1', 'y', 'Y', 'yes', 'Yes', 'True')

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

# ------------------------ Main ------------
if __name__ == '__main__':
    #print("Testing Differentiator")
    #Differentiator.test()
    #print("All good")
    pass
    
