'''
Created on Jan 21, 2021

@author: paepcke
'''

from _collections import OrderedDict
import csv
import datetime
import os
from pathlib import Path
import re

import natsort
import torch
from torchvision import transforms

from birdsong.utils.learning_phase import LearningPhase


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
        NOTE: if changes are made to how filenames
              are constructed, check method parse_filename()
              for needed mods
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
          pred_2021-03-11T10_59_02_net_resnet18_pretrain_0_lr_0.01_opt_SGD_bs_64_ks_7_folds_0_classes_10.csv
        return
          {net_name   : 'resnet18',
           batch_size : 64
              ...
           }
           
        @param fname: file name to parse
        @type fname: str
        @return: dict with the elements and their values
        @rtype: {str : {int|float|str}}
        '''

        fname_elements = {'net' : 'net_name',
                          'pretrain': 'pretrain',
                          'lr' : 'lr',
                          'opt' : 'opt_name',
                          'bs'  : 'batch_size',
                          'ks'  : 'kernel_size',
                          'folds'   : 'num_folds',
                          'classes' : 'num_classes'
                          }

        datatypes      = {'net' : str,
                          'pretrain': int,
                          'lr' : float,
                          'opt' : str,
                          'bs'  : int,
                          'ks'  : int,
                          'folds'   : int,
                          'classes' : int
                          }

        prop_dict = {}
        
        # Remove the file extension:
        fname = str(Path(fname).stem)

        # Get ['lr',0.001,'bs',32,...]:
        fname_els = fname.split('_')
        
        # Find each of the file name elements
        # and their values in the element/val 
        # sequence
        for fname_el in fname_elements.keys():
            # Name of element (e.g. 'bs') in the
            # rest of the code (e.g. 'batch_size')
            long_name = long_name = fname_elements[fname_el] 
            try:
                # Index into the list of fname elements:
                nm_idx = fname_els.index(fname_el)
            except ValueError as e:
                raise ValueError(f"Filename element {fname_el} ({long_name}) missing from {fname}") \
                    from e
            # Value of element always follows
            # the element name in filenames:
            val_idx = nm_idx + 1
            try:
                str_val = fname_els[val_idx]
                # Convert to proper datatype:
                fname_el_val = datatypes[fname_el](str_val)
            except IndexError as e:
                raise IndexError(f"Element {fname_el} in fname {fname} has no value for {fname_el} ({long_name})")\
                    from e
                    
            prop_dict[long_name] = fname_el_val
            
        # Finally: elements 1,2,3, and 4 
        # comprise the date: 
        #  ['2021-03-11T10',59','02']  ==> '2021-03-11T10_59_02'
        
        prop_dict['timestamp'] = '_'.join(fname_els[1:4]) 
        
        return prop_dict

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
        
        @param csv_path: path to CSV file with info
            from a past run
        @type csv_path: str
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

        @param sample_width: desired resize width
        @type sample_width: int
        @param sample_height: desired resize height
        @type sample_height: int
        @param to_grayscale: whether or not to convert
            images to 1-channel grayscale during load
        @type to_grayscale: bool
        @return: transform composite
        @rtype: whatever torchvision.transforms.Compose returns
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
    def to_device(cls, item, device):
        '''
        Moves item to the specified device.
        device may be 'cpu', or 'gpu'
        
        @param item: tensor to move to device
        @type item: pytorch.Tensor
        @param device: one of 'cpu', or 'gpu'
        @type device: str
        @return: the moved item
        @rtype: pytorch.Tensor
        '''
        fastest_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            return item.to('cpu')
        elif device == 'gpu':
            # May still be CPU if no gpu available:
            return item.to(fastest_device)
        else:
            raise ValueError(f"Device must be 'cpu' or 'gpu'")


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
        
        @param path:
        @type path:
        @param acceptable_len: number of letters
            acceptable to keep in the result
        @type path: int
        @return shortened path for printing in messages
        @rtype: str
        
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

# -------------------- Differentiator ---------

class Differentiator:
    
    #------------------------------------
    # back_differentiation 
    #-------------------
    
    @classmethod
    def back_differentiation(cls, points, h=3):
    
        # h: Points to include in each step
    
        if len(points) < h+1:
            raise ValueError(f"Number of points must be at least {h+1}")
        
        point_it = iter(points)

        # Processed points:
        arc_pts  = []
    
        # Get the first h points,
        # over which the first derivative
        # will be computed:
        
        for _i in range(h):
            arc_pts.append(next(point_it))
    
        # Result will be here:
        deriv_pts = []
        
        # The (f(i) - f(i-h))/h formula
        # computed point by point:
        
        for idx, pt_val in enumerate(point_it):
            diff = pt_val - points[idx] 
            deriv_pts.append(diff / h)
    
        return deriv_pts
    
    #------------------------------------
    # test 
    #-------------------
    
    @classmethod
    def test(cls):

        pts = [1,2,3,4,5,6]
        deriv_pts = cls.back_differentiation(pts)
        assert(deriv_pts == [1.0,1.0,1.0])
        
        pts = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
        deriv_pts = cls.back_differentiation(pts)
        assert(deriv_pts == [0.5,0.5,0.5])
        
# ------------------------ Main ------------
if __name__ == '__main__':
    print("Testing Differentiator")
    Differentiator.test()
    print("All good")
    
