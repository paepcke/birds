'''
Created on Mar 7, 2021

@author: paepcke
'''
from collections import deque
import os

from logging_service.logging_service import LoggingService
import torch

from birdsong.utils.file_utils import FileUtils


class ModelArchive:
    '''
    classdocs
    '''

    fname_elements = {
        'net_name' : ('Training', str, 'net'),
        'num_pretrained_layers' : ('Training', int, 'ini'),
        'lr' : ('Training', float, 'lr'),
        'optimizer' : ('Training', str, 'opt'),
        'batch_size' : ('Training', int, 'bs'),
        'kernel_size' : ('Training', int, 'ks'),
        'num_folds' : ('Training', int, 'folds')
        # a num_classes entry will be added by 
        # construct_run_subdir()
        }


    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 config, 
                 num_classes,
                 history_len=8,
                 model_root=None,
                 log=None):
        '''
        Constructor
        '''
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Model root directory:
        if model_root is None:
            self.model_root = os.path.abspath(
                os.path.join(self.curr_dir, 
                             '../runs_models')
                )
        else:
            self.model_root = model_root

        if os.path.exists(self.model_root) and \
                not os.path.isdir(self.model_root):
            raise FileExistsError(f"{self.model_root} exists but is not a directory")

        # Ensure that intermediate dirs exist:
        try:
            os.makedirs(self.model_root)
        except FileExistsError:
            pass

        if log is None:
            self.log = LoggingService()
        else:
            self.log = log
            
        self.history_len = history_len

        # Create a subdirectory of model_root
        # where this archive keeps its models.
        # The subdir is guaranteed to be unique
        # among model_root's siblings, and it will
        # be created:
        
        self.run_subdir = self.construct_run_subdir(config, 
                                                    num_classes,
                                                    self.model_root)

        # Queue to track models, keeping the 
        # number of saved models to history_len:
        
        self.model_fnames = deque(maxlen=self.history_len)
        
    #------------------------------------
    # save_model 
    #-------------------
    
    def save_model(self, model, epoch):
        '''
        Within this subdir the method maintains a queue
        of files of len history_len: 
        
                 fname_1.pth
                 fname_2.pth
                 fname_<history_len>.pth
        
        where <n> is an epoch/step number:
                 
        When history_len model files are already present, 
        removes the oldest.
        
        Assumptions: 
            o self.fname_els_dict contains prop/value
              pairs for use in FileUtils.construct_filename()
                 {'bs' : 32,
                  'lr' : 0.001,
                     ...
                 }
            o self model_fnames is a deque the size of
              which indicates how many models to save
              before discarding the oldest one as new
              ones are added
                 
        @param model: model to save
        @type model: nn.module
        @param epoch: the epoch that created the model
        @type epoch: int
        @param history_len: number of snapshot to retain
        @type history_len: int
        '''
        
        deque_len = len(self.model_fnames)
        if deque_len >= self.history_len:
            # Pushing a new model fname to the
            # front will pop the oldest from the
            # end. That file needs to be deleted:
            oldest_model_path = self.model_fnames[-1]
        else:
            # No file will need to be deleted.
            # Still filling our allotment:
            oldest_model_path = None
            
        model_fname = FileUtils.construct_filename(self.fname_els_dict,
                                                   prefix='mod', 
                                                   suffix=f"_ep{epoch}.pth", 
                                                   incl_date=True)
        
        model_path = os.path.join(self.run_subdir, model_fname)
        
        torch.save(model.state_dict(), model_path)

        self.model_fnames.appendleft(model_path)
        
        if oldest_model_path is not None:
            try:
                os.remove(oldest_model_path)
            except Exception as e:
                self.log.warn(f"Could not remove old model: {repr(e)}")


    #------------------------------------
    # restore_model 
    #-------------------
    
    def restore_model(self, model_path):
        '''
        Given the path to a saved model, 
        load and return it.
        
        @param model_path:
        @type model_path:
        @return: loaded model
        @rtype: torch.nn.module
        '''
        model = torch.load(model_path)
        return model


# ---------------- Utils -------------

    #------------------------------------
    # construct_run_subdir 
    #-------------------
    
    def construct_run_subdir(self, 
                             config, 
                             num_classes, 
                             model_root):
        '''
        Constructs a directory name composed of
        elements specified in cls.fname_elements.
        Ensures that <model_root>/subdir_name does
        not exist. If it does, keeps adding '_r<n>'
        to the end of the dir name.
        
        Final str will look like this:
        
            model_lr_0.001_bs_32_optimizer_Adam
            
        Instance var fname_els_dict will contain 
        all run attr/values needed for callse to 
        FileUtils.construct_filename() 
        
        @param config: run configuration
        @type config: NeuralNetConfig
        @param num_classes: number of target classes 
        @type num_classes: int
        @param model_root: full path to dir where the
            subdir is to be created
        @type model_root: str
        @return: unique subdir name of self.model_root,
            which has been created
        @rtype: str
        '''

        # Using config, gather run-property/value 
        # pairs to include in the dir name:
         
        fname_els_dict = {}

        for el_name, (config_sec, el_type, el_abbr) \
                in self.fname_elements.items():
            # Get sub-config dict:
            section_dict = config[config_sec]
            if el_type == int:
                fname_els_dict[el_abbr] = section_dict.getint(el_name)
            elif el_type == str:
                fname_els_dict[el_abbr] = section_dict.get(el_name)
            elif el_type == float:
                fname_els_dict[el_abbr] = section_dict.getfloat(el_name)

        fname_els_dict['classes'] = num_classes

        # Save this root name:
        self.fname_els_dict = fname_els_dict

        # Get the subdir name (without leading path):
        dir_basename = FileUtils.construct_filename(
            fname_els_dict,
            prefix='models',
            suffix=None, 
            incl_date=True)
        
        final_dir_path = os.path.join(model_root, dir_basename)
        
        # Disambiguate by appending '_r<n>' as needed: 
        disambiguation = 1
        while os.path.exists(final_dir_path):
            new_basename = f"{dir_basename}_r{disambiguation}"
            final_dir_path = os.path.join(model_root, new_basename)
            disambiguation += 1

        os.makedirs(final_dir_path)
        
        return final_dir_path 
