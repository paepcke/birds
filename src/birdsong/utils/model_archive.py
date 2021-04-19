'''
Created on Mar 7, 2021

@author: paepcke
'''
from collections import deque
import os

from logging_service.logging_service import LoggingService
import torch

from birdsong.nets import NetUtils
from birdsong.utils.utilities import FileUtils


class ModelArchive:
    '''
    classdocs
    '''

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
        Constructor:
        
        :param config: configuration structure
        :type config: NeuralNetConfig
        :param num_classes: number of target classes
        :type num_classes: int
        :param history_len: number of model snapshots to 
            maintain
        :type history_len: int
        :param model_root: path to where models
            will be deposited
        :type model_root: str
        :param log: logging service to use. If
            None, create new one for display output
        :type log: LoggingService
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
        
        self.run_subdir = self._construct_run_subdir(config, 
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
        Saves and retains trained models
        on disk. 
        
        Within a subdir the method maintains a queue
        of files of len history_len: 
        
                 fname_1_ep_0.pth
                 fname_2_ep_1.pth
                      ...
                 fname_<history_len>.pth
        
        where ep_<n> is the epoch during training
        where the model of that moment is being 
        saved.
        
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
                 
        :param model: model to save
        :type model: nn.module
        :param epoch: the epoch that created the model
        :type epoch: int
        :param history_len: number of snapshot to retain
        :type history_len: int
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
        
        # As recommended by pytorch, save the
        # state_dict for portability:
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
    
    def restore_model(self, model_path, config=None):
        '''
        Given the path to a saved model, 
        load and return it. The saved file
        is the saved model's state_dict. 
        So, the method must first create a
        model instance of the correct type.
        Then the state is loaded into that
        instance.
        
        :param model_path:
        :type model_path:
        :param config: a config structure that will be
            use to decide which model class to instantiate.
            If None, attempts to reconstruct the 
            information from the model_path.
        :type config: NeuralNetConfig
        :return: loaded model
        :rtype: torch.nn.module
        '''
        
        if config is None:
            model = self._instantiate_model(config=config)
        else:
            model = self._instantiate_model(run_path_str=model_path)
         
        model.load_state_dict(torch.load(model_path))
        return model

    #------------------------------------
    # _instantiate_model 
    #-------------------
    
    def _instantiate_model(self, run_path_str=None, config=None):
        '''
        Returns a model based on information in 
        the config structure, or the info encoded
        in the run_path_str file name. 
        
        One of run_path_str or config must be non-None.
        If both are non-None, uses config.
        
        File paths that encode run parameters look like
        this horror:
        
        model_2021-03-11T10_59_02_net_resnet18_pretrain_0_lr_0.01_opt_SGD_bs_64_ks_7_folds_0_gray_True_classes_10.pth 
        
        :param run_path_str: a path name associated with
            a model. 
        :type run_path_str:
        :param config: run configuration structure 
        :type config: NeuralNetConfig
        :return: a model 
        :rtype: torch.nn.module
        '''
        if config is None:
            # Get a dict with info 
            # in a standard (horrible) file name:
            fname_props = FileUtils.parse_filename(run_path_str)
        else:
            fname_props = config.Training
            data_root   = config.Paths.root_train_test_data
            class_names = FileUtils.find_class_names(data_root)
            fname_props['classes'] = len(class_names)
            fname_props['pretrain'] = config.Training.getint('num_pretrained_layers', 0)
        
        model = NetUtils.get_net(net_name=fname_props['net_name'],
                                 num_classes=fname_props['classes'],
                                 num_layers_to_retain=fname_props['pretrain'],
                                 to_grayscale=fname_props['to_grayscale']
                                 )
        return model

# ---------------- Utils -------------

    #------------------------------------
    # _construct_run_subdir 
    #-------------------
    
    def _construct_run_subdir(self, 
                             config, 
                             num_classes, 
                             model_root):
        '''
        Constructs a directory name composed of
        elements specified in utility.py's 
        FileUtils file/config info dicts.
        
        Ensures that <model_root>/subdir_name does
        not exist. If it does, keeps adding '_r<n>'
        to the end of the dir name.
        
        Final str will look like this:
        
        model_2021-03-23T15_38_39_net_resnet18_pre_True_frz_6_bs_2_folds_5_opt_SGD_ks_7_lr_0.01_gray_False
            
        Details will depend on the passed in 
        configuration.

        Instance var fname_els_dict will contain 
        all run attr/values needed for calls to 
        FileUtils.construct_filename() 
        
        :param config: run configuration
        :type config: NeuralNetConfig
        :param num_classes: number of target classes 
        :type num_classes: int
        :param model_root: full path to dir where the
            subdir is to be created
        :type model_root: str
        :return: unique subdir name of self.model_root,
            which has been created
        :rtype: str
        '''

        # Using config, gather run-property/value 
        # pairs to include in the dir name:
         
        fname_els_dict = {}
        
        section_dict   = config.Training 
        
        for el_name, el_abbr in FileUtils.fname_long_2_short.items():
            
            el_type = FileUtils.fname_el_types[el_abbr]
            
            if el_type == int:
                fname_els_dict[el_name] = section_dict.getint(el_name)
            elif el_type == str:
                fname_els_dict[el_name] = section_dict.get(el_name)
            elif el_type == float:
                fname_els_dict[el_name] = section_dict.getfloat(el_name)
            elif el_type == bool:
                fname_els_dict[el_name] = section_dict.getboolean(el_name)
            elif callable(el_type):
                # A lambda or func. Apply it:
                fname_els_dict[el_name] = el_type(el_name)

        fname_els_dict['num_classes'] = num_classes

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
