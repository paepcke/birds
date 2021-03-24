'''
Created on Feb 4, 2021

@author: paepcke
'''

from configparser import ConfigParser
import io
import json
import os

from birdsong.utils.dottable_config import DottableConfigParser
from birdsong.utils.dottable_map import DottableMap

# ------------ Specialty Exceptions --------

class ConfigError(Exception):
    pass

# ----------- Class NeuralNetConfig ----------

class NeuralNetConfig(DottableConfigParser):
    '''
    A configuration with special knowledge
    (and expectations) of neural net training
    procedures.
    
    Adds setters for (example values just for
    guidance)
        Training.
			 net_name      = resnet18
			 min_epochs    = 15
			 max_epochs    = 100
			 batch_size    = 32
			 num_folds     = 10
			 seed          = 42
			 optimizer     = SGD
			 loss_fn       = CrossEntropyLoss
			 weighted      = True
			 kernel_size   = 7
			 lr            = 0.001
			 momentum      = 0.9
    '''
    
    NEURAL_NET_ATTRS = {
            'net_name'   : 'Training',
            'num_pretrained_layers' : 'Training',
            'min_epochs'  : 'Training',
            'max_epochs'  : 'Training',
            'batch_size'  : 'Training',
            'num_folds'   : 'Training',
            'opt_name'    : 'Training',
            'loss_fn'     : 'Training',
            'weighted'    : 'Training',
            'kernel_size' : 'Training',
            'lr'          : 'Training',
            'momentum'    : 'Training',
            'to_grayscale': 'Training',
            'seed'        : 'Parallelism',
            'all_procs_log' : 'Parallelism'
            }

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, conf_src):
        '''
        If conf_src is a file path,
        the respective file is expected to
        be a configuration. The configuration
        does not need to have the neural net
        related quantities set. They are added
        if their setters are called.
        
        Instead, conf_src may be an instance
        of the parent (DottableConfigParser).
        
        @param conf_src: file path to config file,
            or a DottableConfigParser instance
        @type conf_src: {str | DottableConfigParser}
        '''

        built_in_content, section_names = self.process_init_info(conf_src)
        
        # Copy the parameter/value pairs from the
        # config structure into this instance's
        # dict data struct. The dicts are DottableMap
        # instances to allow dot notation:
        
        for sec_name in section_names:
            self[sec_name] = DottableMap({parm_name : parm_val 
                              for parm_name,parm_val 
                              in built_in_content[sec_name].items()}
                              )

        # Guarantee presence of some sections
        # that are minimally needed for neural nets:
        
        if 'Training' not in self.sections():
            self.add_section('Training')
        if 'Parallelism' not in self.sections():
            self.add_section('Parallelism')

        # Create coinvenience properties
        # (getters and setters) for known
        # nn parameters (listed in NEURAL_NET_ATTRS:
        
        self.define_nn_properties()

    #------------------------------------
    # sections 
    #-------------------
    
    def sections(self):
        return list(self.keys())
        
    #------------------------------------
    # define_nn_properties 
    #-------------------
    
    def define_nn_properties(self):
        
        for prop_name in self.NEURAL_NET_ATTRS.keys():
            fset = NeuralNetConfig.__dict__[f"set_{prop_name}"]
            fget = self.__getattr__
            self.__dict__[prop_name] = property(fget, 
                                                fset, 
                                                f"A neural net property: {prop_name}")
        
    #------------------------------------
    # add_section 
    #-------------------
    
    def add_section(self, sec_name):
        '''
        Add a new section, initializing
        it to an empty DottableMap
        
        @param sec_name: name of new config section
        @type sec_name: str
        '''
        self[sec_name] = DottableMap({})

    #------------------------------------
    # add_neural_net_parm 
    #-------------------
    
    def add_neural_net_parm(self, parm_name, parm_val):
        
        try:
            sec_name = self.NEURAL_NET_ATTRS[parm_name]
        except KeyError:
            raise ValueError(f"Name {parm_name} is not a recognized neural net parameter")
        self[sec_name][parm_name] = parm_val

    #------------------------------------
    # to_json 
    #-------------------
    
    def to_json(self, file_info=None, check_file_exists=True):
        '''
        Convert this NeuralNetConfig structure
        to JSON, write:
        
            o to a string and return or
            o write to a provided IOStream handle
            o write to a provided file path
            
        When writing to a file, intermediate directories
        are created if necessary. 
        
        If a file path is provided, check_file_exists
        is True, and the file already exists, FileExistsError
        is raised. If the file exists, and check_file_exists
        is False, the existing file is overwritten.
        
        @param file_info: If None, return a json string.
            If a StringIO handle, fill the stream and 
            return it. Else, must be a file path where to
            write the JSON
        @type file_info: {None | str | StringIO}
        @param check_file_exists: whether or not to raise
            exception upon destination file existence
        @type check_file_exists: bool
        @return: JSON string, a StringIO filled with the
            JSON, or None if writing to a file
        @rtype: {str | StringIO | None}
        @raise FileExistsError: if file exists and check_file_exists
            is True
        '''
        
        if file_info is None:
            return json.dumps(self)
        
        if type(file_info) == io.StringIO:
            json.dump(self, file_info)
            return file_info
        else:
            # File name: check existence
            if check_file_exists and os.path.exists(file_info):
                raise FileExistsError()
            try:
                os.makedirs(os.path.dirname(file_info))
            except FileExistsError:
                pass
            with open(file_info, 'w') as fd:
                json.dump(self, fd)

        return None

    #------------------------------------
    # from_json 
    #-------------------
    
    @classmethod
    def from_json(cls, json_str):
        '''
        Given a json string that represents
        a NeuralNetConfig instance, reconstitute
        that instance and return it.
        
        @param json_str: json string
        @type json_str: str
        @return: a reconsituted NeuralNetConfig instance
        @rtype: NeuralNetConfig
        '''
        
        content_info = json.loads(json_str)
        new_inst = NeuralNetConfig(content_info)
        return new_inst

    #------------------------------------
    # json_human_readable 
    #-------------------
    
    @classmethod
    def json_human_readable(cls, json_str):
        '''
        Given a config in json format, return
        a human readable string, listing neural-net
        relevant parameters in their respective sections
        
        @param json_str: string to humanize
        @type json_str: str
        @return: printable string with vals for nn parameters
        @rtype: str
        '''
        config = json.loads(json_str)
        human_str = ''
        for sec_name in config.keys():
            first_of_sec = True
            for parm_name in config[sec_name]:
                if parm_name in cls.NEURAL_NET_ATTRS:
                    if first_of_sec:
                        human_str += f". Section {sec_name}: " \
                                if len(human_str) > 0 \
                                else f"Section {sec_name}: " 
                        first_of_sec = False
                    else:
                        human_str += '/'
                    human_str += f"{parm_name}({config[sec_name][parm_name]})"
        return human_str

    #------------------------------------
    # run_name 
    #-------------------
    
    def run_name(self):
        '''
        Create name for use in tensorboard to identify
        a run that uses the nn settings of this configuration.
        
          Exp_lr<num>bs<num>kern<num>opt<optimizer> 
        '''
        
        nm = (f"Exp_lr{self.Training.lr}"
              f"_bs{self.Training.batch_size}"
              f"_kern{self.Training.kernel_size}"
              f"_opt{self.Training.optimizer}"
              )
        return nm

    #------------------------------------
    # copy 
    #-------------------
    
    def copy(self):
        new_inst = NeuralNetConfig(self)
        return new_inst

    #------------------------------------
    # _eq_ 
    #-------------------

    def __eq__(self, other):
        '''
        Called when '==' or '!='
        are invoked. Python automatically
        takes the inverse of this method's
        result for '!='
        
        For equality, all dicts that make
        up the config sections must be equal,
        and both instances must have the same
        number of sections
        
        @param other: instance to compare against
        @type other: NeuralNetConfig
        @return: True for equality, False for not
        @rtype: bool
        '''
        
        sec_dicts_this = [self[sec_name]
                          for sec_name
                          in self.sections()
                          ]
        sec_dicts_other = [other[sec_name]
                           for sec_name
                           in other.sections()
                           ]
        
        if len(sec_dicts_this) != len(sec_dicts_other):
            return False
        
        # All the sections dicts need to 
        # be equal for the two NeuralNetConfig
        # instances to be equal:
        
        is_eq = (sum([dict_self == (dict_other)
                      for dict_self, dict_other
                      in zip(sec_dicts_this, sec_dicts_other)
                      ])) == len(sec_dicts_this)
        
        return is_eq


    #------------------------------------
    # __str__ 
    #-------------------
    
    def __str__(self):
        section_names = self.sections()
        if len(section_names) > 3:
            show_names = ','.join(section_names[:3])
        else:
            show_names = ','.join(section_names)
        the_str = f"<NeuralNetConfig ({show_names})>"
        return the_str

    #------------------------------------
    # __repr__ 
    #-------------------
    
    def __repr__(self):
        
        id_str = hex(id(self))
        return f"<NeuralNetConfig {id_str}>"

    #------------------------------------
    # __setattr__ 
    #-------------------
    
    def __setattr__(self, prop_name, new_val):
        if prop_name in self.NEURAL_NET_ATTRS.keys():
            section_dict = self[self.NEURAL_NET_ATTRS[prop_name]]
            section_dict[prop_name] = new_val
            
        else:
            super().__setattr__(prop_name, new_val)


    # ---------------- Setters ----------

    # NOTE: the proper way to implement the
    #       setters below would be via 
    #       @attribute_name.setter decorators.
    #       I used those many times before. But
    #       the setters just won't be called in 
    #       this case. Maybe it's the built-in
    #       'dict' class being in the inheritance
    #       hierarchy; no idea. So: doing the
    #       dispatching in the __setattr__() method

    # --------- Training Section Setters ----------
    
    def set_net_name(self, new_name):
        conf_dict = self.Training
        conf_dict['net_name'] = new_name

    def set_num_pretrained_layers(self, new_name):
        conf_dict = self.Training
        conf_dict['num_pretrained_layers'] = new_name

    def set_min_epochs(self, new_val):
        
        assert type(new_val) == int, 'Epochs must be ints'
        max_epochs = self.Training.getint('Training', None) 
        if max_epochs is not None:
            assert new_val <= max_epochs, 'Min epoch must be <= max epoch'
            
        conf_dict = self.Training
        conf_dict['min_epochs'] = new_val
        
    def set_max_epochs(self, new_val):

        assert type(new_val) == int, 'Epochs must be ints'
        min_epochs = self.Training.getint('Training', None) 
        if min_epochs is not None:
            assert new_val >= min_epochs, 'Max epoch must be >= max epoch'
        
        conf_dict = self.Training
        conf_dict['max_epochs'] = new_val
        
    def set_batch_size(self, new_val):
        
        assert type(new_val) == int and\
         new_val > 0, 'Batch sizes must be pos ints'
        
        conf_dict = self.Training
        conf_dict['batch_size'] = new_val

    def set_num_folds(self, new_val):
        
        assert type(new_val) == int and\
         new_val > 0, 'Number of folds must be pos int'
        
        conf_dict = self.Training
        conf_dict['num_folds'] = new_val

    def set_optimizer(self, new_val):

        # If desired, add check for specific
        # set of optimizers where it now says 'True':
        assert type(new_val) == str and\
         True, 'Optimizers must be a string'

        conf_dict = self.Training
        conf_dict['optimizer'] = new_val

    def set_loss_fn(self, new_val):
        
        # If desired, add check for specific
        # set loss functions where it now says 'True':
        assert type(new_val) == str and\
         True, 'Loss function must be a string'
        
        conf_dict = self.Training
        conf_dict['loss_fn'] = new_val

    def set_weighted(self, new_val):
        
        assert type(new_val) == bool, \
            'Weighted must be 1/0, or True/False or Yes/No'
        
        conf_dict = self.Training
        conf_dict['set_weighted'] = new_val

    def set_kernel_size(self, new_val):
        
        assert type(new_val) == int and\
         new_val > 0, 'Kernels sizes must be pos ints'
        
        conf_dict = self.Training
        conf_dict['kernel_size'] = new_val

    def set_lr(self, new_val):
        
        assert type(new_val) == float and\
         new_val < 1, 'Learning rate must be a float < 1'
        
        conf_dict = self.Training
        conf_dict['lr'] = new_val

    def set_momentum(self, new_val):
        
        assert type(new_val) == float and\
         new_val >= 0, 'Momentum must be pos float'
        
        conf_dict = self.Training
        conf_dict['momentum'] = new_val
    
    def set_to_grayscale(self, new_val):
        
        assert type(new_val) == bool, \
                'Grayscale conversion instruction must be bool'
               
        
        conf_dict = self.Training
        conf_dict['to_grayscale'] = new_val
    
    
# --------- Parallelism Section Setters ----------

    def set_seed(self, new_val):
        
        assert type(new_val) == int and\
         new_val > 0, 'Seed must be pos int'
        
        conf_dict = self.Parallelism
        conf_dict['seed'] = new_val
        
    def set_all_procs_log(self, new_val):
        
        assert type(new_val) == bool, \
            'Quantity all_procs_log must be 1/0, or True/False or Yes/No'
        
        conf_dict = self.Parallelism
        conf_dict['all_procs_log'] = new_val

# ------------------------- Utilities ----------

    def process_init_info(self, conf_src):
        '''
        Given config information in one of several
        formats, create an intermediate representation
        that the __init__() method can then turn into
        a true NeuralNetConfig instance. Input options:
           
           1 Filename, which is expected to point to
                 a valid config file
           2 A dict structure that mimics the internal
                 struct of a NeuralNetConfig instance:
                    {<section_name1> : {param_name : param_val,...}
                     <section_name2> : {param_name : param_val,...}
                     }
           3 An existing NeuralNetConfig instance
           
           4 A JSON string containing the config info
           
        Returns a structure that behaves as in 2 above,
        and a list of section names. Often those will
        be the same as the keys of the returned outer dict.
        But not always. 
        
        @param conf_src: configuration content
        @type conf_src: {str | dict | NeuralNetConfig}
        @return dict of dict with outer dict being
            section names
        @rtype: dict
        '''
        
        if type(conf_src) == str:
            # A file name
            # Use the Python ConfigParser class
            # to parse the config file:
            built_in_content = ConfigParser()
            built_in_content.read(conf_src)
            section_names = built_in_content.sections()
            
        elif type(conf_src) == dict:
            # Dict of sections, each section being
            # another dict with atomic values:
            for val in conf_src.values():
                if type(val) != dict:
                    raise TypeError("Only dict of dicts allowed for dict structure source")
                
            built_in_content = conf_src
            section_names = list(built_in_content.keys())
            
        elif type(conf_src) != NeuralNetConfig:
            raise TypeError("Only file paths, dict-of-dict, or a NeuralNetConfig instance are allowed")
        else:
            # Creating a copy from an existing NeuralNetConfig.
            built_in_content = conf_src
            section_names = built_in_content.sections()
    
        return built_in_content, section_names
