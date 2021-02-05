'''
Created on Feb 4, 2021

@author: paepcke
'''

from configparser import ConfigParser

from birdsong.utils.dottable_config import DottableConfigParser
from birdsong.utils.dottable_map import DottableMap


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
    
    NEURAL_NET_ATTRS = [
            'min_epochs',
            'max_epochs',
            'batch_size',
            'num_folds',
            'optimizer',
            'loss_fn',
            'weighted',
            'kernel_size',
            'lr',
            'momentum',
            'seed',
            'all_procs_log'
            ]

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

        if type(conf_src) == str:
            build_in_config = ConfigParser()
            build_in_config.read(conf_src)
        else:
            build_in_config = conf_src
            
        for sec_name in build_in_config.sections():
            self[sec_name] = DottableMap(build_in_config[sec_name])
            
        self.sections = build_in_config.sections
        self.build_in_config = build_in_config

        if 'Training' not in self.sections():
            self.add_section('Training')
        if 'Parallelism' not in self.sections():
            self.add_section('Parallelism')

        self.define_nn_properties()
        
    #------------------------------------
    # define_nn_properties 
    #-------------------
    
    def define_nn_properties(self):
        
        for prop_name in self.NEURAL_NET_ATTRS:
            fset = NeuralNetConfig.__dict__[f"set_{prop_name}"]
            fget = lambda self, prop_name: self[prop_name]
            self.__dict__[prop_name] = property(fget, fset, f"A neural net property: {prop_name}")
        
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
        self.build_in_config.add_section(sec_name)

    #------------------------------------
    # __setattr__
    #-------------------

    def __setattr__(self, param_name, new_value):
        '''
        
        Called whenever 
        
          <NeuralNetConfig-obj>.prop_name = value
          
        is executed. Method checks whether 
        the property is one of the special neural net
        related parameter names. If so, the fset
        method of the property with the same name
        is invoked. Else, defer to parent.
        
        @param param_name: name of property
        @type param_name: str
        @param new_value: new value to set
        @type new_value: Any
        '''
        if param_name in NeuralNetConfig.NEURAL_NET_ATTRS:
            prop = self.__dict__[param_name]
            prop.fset(self, new_value)
            #******** Need a reader; just the default;
            #         Then continue building configs in
            #         launch
            prop.fget(self)
        else:
            super().__setattr__(param_name, new_value)

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


