'''
Created on Sep 4, 2021

@author: paepcke
'''

from skorch.callbacks import EpochScoring, TensorBoard, EarlyStopping
from skorch.classifier import NeuralNetBinaryClassifier
from skorch.dataset import CVSplit
from torch import cuda
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam,SGD
from torchvision import transforms

from birdflock.binary_dataset import BinaryDataset, BalancingStrategy
from birdsong.nets import NetUtils
from birdsong.utils.tensorboard_plotter import SummaryWriterPlus


class BinaryClassificationTrainer:
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self,
                 config,
                 focal_species,
                 device=None,
                 experiment=None,
                 transforms=None
                 ):
        '''
        The species_dirs_root contains one
        subdirectory for each species. The focal_species
        is the name of the subdirectory whose recordings
        are examples of the binary classifier's target.
        The other subdirectories contain negative example
        recordings.
        
        transforms are instances of signal processing
        filters to apply.
        
        :param config: a configuration instance read
            from a config file. Client is expected to 
            have loaded the config from the file successfully. 
        :type NeuralNetConfig
        :param focal_species: name of species to recognize
        :type focal_species: str
        :param transforms: list of filters to apply to
            every file before usage
        :type trnasfo: {None | [Filter]
        '''
        species_dirs_root = config['Paths']['root_train_test_data']
        balancing_strategy = config['Training']['balancing_strategy']
        # Replace the string from the config file
        # with an element of the BalancingStrategy enum
        # or raise an exception if str doesn't name one
        # of the enum elements:
        balancing_strategy = self._ensure_balance_strategy_available(balancing_strategy)
        
        balancing_ratio = config.getfloat('Training', 'balancing_ratio')

        batch_size = config.getint('Training', 'batch_size')
        #kernel_size= config.getint('Training', 'kernel_size')
        max_epochs = config.getint('Training', 'max_epochs')
        opt_str    = config['Training']['opt_name']
        optimizer  = self._ensure_implemented_optimizer(opt_str)
        lr         = config.getfloat('Training', 'lr')
        momentum   = config.getfloat('Training', 'momentum')
        loss_fn_nm = config['Training']['loss_fn']
        loss_fn    = self._ensure_implemented_loss_fn(loss_fn_nm)
        net_name   = config['Training']['net_name']
        pretrained = config.getboolean('Training', 'pretrained', False)
        num_folds  = config.getint('Training', 'num_folds')
        freeze     = config.getint('Training', 'freeze', 0)
        to_grayscale = config.getboolean('Training', 'to_grayscale', True)
        self.save_logits = config.getboolean('Training', 'save_logits', False)
        
        early_stop = config.getboolean('Training', 'early_stop', True)
       
        num_classes  = 1

        self.experiment = experiment
        self._prep_tensorboard()
        
        transforms = self.create_transforms()
        
        num_gpus = cuda.device_count()

        if device is None:
            device = 'cuda:0' if cuda.is_available else 'cpu'
        else:
            # Was device given as 'cpu'?
            if device in ['CPU', 'cpu']:
                device = 'cpu'
            elif type(device) == int:
                if device >= num_gpus:
                    raise ValueError(f"Machine only has {num_gpus} GPUs, yet {device} were requested")
                device = f"cuda:{device}" if cuda.is_available else 'cpu'
        
        self.model    = NetUtils.get_net(net_name,
                                 num_classes=num_classes,
                                 pretrained=pretrained,
                                 freeze=freeze,
                                 to_grayscale=to_grayscale
                                 )
        
        
        dataset  = BinaryDataset(species_dirs_root, 
                                 focal_species,
                                 balancing_strategy=balancing_strategy,
                                 balancing_ratio=balancing_ratio,
                                 transforms=transforms)
        cv_split = CVSplit(dataset.split_generator(num_folds, test_percentage=20))


        # Use all defaults for optimizer, loss func, etc.
        # Maybe modify:
        #    o device <--- 'cuda'

        acc_cb = EpochScoring(scoring='accuracy', lower_is_better=False)
        bal_acc_cb = EpochScoring(scoring='balanced_accuracy', lower_is_better=False)
        f1_cb = EpochScoring(scoring='f1', lower_is_better=False)
        tensorboard_cb = TensorBoard(self.tb_writer)
        callbacks = [tensorboard_cb, bal_acc_cb, f1_cb, acc_cb]
        if early_stop:
            early_stop_cb = EarlyStopping(monitor='balanced_accuracy', patience=3, threshold=0.01)
            callbacks.append(early_stop_cb)

        classifier_kwargs = {
            'train_split'   : cv_split,
            'criterion'     : loss_fn,
            'dataset'       : dataset,
            'device'        : device,
            'callbacks'     : callbacks,
            'optimizer'     : optimizer,
            'optimizer__lr' : lr,
            'max_epochs'    : max_epochs,
            'batch_size'    : batch_size
            }
        if optimizer == SGD:
            classifier_kwargs['optimizer__momentum'] =momentum

        self.net = NeuralNetBinaryClassifier(
            self.model,
            **classifier_kwargs
            )
        
        #****************
        #***********
        #for X,y in dataset:
        #    print(f"X: {X}; y: {y}")
        #***********
        
        self.net.fit(dataset, y=None, )

    #------------------------------------
    # create_transforms
    #-------------------
    
    def create_transforms(self, 
                          to_grayscale=True,
                          sample_width=224,
                          sample_height=224
                          ):
        
        img_transforms = [
                          transforms.ToTensor(),
                          Ensure3Channels(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                          transforms.Resize((sample_width, sample_height))
                          ]

        if to_grayscale:
            img_transforms.append(transforms.Grayscale())
                          
        return transforms.Compose(img_transforms)

    #------------------------------------
    # _prep_tensorboard
    #-------------------
    
    def _prep_tensorboard(self):
        
        tb_path = self.experiment.tensorboard_path
        self.tb_writer = SummaryWriterPlus(log_dir=tb_path)

    #------------------------------------
    # _ensure_balance_strategy_available
    #-------------------
    
    def _ensure_balance_strategy_available(self, balancing_strategy_str):
        '''
        Given a string like UNDERSAMPLE or OVERSAMPLE,
        ensure that the str names a member of the 
        BalancingStrategy enum, and return the respective
        enum element for use in place of the given string.
        
        :param balancing_strategy_str: proposed balancing strategy name
        :type balancing_strategy_str: str
        :return enum element that corresponds to the balancing_strategy_str
        :rtype BalancingStrategy
        :raise NotImplementedError if given str does not match a BalancingStrategy
            element name
        '''

        for bal_strat in BalancingStrategy:
            if bal_strat.name == balancing_strategy_str:
                return bal_strat
        raise NotImplementedError(f"Balancing strategy '{balancing_strategy_str}' is not available")

    #------------------------------------
    # _ensure_implemented_optimizer
    #-------------------
    
    def _ensure_implemented_optimizer(self, opt_nm_str):
        
        if opt_nm_str == 'Adam':
            optimizer = Adam
        elif opt_nm_str == 'SGD':
            optimizer = SGD
        else:
            raise NotImplementedError(f"Optimizer {opt_nm_str} unavailable")

        return optimizer

    #------------------------------------
    # _ensure_implemented_loss_fn 
    #-------------------
    
    def _ensure_implemented_loss_fn(self, loss_fn_str):
        
        if loss_fn_str == 'BCEWithLogitsLoss':
            loss_fn = BCEWithLogitsLoss
        elif loss_fn_str == 'CrossEntropyLoss':
            loss_fn = CrossEntropyLoss
        else:
            raise NotImplementedError(f"Loss function {loss_fn_str} is not supported")
        return loss_fn


# --------------------- NeuralNetBinaryClassifierTensorBoardReady --------

# class NeuralNetBinaryClassifierTensorBoardReady(NeuralNetBinaryClassifier):
#
#     def on_epoch_end(self, mystery_arg, **kwargs):
#         pass


# -------------------- Transform Class Ensure3Channels ------

class Ensure3Channels:
    '''
    Given an image tensor, return a 3-channel tensor.
    Tensors are expected with format (channels, height, width).
    If the given tensor already has 3 channels, returns
    the tensor unchanged. If it has 1 channel, replicates that
    channel 3 times. Else: ValueError.
    '''
    
    def __call__(self, grayscale_tensor):
        
        channels, _width, _height = grayscale_tensor.shape
        if channels == 1:
            return grayscale_tensor.repeat(3,1,1)
        elif channels == 3:
            return grayscale_tensor
        else:
            raise ValueError(f"Tensor has shape {grayscale_tensor.shape}; should be (1, h, w) or (3, h, w)")

# ------------------------ Main ------------
if __name__ == '__main__':
    pass
    
    
