'''
Created on Mar 2, 2021

@author: paepcke
'''
from _ast import arg
import argparse
import os
from pathlib import Path
import random
import sys

from logging_service.logging_service import LoggingService
from torch import cuda

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from torch import nn
from torch import optim
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder

from birdsong.nets import NetUtils
from birdsong.utils.dottable_config import DottableConfigParser
from birdsong.utils.file_utils import FileUtils
from birdsong.utils.neural_net_config import NeuralNetConfig, ConfigError
from birdsong.utils.tensorboard_plotter import SummaryWriterPlus, \
    TensorBoardPlotter
import numpy as np


class BirdsTrainBasic:
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------


    def __init__(self, config_info):
        '''
        Constructor
        '''
        
        self.log = LoggingService()
        
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            self.config = self.initialize_config_struct(config_info)
        except Exception as e:
            msg = f"During config init: {repr(e)}"
            self.log.err(msg)
            raise RuntimeError(msg) from e

        try:
            self.root_train_test_data = self.config.getpath(
                'Paths', 
                'root_train_test_data', 
                relative_to=self.curr_dir)
        except ValueError as e:
            raise ValueError("Config file must contain an entry 'root_train_test_data' in section 'Paths'") from e

        self.batch_size = self.config.getint('Training', 'batch_size')
        kernel_size     = self.config.getint('Training', 'kernel_size')
        self.min_epochs = self.config.Training.getint('min_epochs')
        self.max_epochs = self.config.Training.getint('max_epochs')

        self.set_seed(42)
        
        log_dir = os.path.join(self.curr_dir, 'runs')
        self.setup_tensorboard(log_dir)

        self.fastest_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = self.find_num_classes(self.root_train_test_data)
        
        self.model    = NetUtils.get_net('basicnet',
                                         num_classes=self.num_classes,
                                         batch_size=self.batch_size,
                                         kernel_size=kernel_size,
                                         )
        self.to_device(self.model, 'gpu')
        
        lr       = self.config.getfloat('Training', 'lr', 0.001)
        opt_name =  self.config.Training.get('optimizer', 
                                             'Adam') # Default
        self.optimizer = self.get_optimizer(
            opt_name, 
            self.model, 
            lr)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                              self.min_epochs
                                                              )
        
        sample_width  = self.config.getint('Training', 'sample_width', 400)
        sample_height = self.config.getint('Training', 'sample_height', 400)
        self.train_loader, self.val_loader = self.get_dataloaders(sample_width, 
                                                                  sample_height
                                                                  )
        try:
            self.train()
        finally:
            self.close_tensorboard()
        
    #------------------------------------
    # train
    #-------------------

    def train(self): 
        
        for epoch in range(self.max_epochs):
            # Set model to train mode:
            self.model.train()
            
            # Training
            for batch, targets in self.train_loader:

                images = self.to_device(batch, 'gpu')
                labels = self.to_device(targets, 'gpu')
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.to_device(images, 'cpu'),
                self.remember_results('train',
                                      epoch,
                                      self.to_device(outputs, 'cpu'),
                                      self.to_device(labels, 'cpu'),
                                      self.to_device(loss, 'cpu')
                                      )

            # Validation
            self.model.eval()
            for batch, targets in self.val_loader:
                images = self.to_device(batch, 'gpu')
                labels = self.to_device(targets, 'gpu')
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                self.to_device(images, 'cpu')
                self.remember_results('val',
                                      epoch,
                                      self.to_device(outputs, 'cpu'),
                                      self.to_device(labels, 'cpu'),
                                      self.to_device(loss, 'cpu')
                                      )
            self.scheduler.step()
            ##**** Do epoch-level consolidation*****
            self.visualize_epoch(epoch)
            # Back around to next epoch
        
    # ------------- Utils -----------

    #------------------------------------
    # report_acc_loss
    #-------------------
    
    def report_acc_loss(self, phase, epoch, accumulated_loss):
        
        self.writer.add_scalar(f"loss/{phase}", 
                               accumulated_loss, 
                               epoch
                               )

    #------------------------------------
    # remember_results 
    #-------------------
    
    def remember_results(self, 
                         phase,
                         epoch,
                         outputs,
                         labels,
                         loss,
                         ):

        self.results[phase].append(PredictionResult(epoch, 
                                                    outputs, 
                                                    labels, 
                                                    loss))


    #------------------------------------
    # visualize_epochs 
    #-------------------
    
    def visualize_epoch(self, epoch):
        '''
        Take the PredictionResult instances
        in self.results, and report appropriate
        aggregates to tensorboard.

        Separately for train and validation
        results: build one long array 
        of predictions, and a corresponding
        array of labels. Also, average the
        loss across all instances.
        '''
        preds      = []
        labels     = []
        loss       = torch.tensor([])
#****         summed_loss_per_sample   = 0
#         num_samples= 0
        
        for res in self.results['train']:
            # For a batch_size of 2 we output logits like:
            #
            #     tensor([[0.0162, 0.0096, 0.0925, 0.0157],
            #             [0.0208, 0.0087, 0.0922, 0.0141]], grad_fn=<AddmmBackward>)
            #
            # Turn into probabilities along each row:
            
            batch_pred_probs = torch.softmax(res.outputs, dim=1)
            
            # Now have:
            #
            #     tensor([[0.2456, 0.2439, 0.2650, 0.2454],
            #             [0.2466, 0.2436, 0.2648, 0.2449]])
            #
            # Find index of largest probability for each
            # of the batch_size prediction probs along each
            # row to get:
            #
            #  first to tensor([2,2]) then to [2,2]
            
            batch_pred_tensor = torch.argmax(batch_pred_probs, dim=1)
            
            pred_class_list = batch_pred_tensor.tolist()
            preds.extend(pred_class_list)

            labels.extend(res.labels.tolist())
            
            # Loss in PredictionResult instances
            # is per batch. Convert the number
            # to loss per sample (within each batch):
            new_loss = res.loss.detach() / self.batch_size
            loss = torch.cat((loss, torch.unsqueeze(new_loss, dim=0)))
            #*****loss.append(res.loss / self.batch_size)
            
        # Mean loss over all batches
        mean_loss = torch.mean(loss)
        self.writer.add_scalar('loss/train', 
                               mean_loss, 
                               global_step=epoch
                               )
        
        # Compute accuracy, adjust for chance, given 
        # number of classes, and shift to [-1,1] with
        # zero being chance:
         
        balanced_acc = balanced_accuracy_score(labels, 
                                               preds,
                                               adjusted=True)
        self.writer.add_scalar('balanced_accuracy_score/train', 
                               balanced_acc, 
                               epoch
                               )
        
        acc = accuracy_score(labels, preds, normalize=True)
        self.writer.add_scalar('accuracy_score/train', 
                               acc, 
                               epoch
                               )
        prec_macro   = precision_score(labels, preds, average='macro',
                                       zero_division=0)
        prec_micro   = precision_score(labels, preds, average='micro',
                                       zero_division=0)
        prec_weighted= precision_score(labels, preds, average='weighted',
                                       zero_division=0)

        recall_macro   = recall_score(labels, preds, average='macro',
                                      zero_division=0)
        recall_micro   = recall_score(labels, preds, average='micro',
                                      zero_division=0)
        recall_weighted= recall_score(labels, preds, average='weighted',
                                      zero_division=0)
        
#*****************
#         self.writer.add_scalars('prec_rec',
#                                 {'macro' : prec_macro,
#                                  'micro' : prec_micro,
#                                  'weighted' : prec_weighted
#                                  },
#                                 global_step=epoch
#                                 )

        self.writer.add_scalar('prec_rec/macro', 
                               prec_macro, 
                               global_step=epoch)
        self.writer.add_scalar('prec_rec/micro', 
                               prec_micro,
                               global_step=epoch)
        self.writer.add_scalar('prec_rec/weighted', 
                               prec_weighted,
                               global_step=epoch)

    #------------------------------------
    # get_dataloaders 
    #-------------------
    
    def get_dataloaders(self, sample_width, sample_height):
        '''
        Returns a train and a validate dataloader
        '''
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        data_root = self.root_train_test_data
        
        img_transforms = [transforms.Resize((sample_width, sample_height)),  # should actually be 1:3 but broke the system
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                          ]
        # if to_grayscale:
        #    img_transforms.append(transforms.Grayscale())
                          
        transformation = transforms.Compose(img_transforms)

        train_dataset = ImageFolder(os.path.join(data_root, 'train'),
                                    transformation,
                                    is_valid_file=lambda file: Path(file).suffix in IMG_EXTENSIONS
                                    )

        val_dataset   = ImageFolder(os.path.join(data_root, 'validation'),
                                    transformation,
                                    is_valid_file=lambda file: Path(file).suffix in IMG_EXTENSIONS
                                    )
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size, 
                                  shuffle=True, 
                                  drop_last=True 
                                  )
        
        val_loader   = DataLoader(val_dataset,
                                  batch_size=self.batch_size, 
                                  shuffle=True, 
                                  drop_last=True 
                                  )

        return train_loader, val_loader

    #------------------------------------
    # find_num_classes 
    #-------------------
    
    def find_num_classes(self, data_root):
        '''
        Expect two subdirectories under data_root:
        train and validation. Underneath each are 
        further subdirectories whose names are the
        classes:
        
                train               validation
        class1 class2 class3     class1 class2 class3
          imgs   imgs   imgs       imgs   imgs   imgs
        
        No error checking to confirm this structure
        
        @param data_root: path to parent of train/validation
        @type data_root: str
        @return: number of unique classes as obtained
            from the directory names
        @rtype: int
        '''
        self.classes = FileUtils.find_class_names(data_root)
        return len(self.classes)

    #------------------------------------
    # to_device 
    #-------------------
    
    def to_device(self, item, device):
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
        if device == 'cpu':
            return item.to('cpu')
        elif device == 'gpu':
            # May still be CPU if no gpu available:
            return item.to(self.fastest_device)
        else:
            raise ValueError(f"Device must be 'cpu' or 'gpu'")

    #------------------------------------
    # setup_tensorboard 
    #-------------------
    
    def setup_tensorboard(self, logdir):
        '''
        Initialize tensorboard. To easily compare experiments,
        use runs/exp1, runs/exp2, etc.
        
        Method creates the dir if needed.
        
        @param logdir: root for tensorboard events
        @type logdir: str
        '''
        
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        
        # Use SummaryWriterPlus to avoid confusing
        # directory creations when calling add_hparams()
        # on the writer:
        
        self.writer = SummaryWriterPlus(log_dir=logdir)
        
        # Tensorboard image writing:
        self.tensorboard_plotter = TensorBoardPlotter()
    
        # Intermediate storage for train and val results:
        self.results = {'train' : [],
                        'val'   : []
                        }
        
        
        self.log.info(f"To view tensorboard charts: in shell: tensorboard --logdir {logdir}; then browser: localhost:6006")

    #------------------------------------
    # close_tensorboard 
    #-------------------
    
    def close_tensorboard(self):
        try:
            self.writer.close()
        except AttributeError:
            self.log.warn("Method close_tensorboard() called before setup_tensorboard()?")
        except Exception as e:
            raise RuntimeError(f"Problem closing tensorboard: {repr(e)}") from e

    #------------------------------------
    # get_optimizer 
    #-------------------
    
    def get_optimizer(self, 
                      optimizer_name, 
                      model, 
                      lr):
        
        optimizer_name = optimizer_name.lower()
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=lr,
                                   eps=1e-3,
                                   amsgrad=True
                                   )
            return optimizer
    
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), 
                                  lr=lr,
                                  momentum=0.9)
            return optimizer
    
        if optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), 
                                      lr=lr,
                                      momentum=0.9)
            return optimizer

        raise ValueError(f"Optimizer {optimizer_name} not supported")
    
    #------------------------------------
    # initialize_config_struct 
    #-------------------
    
    def initialize_config_struct(self, config_info):
        '''
        Initialize a config dict of dict with
        the application's configurations. Sections
        will be:
        
          config['Paths']       -> dict[attr : val]
          config['Training']    -> dict[attr : val]
          config['Parallelism'] -> dict[attr : val]
        
        The config read method will handle config_info
        being None. 
        
        If config_info is a string, it is assumed either 
        to be a file containing the configuration, or
        a JSON string that defines the config.
         
        Else config_info is assumed to be a NeuralNetConfig.
        The latter is relevant only if using this file
        as a library, rather than a command line tool.
        
        If given a NeuralNetConfig instance, it is returned
        unchanged. 
        
        @param config_info: the information needed to construct
            the structure
        @type config_info: {NeuralNetConfig | str}
        @return a NeuralNetConfig instance with all parms
            initialized
        @rtype NeuralNetConfig
        '''

        if isinstance(config_info, str):
            # Is it a JSON str? Should have a better test!
            if config_info.startswith('{'):
                # JSON String:
                config = NeuralNetConfig.from_json(config_info)
            else: 
                config = self.read_configuration(config_info)
        elif isinstance(config_info, NeuralNetConfig):
            config = config_info
        else:
            msg = f"Error: must have a config file, not {config_info}. See config.cfg.Example in project root"
            # Since logdir may be in config, need to use print here:
            print(msg)
            raise ConfigError(msg)
            
        return config

    #------------------------------------
    # read_configuration 
    #-------------------
    
    def read_configuration(self, conf_file):
        '''
        Parses config file that describes training parameters,
        various file paths, and how many GPUs different machines have.
        Syntax follows Python's configfile package, which includes
        sections, and attr/val pairs in each section.
        
        Expected sections:

           o Paths: various file paths for the application
           o Training: holds batch sizes, number of epochs, etc.
           o Parallelism: holds number of GPUs on different machines
        
        For Parallelism, expect entries like:
        
           foo.bar.com  = 4
           127.0.0.1    = 5
           localhost    = 3
           172.12.145.1 = 6
           
        Method identifies which of the entries is
        'localhost' by comparing against local hostname.
        Though 'localhost' or '127.0.0.1' may be provided.
        
        Returns a dict of dicts: 
            config[section-names][attr-names-within-section]
            
        Types of standard entries, such as epochs, batch_size,
        etc. are coerced, so that, e.g. config['Training']['epochs']
        will be an int. Clients may add non-standard entries.
        For those the client must convert values from string
        (the type in which values are stored by default) to the
        required type. This can be done the usual way: int(...),
        or using one of the configparser's retrieval methods
        getboolean(), getint(), and getfloat():
        
            config['Training'].getfloat('learning_rate')
        
        @param other_gpu_config_file: path to configuration file
        @type other_gpu_config_file: str
        @return: a dict of dicts mirroring the config file sections/entries
        @rtype: dict[dict]
        @raises ValueErr
        @raises TypeError
        '''
        
        if conf_file is None:
            return self.init_defaults()
        
        config = DottableConfigParser(conf_file)
        
        if len(config.sections()) == 0:
            # Config file exists, but empty:
            return(self.init_defaults(config))
    
        # Do type conversion also in other entries that 
        # are standard:
        
        types = {'epochs' : int,
                 'batch_size' : int,
                 'kernel_size' : int,
                 'sample_width' : int,
                 'sample_height' : int,
                 'seed' : int,
                 'pytorch_comm_port' : int,
                 'num_pretrained_layers' : int,
                 
                 'root_train_test_data': str,
                 'net_name' : str,
                 }
        for section in config.sections():
            for attr_name in config[section].keys():
                try:
                    str_val = config[section][attr_name]
                    required_type = types[attr_name]
                    config[section][attr_name] = required_type(str_val)
                except KeyError:
                    # Current attribute is not standard;
                    # users of the corresponding value need
                    # to do their own type conversion when
                    # accessing this configuration entry:
                    continue
                except TypeError:
                    raise ValueError(f"Config file error: {section}.{attr_name} should be convertible to {required_type}")
    
        return config

    #------------------------------------
    # set_seed  
    #-------------------

    def set_seed(self, seed):
        '''
        Set the seed across all different necessary platforms
        to allow for comparison of different models and runs
        
        @param seed: random seed to set for all random num generators
        @type seed: int
        '''
        torch.manual_seed(seed)
        cuda.manual_seed_all(seed)
        # Not totally sure what these two do!
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed = seed

# ------------------------- Class PredictionResult -----------

class PredictionResult:
    
    def __init__(self, epoch, outputs, labels, loss):
        self.epoch = epoch
        self.outputs = outputs
        self.labels = labels
        self.loss = loss


# ------------------------ Main ------------
if __name__ == '__main__':
        parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         description="Basic training setup."
                                         )
    
        parser.add_argument('-c', '--config',
                            help='fully qualified path to config.cfg file',
                            )
    
        args = parser.parse_args();
        
        #*************
        args.config = os.path.join(os.path.dirname(__file__),
                                   '../../config.cfg'
                                   )
        #*************
        BirdsTrainBasic(args.config)
        print('Done')