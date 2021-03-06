#!/usr/bin/env python3
'''
Created on Mar 2, 2021

@author: paepcke
'''
import argparse
import datetime
import os
from pathlib import Path
import random
import sys

from logging_service.logging_service import LoggingService
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch import cuda
from torch import nn
from torch import optim
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder

from birdsong.nets import NetUtils
from birdsong.result_tallying import EpochSummary
from birdsong.utils.dottable_config import DottableConfigParser
from birdsong.utils.file_utils import FileUtils, CSVWriterCloseable
from birdsong.utils.neural_net_config import NeuralNetConfig, ConfigError
from birdsong.utils.tensorboard_plotter import SummaryWriterPlus, TensorBoardPlotter
import numpy as np


#from birdsong.result_tallying import TrainResult, TrainResultCollection, EpochSummary
#*****************
#
# import socket
# if socket.gethostname() in ('quintus', 'quatro', 'sparky'):
#     # Point to where the pydev server 
#     # software is installed on the remote
#     # machine:
#     sys.path.append(os.path.expandvars("$HOME/Software/Eclipse/PyDevRemote/pysrc"))
#        
#     import pydevd
#     global pydevd
#     # Uncomment the following if you
#     # want to break right on entry of
#     # this module. But you can instead just
#     # set normal Eclipse breakpoints:
#     #*************
#     print("About to call settrace()")
#     #*************
#     pydevd.settrace('localhost', port=4040)
# **************** 
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
        self.kernel_size= self.config.getint('Training', 'kernel_size')
        self.min_epochs = self.config.Training.getint('min_epochs')
        self.max_epochs = self.config.Training.getint('max_epochs')
        self.lr         = self.config.Training.getfloat('lr')
        self.net_name = self.config.Training.net_name
        self.pretrain = self.config.Training.getint('num_pretrained_layers')

        self.set_seed(42)
        
        self.log.info("Parameter summary:")
        self.log.info(f"network     {self.net_name}")
        self.log.info(f"pretrain    {self.pretrain}")
        self.log.info(f"min epochs  {self.min_epochs}")
        self.log.info(f"max epochs  {self.max_epochs}")
        self.log.info(f"batch_size  {self.batch_size}")
        
        self.fastest_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = self.find_num_classes(self.root_train_test_data)

        self.model    = NetUtils.get_net(self.net_name,
                                         num_classes=self.num_classes,
                                         num_layers_to_retain=self.pretrain,
                                         to_grayscale=False
                                         )
        self.to_device(self.model, 'gpu')
        
        # No cross validation:
        self.folds    = 0
        self.opt_name =  self.config.Training.get('optimizer', 
                                             'Adam') # Default
        self.optimizer = self.get_optimizer(
            self.opt_name, 
            self.model, 
            self.lr)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                              self.min_epochs
                                                              )
        
        sample_width  = self.config.getint('Training', 'sample_width', 400)
        sample_height = self.config.getint('Training', 'sample_height', 400)
        self.train_loader, self.val_loader = self.get_dataloaders(sample_width, 
                                                                  sample_height
                                                                  )
        log_dir      = os.path.join(self.curr_dir, 'runs')
        raw_data_dir = os.path.join(self.curr_dir, 'runs_raw_results')
        self.setup_tensorboard(log_dir, raw_data_dir=raw_data_dir)
        
        try:
            final_epoch = self.train()
            self.visualize_final_epoch_results(final_epoch)
        finally:
            self.close_tensorboard()
        
    #------------------------------------
    # train
    #-------------------

    def train(self): 
        
        for epoch in range(self.max_epochs):
            
            self.log.info(f"Starting epoch {epoch} training")
            start_time = datetime.datetime.now()
            
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
                
                outputs = self.to_device(outputs, 'cpu')
                labels  = self.to_device(labels, 'cpu')
                loss    = self.to_device(loss, 'cpu')

                self.remember_results('train',
                                      epoch,
                                      outputs,
                                      labels,
                                      loss
                                      )

                del images
                del outputs
                del labels
                del loss
                torch.cuda.empty_cache()

            # Validation
            end_time = datetime.datetime.now()
            train_time_duration = end_time - start_time
            # A human readable duration st down to minues:
            duration_str = self.time_delta_str(train_time_duration, granularity=4)
            
            self.log.info(f"Done epoch {epoch} training (duration: {duration_str})")
            
            start_time = datetime.datetime.now()
            self.log.info(f"Starting epoch {epoch} validation")
            
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
            end_time = datetime.datetime.now()
            val_time_duration = end_time - start_time
            # A human readable duration st down to minues:
            duration_str = self.time_delta_str(val_time_duration, granularity=4)
            self.log.info(f"Done validation (duration: {duration_str})")
            
            
            total_duration = train_time_duration + val_time_duration
            duration_str = self.time_delta_str(total_duration, granularity=4)
            
            self.log.info(f"Done epoch {epoch}  (total duration: {duration_str})")

            self.scheduler.step()
            ##**** Do epoch-level consolidation*****
            self.visualize_epoch(epoch)
            
            # Fresh results tallying 
            self.results['train'] = []
            self.results['val']   = []
            # Back around to next epoch

        self.log.info(f"Training complete after {epoch + 1} epochs")
        # The final epoch number:
        return epoch
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
        in self.results, plus the label/preds
        csv files and report appropriate
        aggregates to tensorboard.

        Separately for train and validation
        results: build one long array 
        of predictions, and a corresponding
        array of labels. Also, average the
        loss across all instances.
        '''
        train_preds      = []
        train_labels     = []
        val_preds        = []
        val_labels       = []

        train_loss       = torch.tensor([])
        val_loss         = torch.tensor([])
#****         summed_loss_per_sample   = 0
#         num_samples= 0
        
        for train_res, val_res in zip(self.results['train'],
                                      self.results['val']
                                      ):
            # For a batch_size of 2 we output logits like:
            #
            #     tensor([[0.0162, 0.0096, 0.0925, 0.0157],
            #             [0.0208, 0.0087, 0.0922, 0.0141]], grad_fn=<AddmmBackward>)
            #
            # Turn into probabilities along each row:
            
            batch_train_pred_probs = torch.softmax(train_res.outputs, dim=1)
            batch_val_pred_probs   = torch.softmax(val_res.outputs, dim=1)
            
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
            
            batch_train_pred_tensor = torch.argmax(batch_train_pred_probs, dim=1)
            batch_val_pred_tensor   = torch.argmax(batch_val_pred_probs, dim=1)
            
            train_pred_class_list = batch_train_pred_tensor.tolist()
            val_pred_class_list   = batch_val_pred_tensor.tolist()
             
            train_preds.extend(train_pred_class_list)
            val_preds.extend(val_pred_class_list)

            train_labels.extend(train_res.labels.tolist())
            val_labels.extend(val_res.labels.tolist())
            
            # Loss in PredictionResult instances
            # is per batch. Convert the number
            # to loss per sample (within each batch):
            train_new_loss = train_res.loss.detach() / self.batch_size
            train_loss = torch.cat((train_loss, torch.unsqueeze(train_new_loss, dim=0)))
            
            val_new_loss = train_res.loss.detach() / self.batch_size
            val_loss     = torch.cat((val_loss, torch.unsqueeze(val_new_loss, dim=0)))
            
            #*****loss.append(res.loss / self.batch_size)

        # Now we have two long sequences: predicted classes
        # (resolved from logits into actual class IDs), and
        # another seq with the corresponding correct labels.
        
        # If we are to write preds and labels to
        # .csv for later additional processing:

        if self.csv_writers is not None:
            self.csv_writers['preds'].writerow(val_preds)
            self.csv_writers['labels'].writerow(val_labels)
            
        
        # Save these lates results in case
        # this is the final epoch, and we'll
        # want results only on it. We'll overwrite
        # if there is another epoch after this one:
        
        self.latest_results = {'train_preds' : train_preds,
                               'train_labels': train_labels,
                               'val_preds'   : val_preds,
                               'val_labels'  : val_labels
                               }

        # Mean loss over all batches
        train_mean_loss = torch.mean(train_loss)
        val_mean_loss   = torch.mean(val_loss)
        self.writer.add_scalar('loss/train', 
                               train_mean_loss, 
                               global_step=epoch
                               )
        self.writer.add_scalar('loss/val', 
                               val_mean_loss, 
                               global_step=epoch
                               )
        
        # Compute accuracy, adjust for chance, given 
        # number of classes, and shift to [-1,1] with
        # zero being chance:
         
        train_balanced_acc = balanced_accuracy_score(train_labels, 
                                                     train_preds,
                                                     adjusted=True)
        self.writer.add_scalar('balanced_accuracy_score/train', 
                               train_balanced_acc, 
                               epoch
                               )

        val_balanced_acc = balanced_accuracy_score(val_labels, 
                                                   val_preds,
                                                   adjusted=True)
        self.writer.add_scalar('balanced_accuracy_score/val', 
                               val_balanced_acc, 
                               epoch
                               )

        
        train_acc = accuracy_score(train_labels, train_preds, normalize=True)
        self.writer.add_scalar('accuracy_score/train', 
                               train_acc, 
                               epoch
                               )
        val_acc = accuracy_score(val_labels, val_preds, normalize=True)
        self.writer.add_scalar('accuracy_score/val', 
                               val_acc, 
                               epoch
                               )

        # The following metrics are only 
        # reported for validation set:
        
        f1_macro    = f1_score(val_labels, val_preds, average='macro',
                               zero_division=0
                               )
        f1_micro   = precision_score(val_labels, val_preds, average='micro',
                                     zero_division=0
                                     )
        f1_weighted  = f1_score(val_labels, val_preds, average='weighted',
                                zero_division=0
                                )
        
        prec_macro   = precision_score(val_labels, val_preds, average='macro',
                                       zero_division=0
                                       )
        prec_micro   = precision_score(val_labels, val_preds, average='micro',
                                       zero_division=0
                                       )
        prec_weighted= precision_score(val_labels, val_preds, average='weighted',
                                       zero_division=0
                                       )

        recall_macro   = recall_score(val_labels, val_preds, average='macro',
                                      zero_division=0
                                      )
        recall_micro   = recall_score(val_labels, val_preds, average='micro',
                                      zero_division=0
                                      )
        recall_weighted= recall_score(val_labels, val_preds, average='weighted',
                                      zero_division=0
                                      )

        # Versions of the f1 score:
        
        self.writer.add_scalar('val_f1/macro', 
                               f1_macro, 
                               global_step=epoch)
        self.writer.add_scalar('val_f1/micro', 
                               f1_micro,
                               global_step=epoch)
        self.writer.add_scalar('val_f1/weighted', 
                               f1_weighted,
                               global_step=epoch)


        # Versions of precision/recall:
        
        self.writer.add_scalar('val_prec/macro', 
                               prec_macro, 
                               global_step=epoch)
        self.writer.add_scalar('val_prec/micro', 
                               prec_micro,
                               global_step=epoch)
        self.writer.add_scalar('val_prec/weighted', 
                               prec_weighted,
                               global_step=epoch)

        self.writer.add_scalar('val_recall/macro', 
                               recall_macro, 
                               global_step=epoch)
        self.writer.add_scalar('val_recall/micro', 
                               recall_micro,
                               global_step=epoch)
        self.writer.add_scalar('val_recall/weighted', 
                               recall_weighted,
                               global_step=epoch)


    #------------------------------------
    # visualize_final_epoch_results 
    #-------------------
    
    def visualize_final_epoch_results(self, epoch):
        '''
        Reports to tensorboard just for the
        final epoch.
            self.latest_results is {'train_preds' : sequence of class predictions
                                    'train_labels : corresponding labels
                                    'val_preds'   : sequence of class predictions
                                    'val_labels   : corresponding labels
                                    }
                                    
        where train_preds and val_preds are the processed,
        final class IDs, not the raw logits or probabilities
        '''
        train_preds  = self.latest_results['train_preds']
        train_labels = self.latest_results['train_labels']
        
        val_preds    = self.latest_results['val_preds']
        val_labels   = self.latest_results['val_labels']
        
        # First: the table of f1 scores:
        
        train_f1_macro = f1_score(train_labels,
                                        train_preds,
                                        average='macro'
                                        ).round(1)
        val_f1_macro   = f1_score(val_labels,
                                        val_preds,
                                        average='macro'
                                        ).round(1)
        
        train_f1_per_class = f1_score(train_labels,
                                            train_preds,
                                            average=None # get f1 separately for each class
                                            ).round(1) 
        val_f1_per_class = f1_score(train_labels,
                                          train_preds,
                                          average=None # get f1 separately for each class
                                          ).round(1)
        
        # Doesn't matter whether the class name
        # to int id comes from the train_loader, 
        # or the val_loader:
        classes_to_ids = self.train_loader.dataset.class_to_idx
        
        # Get reverse lookup: class_id --> class_name:
        ids_to_classes = {class_id : class_name
                          for class_name, class_id
                           in list(classes_to_ids.items()) 
                          }
        
        # Build table: class ID => class name:
        class_key = (f"Class ID|Class Name  \n"
                      "--------|------      \n"
                      )
        for cname, cidx in classes_to_ids.items():
            class_key += f"{cidx}|{cname}  \n"
            
        self.writer.add_text('class_key', 
                             class_key,
                             global_step=epoch)
        
        
        # Build:
        # |phase |f1_macro        |
        # |------|----------------|
        # |Train |{train_f1_macro}|
        # |Val   |{val_f1_macro}  |

        tbl =  (f"|phase|f1_macro|  \n"
                f"|------|-------| \n"
                f"|Train|{train_f1_macro}|  \n"
                f"|Val  |{val_f1_macro}  |  \n"
                )
        
        self.writer.add_text('f1/macro', tbl, epoch)
               
        # Build f1 for each class separately:
        
        per_class_tbl = (f"|Class|F1 train|F1 validation|  \n"
                         f"|-----|--------|-------------|  \n"
                         )
        
        for class_id, (train_f1, val_f1) in enumerate(zip(train_f1_per_class, 
                                                          val_f1_per_class
                                                          )):
            class_nm = ids_to_classes[class_id] 
            per_class_tbl += f"|{class_nm}|{train_f1}|{val_f1}|  \n"
            
        self.writer.add_text('f1/macro', per_class_tbl, epoch)

    #------------------------------------
    # report_hparams_summary 
    #-------------------
    
    def report_hparams_summary(self, 
                               tally_coll, 
                               epoch,
                               network_name,
                               num_pretrained_layers
                               ):
        '''
        Called at the end of training. Constructs
        a summary to report for the hyperparameters
        used in this process. Reports to the tensorboard.
        
        Hyperparameters reported:
        
           o lr
           o optimizer
           o batch_size
           o kernel_size
        
        Included in the measures are:
        
           o balanced_adj_accuracy_score_train
           o balanced_adj_accuracy_score_val
           o mean_accuracy_train
           o mean_accuracy_val
           o epoch_mean_weighted_precision
           o epoch_mean_weighted_recall
           o epoch_loss_train
           o epoch_loss_val
        
        @param tally_coll: the collection of results for
            each epoch
        @type tally_coll: TrainResultCollection
        '''
        
        summary = EpochSummary(tally_coll, epoch, logger=self.log)
        hparms_vals = {
            'pretrained_layers' : f"{network_name}_{num_pretrained_layers}",
            'lr_initial': self.config.Training.lr,
            'optimizer' : self.config.Training.optimizer,
            'batch_size': self.config.Training.batch_size,
            'kernel_size' : self.config.Training.kernel_size
            }
        
        # Epoch loss summaries may be 
        # unavailable, because not even enough
        # data for a single batch was available.
        # Put in -1 as a warning; it's unlikely
        # we could send nan to tensorboard:
        try:
            loss_train = summary.epoch_loss_train
        except AttributeError:
            loss_train = -1.0
            
        try:
            loss_val = summary.epoch_loss_val
        except AttributeError:
            loss_val = -1.0
        
        
        metric_results = {
                'zz_balanced_adj_accuracy_score_train' : summary.balanced_adj_accuracy_score_train,
                'zz_balanced_adj_accuracy_score_val':summary.balanced_adj_accuracy_score_val,
                'zz_mean_accuracy_train' : summary.mean_accuracy_train,
                'zz_mean_accuracy_val': summary.mean_accuracy_val,
                'zz_epoch_mean_weighted_precision': summary.epoch_mean_weighted_precision,
                'zz_epoch_mean_weighted_recall' : summary.epoch_mean_weighted_recall,
                'zz_epoch_loss_train' : loss_train,
                'zz_epoch_loss_val' : loss_val
                }
        
        self.writer.add_hparams(hparms_vals, metric_results)

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
    
    def setup_tensorboard(self, logdir, raw_data_dir=True):
        '''
        Initialize tensorboard. To easily compare experiments,
        use runs/exp1, runs/exp2, etc.
        
        Method creates the dir if needed.
        
        Additionally, sets self.csv_pred_writer and self.csv_label_writer
        to None, or open CSV writers, depending on the value of raw_data_dir,
        see set_csv_writers()
        
        @param logdir: root for tensorboard events
        @type logdir: str
        '''
        
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
            
        self.set_csv_writers(raw_data_dir)

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
        # For just the most recent result:
        self.latest_results = {'train': None,
                               'val'  : None
                               }
        
        
        self.log.info(f"To view tensorboard charts: in shell: tensorboard --logdir {logdir}; then browser: localhost:6006")

    #------------------------------------
    # set_csv_writers 
    #-------------------
    
    def set_csv_writers(self, raw_data_dir):
        '''
        If raw_data_dir is provided as a str, it is
        taken as the directory where csv files with predictions
        and labels are to be written. The dir is created if necessary.
         
        If the arg is instead set to True, a dir 'runs_raw_results' is
        created under this script's directory if it does not
        exist. Then a subdirectory is created for this run,
        using the hparam settings to build a file name. The dir
        is created if needed. Result ex.:
        
              <script_dir>
                   runs_raw_results
                       Run_lr_0.001_br_32
                           pred_2021_05_ ... _lr_0.001_br_32.csv
                           labels_2021_05_ ... _lr_0.001_br_32.csv
                           
        
                         
        
        Then two .csv file names created, again from the run
        hparam settings. If one of those files exists, user is asked whether
        to remove or append. The inst var dict self.csv_writers is
        initialized to
           {'pred_writer'   : <csv predictions writer>,
            'labels_writer' : <csv labels writer>,
        
           o With None if csv file exists, but is not to 
             be overwritten or appended-to
           o A filed descriptor for a file open for either
             'write' or 'append.
        
        @param raw_data_dir: If simply True, create dir and file names
            from hparams, and create as needed. If a string, it is 
            assumed to be the directory where a .csv file is to be
            created. If None, self.csv_writer is set to None.
        @type raw_data_dir: {None | True | str|
        '''

        # Ensure the csv file root dir exists if
        # we'll do a csv dir and run-file below it:
        
        if type(raw_data_dir) == str:
            raw_data_root = raw_data_dir
        else:
            raw_data_root = os.path.join(self.curr_dir, 'runs_raw_results')

        if not os.path.exists(raw_data_root):
            os.mkdir(raw_data_root)

        # Can rely on raw_data_root being defined and existing:
        
        if raw_data_dir is None:
            self.csv_writers = None
            return

        # Create both a raw dir sub-directory and a .csv file
        # for this run:

        fname_elements = {'net' : self.net_name,
                          'pretrain': self.pretrain,
                          'lr' : self.lr,
                          'opt' : self.opt_name,
                          'bs'  : self.batch_size,
                          'ks'  : self.kernel_size,
                          'folds'   : 0,
                          'classes' : self.num_classes
                          }
        csv_subdir_name = FileUtils.construct_filename(fname_elements, 
                                                       prefix='Run', 
                                                       incl_date=True)
        os.makedirs(csv_subdir_name)
        
        # Create a csv file name:
        csv_preds_file_nm = FileUtils.construct_filename(fname_elements, 
                                                         prefix='pred',
                                                         suffix='.csv',
                                                         incl_date=True)
        csv_labels_file_nm = FileUtils.construct_filename(fname_elements, 
                                                          prefix='labels',
                                                          suffix='.csv',
                                                          incl_date=True)
        
        csv_preds_fn = os.path.join(raw_data_root, csv_preds_file_nm)
        csv_labels_fn = os.path.join(raw_data_root, csv_labels_file_nm)
        
        # Get csv_raw_fd appropriately:
        
        if os.path.exists(csv_preds_file_nm):
            do_overwrite = FileUtils.user_confirm(f"File {self.csv_pred_file_nm} exists; overwrite?", default='N')
            if not do_overwrite:
                do_append = FileUtils.user_confirm(f"Append instead?", default='N')
                if not do_append:
                    self.csv_writers = None
                else:
                    mode = 'a'
        else:
            mode = 'w'
            
        self.csv_writers = {
            'preds'  : CSVWriterCloseable(csv_preds_fn, mode=mode, delimiter=','),
            'labels' : CSVWriterCloseable(csv_labels_fn, mode=mode, delimiter=',')
            }

    #------------------------------------
    # close_tensorboard 
    #-------------------
    
    def close_tensorboard(self):
        if self.csv_writers is not None:
            try:
                self.csv_writers['preds'].close()
            except Exception as e:
                self.log.warn(f"Could not close csv file: {repr(e)}")
            try:
                self.csv_writers['labels'].close()
            except Exception as e:
                self.log.warn(f"Could not close csv file: {repr(e)}")
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

    #------------------------------------
    # time_delta_str 
    #-------------------
    
    def time_delta_str(self, epoch_delta, granularity=2):
        '''
        Takes the difference between two datetime times:
        
               start_time = datetime.datetime.now()
               <some time elapses>
               end_time = datetime.datetime.now()
               
               delta = end_time - start_time
               time_delta_str(delta
        
        Depending on granularity, returns a string like:
        
            Granularity:
                      1  '160.0 weeks'
                      2  '160.0 weeks, 4.0 days'
                      3  '160.0 weeks, 4.0 days, 6.0 hours'
                      4  '160.0 weeks, 4.0 days, 6.0 hours, 42.0 minutes'
                      5  '160.0 weeks, 4.0 days, 6.0 hours, 42.0 minutes, 13.0 seconds'
        
            For smaller time deltas, such as 10 seconds,
            does not include leading zero times. For
            any granularity:
            
                          '10.0 seconds'

            If duration is less than second, returns '< 1sec>'
            
        @param epoch_delta:
        @type epoch_delta:
        @param granularity:
        @type granularity:
        '''
        intervals = (
            ('weeks', 604800),  # 60 * 60 * 24 * 7
            ('days', 86400),    # 60 * 60 * 24
            ('hours', 3600),    # 60 * 60
            ('minutes', 60),
            ('seconds', 1),
            )
        secs = epoch_delta.total_seconds()
        result = []
        for name, count in intervals:
            value = secs // count
            if value:
                secs -= value * count
                if value == 1:
                    name = name.rstrip('s')
                result.append("{} {}".format(value, name))
        dur_str = ', '.join(result[:granularity])
        if len(dur_str) == 0:
            dur_str = '< 1sec>'
        return dur_str
        
        


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