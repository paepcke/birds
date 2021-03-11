#!/usr/bin/env python3
'''
Created on Mar 2, 2021

@author: paepcke
'''
import argparse
import datetime
from logging import DEBUG
import os
from pathlib import Path
import random
import sys

from logging_service.logging_service import LoggingService
from torch import cuda
from torch import nn
from torch import optim
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder

from birdsong.nets import NetUtils
from birdsong.result_tallying import ResultTally, ResultCollection
from birdsong.utils.dottable_config import DottableConfigParser
from birdsong.utils.file_utils import FileUtils, CSVWriterCloseable
from birdsong.utils.learning_phase import LearningPhase
from birdsong.utils.model_archive import ModelArchive
from birdsong.utils.neural_net_config import NeuralNetConfig, ConfigError
from birdsong.utils.tensorboard_plotter import SummaryWriterPlus, TensorBoardPlotter
import numpy as np

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
    # Number of intermediate models to save
    # during training:
     
    MODEL_ARCHIVE_SIZE = 20
    
    # For some tensorboard displays:
    # for how many epochs in the past
    # to display data:
     
    DISPLAY_HISTORY_LEN = 10
    
    #------------------------------------
    # Constructor 
    #-------------------


    def __init__(self, 
                 config_info, 
                 debugging=False
                 ):
        '''
        Constructor
        '''
        
        self.log = LoggingService()
        if debugging:
            self.log.logging_level = DEBUG
        
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
        self.log.debug(f"Before any gpu push: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
        
        self.to_device(self.model, 'gpu')
        
        self.log.debug(f"Before after model push: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
        
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
        self.class_names = self.train_loader.dataset.classes
        
        log_dir      = os.path.join(self.curr_dir, 'runs')
        raw_data_dir = os.path.join(self.curr_dir, 'runs_raw_results')
        
        self.setup_tensorboard(log_dir, raw_data_dir=raw_data_dir)

        # Log a few example spectrograms to tensorboard;
        # one per class:
        TensorBoardPlotter.write_img_grid(self.writer,
                                          self.root_train_test_data,
                                          len(self.class_names), # Num of train examples
                                          )

        # All ResultTally instances are
        # collected here (two per epoch, for
        # for all training loop runs, and one
        # for all val loop runs:
        
        self.epoch_results = ResultCollection()
        
        self.log.debug(f"Just before train: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
        try:
            final_epoch = self.train()
            self.visualize_final_epoch_results(final_epoch)
        finally:
            self.close_tensorboard()
        
    #------------------------------------
    # train
    #-------------------

    def train(self): 
        
        overall_start_time = datetime.datetime.now()
        for epoch in range(self.max_epochs):
            
            self.log.info(f"Starting epoch {epoch} training")
            start_time = datetime.datetime.now()
            
            # Set model to train mode:
            self.model.train()
            
            # Training
            for batch, targets in self.train_loader:

                self.log.debug(f"Top of training loop: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
                
                images = self.to_device(batch, 'gpu')
                labels = self.to_device(targets, 'gpu')
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.log.debug(f"Just before clearing gpu: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
                
                images  = self.to_device(images, 'cpu')
                outputs = self.to_device(outputs, 'cpu')
                labels  = self.to_device(labels, 'cpu')
                loss    = self.to_device(loss, 'cpu')

                self.remember_results(LearningPhase.TRAINING,
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
                
                self.log.debug(f"Just after clearing gpu: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")                

            # Validation
            
            end_time = datetime.datetime.now()
            train_time_duration = end_time - start_time
            # A human readable duration st down to minues:
            duration_str = self.time_delta_str(train_time_duration, granularity=4)
            
            self.log.info(f"Done epoch {epoch} training (duration: {duration_str})")

            self.log.debug(f"Start of validation: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
            
            start_time = datetime.datetime.now()
            self.log.info(f"Starting epoch {epoch} validation")
            
            self.model.eval()
            with torch.no_grad():
                for batch, targets in self.val_loader:
                    images = self.to_device(batch, 'gpu')
                    labels = self.to_device(targets, 'gpu')
                    
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    
                    images  = self.to_device(images, 'cpu')
                    outputs = self.to_device(outputs, 'cpu')
                    labels  = self.to_device(labels, 'cpu')
                    loss    = self.to_device(loss, 'cpu')
                    
                    self.remember_results(LearningPhase.VALIDATING,
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

            self.log.debug(f"After eval: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
            
            end_time = datetime.datetime.now()
            val_time_duration = end_time - start_time
            # A human readable duration st down to minues:
            duration_str = self.time_delta_str(val_time_duration, granularity=4)
            self.log.info(f"Done validation (duration: {duration_str})")
            
            
            epoch_duration = train_time_duration + val_time_duration
            epoch_dur_str  = self.time_delta_str(epoch_duration, granularity=4)
            
            cumulative_dur = end_time - overall_start_time
            cum_dur_str    = self.time_delta_str(cumulative_dur, granularity=4)
            
            msg = f"Done epoch {epoch}  (epoch duration: {epoch_dur_str}; cumulative: {cum_dur_str})"
            self.log.info(msg)

            # Save model, keeping self.model_archive_size models: 
            self.model_archive.save_model(self.model, epoch)

            self.scheduler.step()

            self.visualize_epoch(epoch)
            
            # Fresh results tallying 
            self.results.clear()

            # Back around to next epoch

        self.log.info(f"Training complete after {epoch + 1} epochs")
        
        # All seems to have gone well. Report the 
        # overall result of the final epoch for the 
        # hparms config used in this process:
        
        self.report_hparams_summary(self.latest_result)

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

        # Add the results 
        tally = ResultTally(epoch,
                            phase, 
                            outputs, 
                            labels, 
                            loss,
                            self.num_classes,
                            self.batch_size)
        # Add result to intermediate results collection of
        # tallies:
        self.results[epoch] = tally
        
        # Same with the session-wide
        # collection:
        
        self.epoch_results.add(tally)
        

    #------------------------------------
    # visualize_epoch 
    #-------------------
    
    def visualize_epoch(self, epoch):
        '''
        Take the ResultTally instances
        in the train and val ResultCollections
        in self.results, and report appropriate
        aggregates to tensorboard. Computes
        f1 scores, accuracies, etc. for given
        epoch.

        Separately for train and validation
        results: build one long array 
        of predictions, and a corresponding
        array of labels. Also, average the
        loss across all instances.
        
        The the preds and labels as rows to csv 
        files.
        
        @return: a ResultTally instance with all
            metrics computed for display
        @rtype: ResultTally
        '''

        val_tally   = self.results[(epoch, str(LearningPhase.VALIDATING))]
        train_tally = self.results[(epoch, str(LearningPhase.TRAINING))]

        self.latest_result = {'train': train_tally,
                              'val'  : val_tally
                              }
        
        # If we are to write preds and labels to
        # .csv for later additional processing:

        if self.csv_writer is not None:
            self.csv_writer.writerow(
                [epoch, 
                 train_tally.preds,
                 train_tally.labels,
                 val_tally.preds,
                 val_tally.labels
                 ])
            
        self.writer.add_scalar('loss/train', 
                               train_tally.mean_loss, 
                               global_step=epoch
                               )
        self.writer.add_scalar('loss/val', 
                               val_tally.mean_loss, 
                               global_step=epoch
                               )
        
        self.writer.add_scalar('balanced_accuracy_score/train', 
                               train_tally.balanced_acc, 
                               global_step=epoch
                               )

        self.writer.add_scalar('balanced_accuracy_score/val', 
                               val_tally.balanced_acc, 
                               global_step=epoch
                               )

        self.writer.add_scalar('accuracy_score/train', 
                               train_tally.accuracy, 
                               global_step=epoch
                               )
        self.writer.add_scalar('accuracy_score/val', 
                               val_tally.accuracy, 
                               global_step=epoch
                               )
        # History of learning rate adjustments:
        lr_this_epoch = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr_this_epoch,
                               global_step=epoch
                               )

        # Submit the confusion matrix image
        # to the tensorboard. In the following:
        # do not provide a separate title, such as
        #  title=f"Confusion Matrix (Validation): Epoch{epoch}"
        # That will put each matrix into its own slot
        # on tensorboard, rather than having a time slider
        
        TensorBoardPlotter.conf_matrix_to_tensorboard(
            self.writer,
            val_tally.conf_matrix,
            self.class_names,
            epoch=epoch,
            title=f"Confusion Matrix Series"
            )

        # Versions of the f1 score:
        
        self.writer.add_scalar('val_f1/macro', 
                               val_tally.f1_macro, 
                               global_step=epoch)
        self.writer.add_scalar('val_f1/micro', 
                               val_tally.f1_micro,
                               global_step=epoch)
        self.writer.add_scalar('val_f1/weighted', 
                               val_tally.f1_weighted,
                               global_step=epoch)


        # Versions of precision/recall:
        
        self.writer.add_scalar('val_prec/macro', 
                               val_tally.prec_macro, 
                               global_step=epoch)
        self.writer.add_scalar('val_prec/micro', 
                               val_tally.prec_micro,
                               global_step=epoch)
        self.writer.add_scalar('val_prec/weighted', 
                               val_tally.prec_weighted,
                               global_step=epoch)

        self.writer.add_scalar('val_recall/macro', 
                               val_tally.recall_macro, 
                               global_step=epoch)
        self.writer.add_scalar('val_recall/micro', 
                               val_tally.recall_micro,
                               global_step=epoch)
        self.writer.add_scalar('val_recall/weighted', 
                               val_tally.recall_weighted,
                               global_step=epoch)

        return val_tally

    #------------------------------------
    # visualize_final_epoch_results 
    #-------------------
    
    def visualize_final_epoch_results(self, epoch):
        '''
        Reports to tensorboard just for the
        final epoch.
 
        Expect self.latest_result to be the latest
        ResultTally.
        '''
        # DISPLAY_HISTORY_LEN holds the number
        # of historic epochs we will show. Two
        # results per epochs --> need
        # 2*DISPLAY_HISTORY_LEN results. But check
        # that there are that many, and show fewer
        # if needed:
        
        num_res_to_show = min(len(self.epoch_results),
                              2*self.DISPLAY_HISTORY_LEN)
        
        f1_hist = self.epoch_results[- num_res_to_show:]
        
        # First: the table of train and val f1-macro
        # scores for the past few epochs:
        #
        #      |phase|ep0  |ep1 |ep2 |
        #      |-----|-----|----|----|
        #      |train| f1_0|f1_1|f1_2|
        #      |  val| f1_0|f1_1|f1_2|
        
        f1_macro_tbl = TensorBoardPlotter.make_f1_train_val_table(f1_hist)
        self.writer.add_text('f1/history', f1_macro_tbl)
        
        # Now, in the same tensorboard row: the
        # per_class train/val f1 scores for each
        # class separately:
        #
        # |class|weighted mean f1 train|weighted mean f1 val| 
        # |-----|----------------------|--------------------| 							
        # |  c1 |0.1                   |0.6                 | 									
        # |  c2 |0.1                   |0.6                 | 									
        # |  c3 |0.1                   |0.6                 | 									        
        # ------|----------------------|--------------------|                                             
        
        f1_all_classes = TensorBoardPlotter.make_all_classes_f1_table(
            self.latest_result,
            self.class_names
            )
        self.writer.add_text('f1/per-class', f1_all_classes)

    #------------------------------------
    # report_hparams_summary 
    #-------------------
    
    def report_hparams_summary(self, latest_result):
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
         
           o balanced_accuracy      (train and val)
           o mean_accuracy_train    (train and val)
           o epoch_prec_weighted
           o epoch_recall_weighted
           o epoch_mean_loss        (train and val)
           
         
        @param latest_result: dict with keys 'train' and
            'val', holding the respective most recent
            (i.e. last-epoch) ResultTally
        @type latest_result: {'train' : ResultTally,
                               'val'   : ResultTally
                               }
        '''
         
        # Get the latest validation tally:
        train_tally = latest_result['train']
        val_tally   = latest_result['val']
         
        hparms_vals = {
            'pretrained_layers' : f"{self.net_name}_{self.pretrain}",
            'lr_initial': self.config.Training.lr,
            'optimizer' : self.config.Training.optimizer,
            'batch_size': self.config.getint('Training', 'batch_size'),
            'kernel_size' : self.config.getint('Training', 'kernel_size')
            }

        metric_results = {
                'zz_balanced_adj_acc_train' : train_tally.balanced_acc,
                'zz_balanced_adj_acc_val'   : val_tally.balanced_acc,
                'zz_acc_train' : train_tally.accuracy,
                'zz_acc_val'   : val_tally.accuracy,
                'zz_epoch_weighted_prec': val_tally.prec_weighted,
                'zz_epoch_weighted_recall' : val_tally.recall_weighted,
                'zz_epoch_mean_loss_train' : train_tally.mean_loss,
                'zz_epoch_mean_loss_val' : val_tally.mean_loss
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
        see create_csv_writer()
        
        @param logdir: root for tensorboard events
        @type logdir: str
        '''
        
        if not os.path.isdir(logdir):
            os.makedirs(logdir)

        # For storing train/val preds/labels
        # for every epoch. Used to create charts
        # after run is finished:
        self.csv_writer = self.create_csv_writer(raw_data_dir)
        
        # Place to store intermediate models:
        self.model_archive = self.create_model_archive(self.config, 
                                                       self.num_classes
                                                       )

        # Use SummaryWriterPlus to avoid confusing
        # directory creations when calling add_hparams()
        # on the writer:
        
        self.writer = SummaryWriterPlus(log_dir=logdir)
        
        # Intermediate storage for train and val results:
        self.results = ResultCollection()

        self.log.info(f"To view tensorboard charts: in shell: tensorboard --logdir {logdir}; then browser: localhost:6006")

    #------------------------------------
    # create_csv_writer 
    #-------------------
    
    def create_csv_writer(self, raw_data_dir):
        '''
        Create a csv_writer that will fill a csv
        file during training/validation as follows:
        
            epoch  train_preds   train_labels  val_preds  val_labels
            
        Cols after the integer 'epoch' col will each be
        an array of ints:
        
                  train_preds    train_lbls   val_preds  val_lbls
                2,"[2,5,1,2,3]","[2,6,1,2,1]","[1,2]",    "[1,3]" 
        
        If raw_data_dir is provided as a str, it is
        taken as the directory where csv file with predictions
        and labels are to be written. The dir is created if necessary.
         
        If the arg is instead set to True, a dir 'runs_raw_results' is
        created under this script's directory if it does not
        exist. Then a subdirectory is created for this run,
        using the hparam settings to build a file name. The dir
        is created if needed. Result ex.:
        
              <script_dir>
                   runs_raw_results
                       Run_lr_0.001_br_32
                           run_2021_05_ ... _lr_0.001_br_32.csv
        
        
        Then file name is created, again from the run
        hparam settings. If this file exists, user is asked whether
        to remove or append. The inst var self.csv_writer is
        initialized to:
        
           o None if csv file exists, but is not to 
             be overwritten nor appended-to
           o A filed descriptor for a file open for either
             'write' or 'append.
        
        @param raw_data_dir: If simply True, create dir and file names
            from hparams, and create as needed. If a string, it is 
            assumed to be the directory where a .csv file is to be
            created. If None, self.csv_writer is set to None.
        @type raw_data_dir: {None | True | str|
        @return: CSV writer ready for action. Set either to
            write a fresh file, or append to an existing file.
            Unless file exists, and user decided not to overwrite
        @rtype: {None | csv.writer}
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
            return None

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
        csv_file_nm = FileUtils.construct_filename(fname_elements, 
                                                   prefix='run',
                                                   suffix='.csv',
                                                   incl_date=True)
        
        csv_path = os.path.join(raw_data_root, csv_file_nm)
        
        # Get csv_raw_fd appropriately:
        
        if os.path.exists(csv_path):
            do_overwrite = FileUtils.user_confirm(f"File {csv_path} exists; overwrite?", default='N')
            if not do_overwrite:
                do_append = FileUtils.user_confirm(f"Append instead?", default='N')
                if not do_append:
                    return None
                else:
                    mode = 'a'
        else:
            mode = 'w'
            
        csv_writer = CSVWriterCloseable(csv_path, 
                                        mode=mode, 
                                        delimiter=',')

        header = ['epoch', 'train_preds', 'train_labels', 'val_preds', 'val_labels']
        csv_writer.writerow(header)

        
        return csv_writer

    #------------------------------------
    # create_model_archive 
    #-------------------
    
    def create_model_archive(self, config, num_classes):
        '''
        Creates facility for saving partially trained
        models along the way.
        
        @param config:
        @type config:
        @param num_classes:
        @type num_classes:
        @return: ModelArchive instance ready
            for calls to save_model()
        @rtype: ModelArchive
        '''
        model_archive = ModelArchive(config,
                                     num_classes,
                                     history_len=self.MODEL_ARCHIVE_SIZE,
                                     log=self.log
                                     )
        return model_archive

    #------------------------------------
    # close_tensorboard 
    #-------------------
    
    def close_tensorboard(self):
        if self.csv_writer is not None:
            try:
                self.csv_writer.close()
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
        
# ------------------------ Main ------------
if __name__ == '__main__':
        parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         description="Basic training setup."
                                         )

        parser.add_argument('-d', '--debug',
                            action='store_true',
                            help='maximally detailed debug message (default False)',
                            default=False
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
        BirdsTrainBasic(args.config, args.debug)
        print('Done')