#!/usr/bin/env python3
'''
Created on Mar 2, 2021

@author: paepcke
'''
from _collections import OrderedDict
import argparse
import datetime
from logging import DEBUG
import os
import random
import sys

import numpy as np
import pandas as pd

from logging_service.logging_service import LoggingService
from torch import cuda, unsqueeze
from torch import nn
from torch import optim
import torch

#from ml_experiment_manager.experiment_manager import ExperimentManager
from experiment_manager.experiment_manager import ExperimentManager

from birdsong.cross_validation_dataloader import CrossValidatingDataLoader, \
    EndOfSplit
from birdsong.nets import NetUtils
from birdsong.result_tallying import ResultTally, ResultCollection
from birdsong.rooted_image_dataset import SingleRootImageDataset
from birdsong.samplers import SKFSampler
from birdsong.utils.learning_phase import LearningPhase
from birdsong.utils.model_archive import ModelArchive
from birdsong.utils.neural_net_config import NeuralNetConfig, ConfigError
from birdsong.utils.tensorboard_plotter import SummaryWriterPlus, TensorBoardPlotter
from birdsong.utils.utilities import FileUtils, CSVWriterCloseable  # , Differentiator
from data_augmentation.utils import Utils


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



class BirdsBasicTrainerCV:
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
                 device=0,
                 percentage=None,
                 debugging=False
                 ):
        '''
        
        :param config_info: all path and training parameters
        :type config_info: NeuralNetConfig
        :param debugging: output lots of debug info
        :type debugging: bool
        :param device: number of GPU to use; default is dev 0
            if any GPU is available
        :type device: {None | int}
        :param percentage: percentage of training data to 
            use
        :type percentage: {int | float}
        '''
        
        self.log = LoggingService()
        if debugging:
            self.log.logging_level = DEBUG

        if percentage is not None:
            # Integrity check:
            if type(percentage) not in [int, float]:
                raise TypeError(f"Percentage must be int or float, not {type(percentage)}")
            if percentage < 1 or percentage > 100:
                raise ValueError(f"Percentage must be between 1 and 100, not {percentage}")

        if device is None:
            device = 0
            torch.cuda.set_device(device)
        else:
            available_gpus = torch.cuda.device_count()
            if available_gpus == 0:
                self.log.info("No GPU available; running on CPU")
            else:
                if device > available_gpus - 1:
                    raise ValueError(f"Asked to operate on device {device}, but only {available_gpus} are available")
                torch.cuda.set_device(device)

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
        self.net_name   = self.config.Training.net_name
        self.pretrained = self.config.Training.getboolean('pretrained',
                                                            False)
        self.num_folds  = self.config.Training.getint('num_folds')
        self.freeze     = self.config.Training.getint('freeze', 0)
        self.to_grayscale = self.config.Training.getboolean('to_grayscale', True)
        self.save_logits  = self.config.Training.getboolean('save_logits', False)
        self.set_seed(self.config.Training.getint('seed', 42))
        
        self.log.info("Parameter summary:")
        self.log.info(f"network     {self.net_name}")
        self.log.info(f"pretrained  {self.pretrained}")
        if self.pretrained:
            self.log.info(f"freeze      {self.freeze}")
        self.log.info(f"min epochs  {self.min_epochs}")
        self.log.info(f"max epochs  {self.max_epochs}")
        self.log.info(f"batch_size  {self.batch_size}")
        self.log.info(f"save_logits {self.save_logits}")
        
        self.fastest_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = self.fastest_device
        self.num_classes = self.find_num_classes(self.root_train_test_data)

        self.initialize_model()
        
        sample_width  = self.config.getint('Training', 'sample_width', 400)
        sample_height = self.config.getint('Training', 'sample_height', 400)
        
        self.train_loader = self.get_dataloader(sample_width,
                                                sample_height,
                                                perc_data_to_use=percentage
                                                )
        self.log.info(f"Expecting {len(self.train_loader)} batches per epoch")
        num_train_samples = len(self.train_loader.dataset)
        num_classes = len(self.train_loader.dataset.class_names())
        self.log.info(f"Training set contains {num_train_samples} samples across {num_classes} classes")
        
        self.class_names = self.train_loader.dataset.class_names()
        
        exp_dir  = os.path.join(self.curr_dir, f"experiment_{FileUtils.file_timestamp()}")
        
        self.exp = ExperimentManager(exp_dir)
        self.log.info(f"Results will be under {self.exp.root}")
        
        # Save a copy of the configuration:
        self.exp.save('hparams', self.config)
        # Save the class names just as a regular
        # persistent attr/val pair in the experiment
        # manager:
        self.exp['class_names'] = self.class_names
        
        self.setup_data_logging()

        # Log a few example spectrograms to tensorboard;
        # one per class:
        TensorBoardPlotter.write_img_grid(self.writer,
                                          self.root_train_test_data,
                                          len(self.class_names), # Num of train examples
                                          )

        # All ResultTally instances are
        # collected here: (num_folds * num-epochs) 
        # each for training and validation steps.
        
        self.step_results = ResultCollection()
        
        self.log.debug(f"Just before train: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
        try:
            final_step = self.train()
            self.visualize_final_epoch_results(final_step)
        finally:
            self.close_tensorboard()
            self.log.info(f"Experiment data under {self.exp.root}")
        
    #------------------------------------
    # train
    #-------------------

    def train(self): 
        
        overall_start_time = datetime.datetime.now()
        # Just for sanity: keep track 
        # of number of batches...
        total_batch_num = 0
        
        # Note: since we are cross validating, the
        # data loader's set_epoch() method is only
        # called once (automatically) during instantiation
        # of the associated sampler. Moving from split
        # to split includes shuffling if the caller
        # specified that. 

        # Training
        for split_num in range(self.train_loader.num_folds):

            split_start_time = datetime.datetime.now()
            self.initialize_model()
            for epoch in range(self.max_epochs):

                # Set model to train mode:
                self.model.train()

                epoch_start_time = datetime.datetime.now()
        
                # Sanity check record: will record
                # how many samples from each class were
                # used:
                self.class_coverage = {}
    
                # Sanity records: will record number
                # of samples of each class that are used
                # during training and validation: 
                label_distrib = {}
                batch_num = 0
    
                self.log.info(f"Split {split_num}/{self.train_loader.num_folds}: Epoch {epoch}/{self.max_epochs} ")
                try:
                    for batch, targets in self.train_loader:
                        # Update the sanity check
                        # num of batches seen, and distribution
                        # of samples across classes: 
                        batch_num += 1
                                                                        
                        total_batch_num += 1
                        
                        # Update sanity check records:
                        for lbl in targets:
                            lbl = int(lbl)
                            try:
                                label_distrib[lbl] += 1
                            except KeyError:
                                label_distrib[lbl] = 1
                            try:
                                self.class_coverage[lbl]['train'] += 1
                            except KeyError:
                                self.class_coverage[lbl] = {'train' : 1, 'val' : 0}
    
                        self.log.debug(f"Top of training loop: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
                        
                        images = FileUtils.to_device(batch, 'gpu')
                        labels = FileUtils.to_device(targets, 'gpu')
                        
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, labels)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        
                        # Remember the last batch's train result of this
                        # split (results for earlier batches of 
                        # the same split will be overwritten). This statement
                        # must sit before deleting output and labels:
                        
                        step_num = self.step_number(epoch, split_num, self.num_folds)
                        
                        self.remember_results(LearningPhase.TRAINING,
                                              step_num,
                                              outputs,
                                              labels,
                                              loss
                                              )
    
                        self.log.debug(f"Just before clearing gpu: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
                        
                        images  = FileUtils.to_device(images, 'cpu')
                        outputs = FileUtils.to_device(outputs, 'cpu')
                        labels  = FileUtils.to_device(labels, 'cpu')
                        loss    = FileUtils.to_device(loss, 'cpu')

                        # If want to save all logits, do that
                        # for the entire batch at once, appending 
                        # column vectors for the labels and the step
                        # number:
                        if self.save_logits:
                            step_arr = np.atleast_2d(np.array([step_num]*self.batch_size,dtype=int)).T
                            logits_fld_names = self.exp.col_names('logits')
                            self.exp.save('logits',
                                          pd.DataFrame(
                                              np.hstack((outputs.detach().numpy(), 
                                                         np.atleast_2d(labels).T, 
                                                         step_arr)),
                                              columns=logits_fld_names)
                                         )
                        del images
                        del outputs
                        del labels
                        del loss
                        torch.cuda.empty_cache()
                        
                        self.log.debug(f"Just after clearing gpu: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")                
                except EndOfSplit:
                    
                    end_time = datetime.datetime.now()
                    train_time_duration = end_time - epoch_start_time
                    # A human readable duration st down to minutes:
                    duration_str = Utils.time_delta_str(train_time_duration, granularity=4)
                    
                    self.log.info(f"Done training epoch {epoch} of split {split_num} (duration: {duration_str})")
    
                    #***********
                    #print(f"****** num_batches in split: {batch_num}" )
                    #print(f"****** LblDist: {label_distrib}")
                    #***********
                    self.validate_split(step_num)
                    try:
                        self.visualize_step(step_num)
                    except Exception as e:
                        # Make sure the reporting to Tensorboard, etc.
                        # does not throw us:
                        self.log.err(f"While viz for step {step_num} {repr(e)}; continuing training")
                        
                    # Save model, keeping self.model_archive_size models:
                    self.exp.save(f"model_{epoch}", self.model) 
    
                    self.log.debug(f"After eval: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
                    
                    # Next Epoch
                    continue 
            
            # End of one epoch of one split:
            end_time = datetime.datetime.now()
            train_time_duration = end_time - split_start_time
            # A human readable duration st down to minutes:
            duration_str = Utils.time_delta_str(train_time_duration, granularity=4)
            
            self.log.info(f"Done training split {split_num} through {self.max_epochs} epochs (duration: {duration_str})")
            
            # Next split
            continue
            
        end_time       = datetime.datetime.now()
        epoch_duration = end_time - epoch_start_time
        epoch_dur_str  = Utils.time_delta_str(epoch_duration, granularity=4)
        
        cumulative_dur = end_time - overall_start_time
        cum_dur_str    = Utils.time_delta_str(cumulative_dur, granularity=4)
        
        msg = f"Done training all splits (duration: {epoch_dur_str}; cumulative: {cum_dur_str})"
        self.log.info(msg)

        #******self.scheduler.step()

        # Fresh results tallying 
        #self.results.clear()

        self.log.info(f"Training complete after {self.train_loader.num_folds} splits")
        
        # Report the sanity checks:
        self.log.info(f"Total batches processed: {total_batch_num}")
        for cid in self.class_coverage.keys(): 
            train_use, val_use = self.class_coverage[cid].items()
            self.log.info(f"{self.class_names[cid]} Training: {train_use}, Validation: {val_use}")
        
        # All seems to have gone well. Report the 
        # overall result of the final epoch for the 
        # hparms config used in this process:
        
        self.report_hparams_summary(self.latest_result)

        # The final epoch number:
        return epoch
    
    #------------------------------------
    # validate_split
    #-------------------
    
    def validate_split(self, step):
        '''
        Validate one split, using that split's 
        validation fold. Return time taken. Record
        results for tensorboard and other record keeping.
        
        :param step: current combination of epoch and 
            split
        :type step: int
        :return: number of epoch seconds needed for the validation
        :rtype: int
        '''
        # Validation
        
        self.log.debug(f"Start of validation: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
        
        start_time = datetime.datetime.now()
        self.log.info(f"Starting validation for step {step}")
        
        self.model.eval()
        with torch.no_grad():
            for img_tensor, target in self.train_loader.validation_samples():
                expanded_img_tensor = unsqueeze(img_tensor, dim=0)
                expanded_target    = unsqueeze(target, dim=0)

                # Update sanity record:
                self.class_coverage[int(target)]['val'] += 1
                
                images = FileUtils.to_device(expanded_img_tensor, 'gpu')
                label = FileUtils.to_device(expanded_target, 'gpu')
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, label)
                
                images  = FileUtils.to_device(images, 'cpu')
                outputs = FileUtils.to_device(outputs, 'cpu')
                label   = FileUtils.to_device(label, 'cpu')
                loss    = FileUtils.to_device(loss, 'cpu')
                
                self.remember_results(LearningPhase.VALIDATING,
                                      step,
                                      outputs,
                                      label,
                                      loss
                                      )
                del images
                del outputs
                del label
                del loss
                torch.cuda.empty_cache()

        end_time = datetime.datetime.now()
        val_time_duration = end_time - start_time
        # A human readable duration st down to minues:
        duration_str = Utils.time_delta_str(val_time_duration, granularity=4)
        self.log.info(f"Done validation (duration: {duration_str})")

        return val_time_duration
        
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
                         step,
                         outputs,
                         labels,
                         loss,
                         ):

        # Add the results 
        tally = ResultTally(step,
                            phase, 
                            outputs, 
                            labels, 
                            loss,
                            self.class_names,
                            self.batch_size)
        # Add result to intermediate results collection of
        # tallies:
        self.results[step] = tally
        
        # Same with the session-wide
        # collection:
        
        self.step_results.add(tally)
        

    #------------------------------------
    # visualize_step 
    #-------------------
    
    def visualize_step(self, step):
        '''
        Take the ResultTally instances
        in the train and val ResultCollections
        in self.results, and report appropriate
        aggregates to tensorboard. Computes
        f1 scores, accuracies, etc. for given
        step.

        Separately for train and validation
        results: build one long array 
        of predictions, and a corresponding
        array of labels. Also, average the
        loss across all instances.
        
        The preds and labels as rows to csv 
        files.

        '''

        val_tally   = self.results[(step, str(LearningPhase.VALIDATING))]
        train_tally = self.results[(step, str(LearningPhase.TRAINING))]
        
        result_coll = ResultCollection()
        result_coll.add(val_tally, step)
        result_coll.add(train_tally, step)

        self.latest_result = {'train': train_tally,
                              'val'  : val_tally
                              }

        # If we are to write preds and labels to
        # .csv for later additional processing:

        self.exp.save('predictions', 
                      {'step'        : step, 
                       'train_preds' : train_tally.preds,
                       'train_labels': train_tally.labels,
                       'val_preds'   : val_tally.preds,
                       'val_labels'  : val_tally.labels
                       })

        TensorBoardPlotter.visualize_step(result_coll,
                                           self.writer, 
                                           [LearningPhase.TRAINING,
                                            LearningPhase.VALIDATING
                                            ],
                                           step
                                           )

        # History of learning rate adjustments:
        lr_this_step = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr_this_step,
                               global_step=step
                               )

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
        
        num_res_to_show = min(len(self.step_results),
                              2*self.DISPLAY_HISTORY_LEN)
        
        f1_hist = self.step_results[- num_res_to_show:]
        
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
           
         
        :param latest_result: dict with keys 'train' and
            'val', holding the respective most recent
            (i.e. last-epoch) ResultTally
        :type latest_result: {'train' : ResultTally,
                               'val'   : ResultTally
                               }
        '''
         
        # Get the latest validation tally:
        train_tally = latest_result['train']
        val_tally   = latest_result['val']
         
        hparms_vals = OrderedDict({
            'net'       : self.net_name,
            'pretrained': f"{self.pretrained}",
            'lr_initial': self.config.Training.lr,
            'optimizer' : self.config.Training.opt_name,
            'batch_size':  self.config.getint('Training', 'batch_size'),
            'kernel_size'  : self.config.getint('Training', 'kernel_size'),
            'to_grayscale' : self.to_grayscale
            })

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
    # get_dataloader 
    #-------------------
    
    def get_dataloader(self, 
                       sample_width,
                       sample_height,
                       perc_data_to_use=None):
        '''
        Returns a cross validating dataloader. 
        If perc_data_to_use is None, all samples
        under self.root_train_test_data will be
        used for training. Else percentage indicates
        the percentage of those samples to use. The
        selection is random.
        
        :param sample_width: pixel width of returned images
        :type sample_width: int
        :param sample_height: pixel height of returned images
        :type sample_height: int
        :param perc_data_to_use: amount of available training
            data to use.
        :type perc_data_to_use: {None | int | float}
        :return: a data loader that serves batches of
            images and their assiated labels
        :rtype: CrossValidatingDataLoader
        '''

        data_root = self.root_train_test_data
        
        train_dataset = SingleRootImageDataset(data_root,
                                               sample_width=sample_width,
                                               sample_height=sample_height,
                                               percentage=perc_data_to_use, 
                                               to_grayscale=True)
        
        sampler = SKFSampler(
            train_dataset,
            num_folds=self.num_folds,
            seed=42,
            shuffle=True,
            drop_last=True
            )

        train_loader = CrossValidatingDataLoader(train_dataset,
                                                 batch_size=self.batch_size, 
                                                 shuffle=True, 
                                                 drop_last=True,
                                                 sampler=sampler,
                                                 num_folds=self.num_folds 
                                                 )
        return train_loader
    
    #------------------------------------
    # initialize_model 
    #-------------------
    
    def initialize_model(self):
        self.model    = NetUtils.get_net(self.net_name,
                                         num_classes=self.num_classes,
                                         pretrained=self.pretrained,
                                         freeze=self.freeze,
                                         to_grayscale=self.to_grayscale
                                         )
        self.log.debug(f"Before any gpu push: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
        
        FileUtils.to_device(self.model, 'gpu')
        
        self.log.debug(f"Before after model push: \n{'none--on CPU' if self.fastest_device.type == 'cpu' else torch.cuda.memory_summary()}")
        
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
        
        :param data_root: path to parent of train/validation
        :type data_root: str
        :return: number of unique classes as obtained
            from the directory names
        :rtype: int
        '''
        self.classes = FileUtils.find_class_names(data_root)
        return len(self.classes)

    #------------------------------------
    # setup_data_logging 
    #-------------------
    
    def setup_data_logging(self):
        '''
        Initialize tensorboard. To easily compare experiments,
        use runs/exp1, runs/exp2, etc.
        
        Additionally, sets self.experiment for logging
        predictions and logits (if requested in config file) 
        
        '''
        

        tb_path = self.exp.save(f"tensorboard_{FileUtils.file_timestamp()}")
        self.log.info(f"Tensorboard data will be written to {tb_path}")
        
        # Use SummaryWriterPlus to avoid confusing
        # directory creations when calling add_hparams()
        # on the writer:
        
        self.writer = SummaryWriterPlus(log_dir=tb_path)

        # Start experiment saving of predictions:
        header   = ['step', 'train_preds', 'train_labels', 'val_preds', 'val_labels']
        self.exp.save('predictions', header=header)
        
        if self.save_logits:
            header = self.class_names.copy()
            header.extend(['label', 'step'])
        self.exp.save('logits', header=header)

        # Intermediate storage for train and val results:
        self.results = ResultCollection()

        self.log.info(f"To view tensorboard charts: in shell: tensorboard --logdir {tb_path}; then browser: localhost:6006")

    #------------------------------------
    # create_csv_writer 
    #-------------------
    
    #*******************
    def create_csv_writerDISABLED(self, raw_data_dir):
    #*******************        
        '''
        Create a csv_writer that will fill a csv
        file during training/validation as follows:
        
            step  train_preds   train_labels  val_preds  val_labels
            
        Cols after the integer 'step' col will each be
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
        to remove or append. The first of the two return values 
        will be:
        
           o None if csv file exists, but is not to 
             be overwritten nor appended-to
           o A CSV writer for a file open for either
             'write' or 'append.
             
        The second return value is: 
        
           o None if self.save_logits is False, or a logits file already
             exists and user decided to save it.
           o A file descriptor in append mode 
        
        :param raw_data_dir: If simply True, create dir and file names
            from hparams, and create as needed. If a string, it is 
            assumed to be the directory where a .csv file is to be
            created. If None, self.csv_writer is set to None.
        :type raw_data_dir: {None | True | str|
        :return: a 2-tuple; first element: CSV writer ready for action. Set either to
            write a fresh file, or append to an existing file.
            Unless file exists, and user decided not to overwrite.
            Second element: None if self.save_logits is False, or an fd 
            in append mode, with the logits header written to the
            file.
            
        :rtype: ({None | csv.writer}, {None | fd})
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
            return None, None

        # Create both a raw dir sub-directory and a .csv file
        # for this run:
        csv_subdir_name = FileUtils.construct_filename(self.config.Training, 
                                                       prefix='Run', 
                                                       incl_date=True)
        os.makedirs(csv_subdir_name)
        
        # Create a csv file name:
        csv_file_nm = FileUtils.construct_filename(self.config.Training, 
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
                    return None, None
                else:
                    mode = 'a'
        else:
            mode = 'w'
            
        csv_writer = CSVWriterCloseable(csv_path, 
                                        mode=mode, 
                                        delimiter=',')

        header = ['step', 'train_preds', 'train_labels', 'val_preds', 'val_labels']
        csv_writer.writerow(header)

        # If we want to save all the logits to a csv file,
        # create an open fd with append mode:
        if self.save_logits:
            csv_raw_file_nm = FileUtils.construct_filename(self.config.Training, 
                                                           prefix='run_logits',
                                                           suffix='.csv',
                                                           incl_date=True)

            csv_raw_path = os.path.join(raw_data_root, csv_raw_file_nm)

            if os.path.exists(csv_raw_path):
                do_overwrite = FileUtils.user_confirm(f"File {csv_raw_path} exists; overwrite?", default='N')
                if not do_overwrite:
                    do_append = FileUtils.user_confirm(f"Append instead?", default='N')
                    if not do_append:
                        return csv_writer, None
                    else:
                        mode = 'a'
            else:
                mode = 'w'

            
            logit_fd = open(csv_raw_path, mode)
            # Header will be the list of class IDs,
            # followed by columns for each row's label
            # and step:
            header = self.class_names.copy()
            header.extend(['label', 'step'])
            logit_fd.write(f"{','.join(header)}" + '\n')
            logit_fd.flush()
        else:
            logit_fd = None
        
        return csv_writer, logit_fd

    #------------------------------------
    # create_model_archive 
    #-------------------
    
    def create_model_archive(self, config, num_classes):
        '''
        Creates facility for saving partially trained
        models along the way.
        
        :param config:
        :type config:
        :param num_classes:
        :type num_classes:
        :return: ModelArchive instance ready
            for calls to save_model()
        :rtype: ModelArchive
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
        try:
            self.exp.close()
        except Exception as e:
            self.log.warn(f"Could not close experiment instance: {repr(e)}")

        try:
            self.writer.close()
        except AttributeError:
            self.log.warn("Method close_tensorboard() called before setup_data_logging()?")
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
        
        :param config_info: the information needed to construct
            the structure
        :type config_info: {NeuralNetConfig | str}
        :return a NeuralNetConfig instance with all parms
            initialized
        :rtype NeuralNetConfig
        '''

        if isinstance(config_info, str):
            # Is it a JSON str? Should have a better test!
            if config_info.startswith('{'):
                # JSON String:
                config = NeuralNetConfig.from_json(config_info)
            else: 
                config = Utils.read_configuration(config_info)
        elif isinstance(config_info, NeuralNetConfig):
            config = config_info
        else:
            msg = f"Error: must have a config file, not {config_info}. See config.cfg.Example in project root"
            # Since logdir may be in config, need to use print here:
            print(msg)
            raise ConfigError(msg)
            
        return config


    #------------------------------------
    # set_seed  
    #-------------------

    def set_seed(self, seed):
        '''
        Set the seed across all different necessary platforms
        to allow for comparison of different models and runs
        
        :param seed: random seed to set for all random num generators
        :type seed: int
        '''
        torch.manual_seed(seed)
        cuda.manual_seed_all(seed)
        # Not totally sure what these two do!
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)

    #------------------------------------
    # step_number
    #-------------------
    
    def step_number(self, epoch, split_num, num_folds):
        '''
        Combines an epoch with a split number into 
        a single integer series as epochs increase,
        and split_num cycles from 0 to num_folds.
        
        :param epoch: epoch to encode
        :type epoch: int
        :param split_num: split number to encode
        :type split_num: int
        :param num_folds: number of folds for CV splitting
            must be contant!
        :type num_folds: int
        :return: an integer the combines epoch and split-num
        :rtype: int
        '''
        
        step_num = epoch*num_folds + split_num
        return step_num

    #------------------------------------
    # cleanup 
    #-------------------
    
    def cleanup(self):
        '''
        Recover resources taken by collaborating
        processes. OK to call multiple times.
        '''
        # self.clear_gpu()
        
        try:
            self.writer.close()
        except Exception as e:
            self.log.err(f"Could not close tensorboard writer: {repr(e)}")

# ------------------------ Main ------------
if __name__ == '__main__':
        parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         description="Basic training setup."
                                         )

        parser.add_argument('-g', '--debug',
                            action='store_true',
                            help='maximally detailed debug message (default False)',
                            default=False
                            )

        parser.add_argument('-d', '--device',
                            type=int,
                            help='gpu device to use; integer rooted at 0',
                            default=0
                            )

        parser.add_argument('-p', '--percentage',
                            type=int,
                            help='percentage of training data to use; default: all',
                            default=None
                            )
        
        parser.add_argument('config',
                            help='fully qualified path to config.cfg file',
                            )
    
        args = parser.parse_args();
        
        BirdsBasicTrainerCV(args.config, 
                            device=args.device,
                            percentage=args.percentage,
                            debugging=args.debug
                            )
        print('Done')
