#!/usr/bin/env python3
'''
Created on Mar 12, 2021

@author: paepcke
'''
import os
from pathlib import Path

from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from torch import nn
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.folder import ImageFolder

from birdsong.result_tallying import ResultTally, ResultCollection
from birdsong.utils.learning_phase import LearningPhase
from birdsong.utils.tensorboard_plotter import SummaryWriterPlus
from birdsong.utils.utilities import FileUtils, CSVWriterCloseable
from birdsong.utils.utilities.FileUtils import IMG_EXTENSIONS


class Inferencer(object):
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 model_path, 
                 samples_path,
                 batch_size=64, 
                 labels_path=None):
        '''
        Constructor
        '''
        self.model_path = model_path
        self.samples_path = samples_path
        self.batch_size = batch_size
        self.labels_path = labels_path
        
        curr_dir     = os.path.dirname(__file__)
        model_file   = os.path.basename(model_path)
        model_props  = FileUtils.parse_filename(model_file)
        
        csv_dir = os.path.join(curr_dir, 'runs_raw_inferences')
        csv_file_nm = FileUtils.construct_filename(
            model_props,
            prefix='inf',
            suffix='.csv', 
            incl_date=True)
        csv_path = os.path.join(csv_dir, csv_file_nm)
        
        self.csv_writer = CSVWriterCloseable(csv_path)
        
        tensorboard_dest = os.path.join(curr_dir, 'runs_inferences')
        self.writer = SummaryWriterPlus(log_dir=tensorboard_dest)
        
        transformations = FileUtils.get_image_transforms()
        dataset = ImageFolder(samples_path,
                              transformations,
                              is_valid_file=lambda file: Path(file).suffix in IMG_EXTENSIONS
                              )
        self.loader = DataLoader(dataset,
                                 batch_size=self.batch_size, 
                                 shuffle=False, 
                                 drop_last=True 
                                 )
        self.num_classes = len(dataset.classes)
        self.class_names = dataset.classes


    #------------------------------------
    # run_inferencer 
    #-------------------
    
    def run_inference(self, model_path, samples_path):
        
        model = torch.load(model_path)
        loss_fn = nn.CrossEntropyLoss()

        result_coll = ResultCollection()
                
        model.eval()
        with torch.no_grad():
            for batch, targets in self.loader:
                images = FileUtils.to_device(batch, 'gpu')
                labels = FileUtils.to_device(targets, 'gpu')
                
                outputs = model(images)
                loss    = loss_fn(outputs, labels)
                
                images  = FileUtils.to_device(images, 'cpu')
                outputs = FileUtils.to_device(outputs, 'cpu')
                labels  = FileUtils.to_device(labels, 'cpu')
                loss    = FileUtils.to_device(loss, 'cpu')
                
                tally = ResultTally(None,
                                    LearningPhase.TESTING,
                                    outputs, 
                                    labels, 
                                    loss,
                                    self.num_classes,
                                    self.batch_size)
                result_coll.add(tally)
                del images
                del outputs
                del labels
                del loss
                torch.cuda.empty_cache()
        
        return result_coll
        
    #------------------------------------
    # report_result 
    #-------------------
    
    def report_result(self, tally_col):
        '''
        Give a sequence of tallies with results
        from a series of batches, create long
        outputs, and inputs lists from all tallies
        Then write a CSV file, and create a text
        table with the results. Report the table 
        to tensorboard if possible, and return the
        table text.
        
        @param tally_col: collect of tallies from batches
        @type tally_col: ResultCollection
        @return table of results
        @rtype: str
        '''
        
        all_preds   = []
        all_labels  = []
        
        for tally in tally_col:
            all_preds.extend(tally.outputs)
            all_labels.extend(tally.labels)
        
        prec_macro       = precision_score(all_labels, all_preds, average='macro')
        prec_micro       = precision_score(all_labels, all_preds, average='micro')
        prec_wheighted   = precision_score(all_labels, all_preds, average='weighted')
        prec_by_class    = precision_score(all_labels, all_preds, average=None)
        
        recall_macro     = recall_score(all_labels, all_preds, average='macro')
        recall_micro     = recall_score(all_labels, all_preds, average='micro')
        recall_wheighted = recall_score(all_labels, all_preds, average='weighted')
        recall_by_class  = recall_score(all_labels, all_preds, average=None)
        
        accuracy          = accuracy_score(all_labels, all_preds)
        balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
        
        conf_matrix = FileUtils.compute_confusion_matrix(
            all_labels,
            all_preds,
            self.num_classes,
            normalize=True
            )
        FileUtils.conf_matrix_to_tensorboard(
            self.writer,
            conf_matrix,
            self.class_names
            )
        
