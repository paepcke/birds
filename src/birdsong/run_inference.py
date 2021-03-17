#!/usr/bin/env python3
'''
Created on Mar 12, 2021

@author: paepcke
'''
import argparse
import os
from pathlib import Path
import sys

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score 
from torch import nn
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.folder import ImageFolder

from birdsong.nets import NetUtils
from birdsong.result_tallying import ResultTally, ResultCollection
from birdsong.utils.github_table_maker import GithubTableMaker
from birdsong.utils.learning_phase import LearningPhase
from birdsong.utils.tensorboard_plotter import SummaryWriterPlus
from birdsong.utils.utilities import FileUtils, CSVWriterCloseable


class Inferencer:
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 model_path, 
                 samples_path,
                 batch_size=None, 
                 labels_path=None):
        '''
        Constructor
        '''
        self.model_path = model_path
        self.samples_path = samples_path
        self.labels_path = labels_path
        
        self.IMG_EXTENSIONS = FileUtils.IMG_EXTENSIONS
        
        curr_dir          = os.path.dirname(__file__)
        model_fname       = os.path.basename(model_path)
        
        # Extract model properties
        # from the model filename:
        self.model_props  = FileUtils.parse_filename(model_fname)
        
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = self.model_props['batch_size']

        csv_dir = os.path.join(curr_dir, 'runs_raw_inferences')
        csv_file_nm = FileUtils.construct_filename(
            self.model_props,
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
                              is_valid_file=lambda file: Path(file).suffix \
                                               in self.IMG_EXTENSIONS
                              )
        self.loader = DataLoader(dataset,
                                 batch_size=self.batch_size, 
                                 shuffle=False, 
                                 drop_last=True 
                                 )
        self.num_classes = len(dataset.classes)
        self.class_names = dataset.classes
        
        self.model = NetUtils.get_net(
            self.model_props['net_name'],
            num_classes=self.num_classes,
            num_layers_to_retain=self.model_props['pretrain'],
            #*****to_grayscale=False
            to_grayscale=self.model_props['to_grayscale']
            )        
        
    #------------------------------------
    # run_inferencer 
    #-------------------
    
    def run_inference(self):
        
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(
                self.model_path,
                map_location=torch.device('cpu')
                ))

        loss_fn = nn.CrossEntropyLoss()

        result_coll = ResultCollection()
                
        self.model.eval()
        with torch.no_grad():
            for batch, targets in self.loader:
                images = FileUtils.to_device(batch, 'gpu')
                labels = FileUtils.to_device(targets, 'gpu')
                
                outputs = self.model(images)
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
        prec_weighted    = precision_score(all_labels, all_preds, average='weighted')
        prec_by_class    = precision_score(all_labels, all_preds, average=None)
        
        recall_macro     = recall_score(all_labels, all_preds, average='macro')
        recall_micro     = recall_score(all_labels, all_preds, average='micro')
        recall_weighted = recall_score(all_labels, all_preds, average='weighted')
        recall_by_class  = recall_score(all_labels, all_preds, average=None)
        
        
        f1_macro         = f1_score(all_labels, all_preds, average='macro')
        f1_micro         = f1_score(all_labels, all_preds, average='micro')
        f1_weighted      = f1_score(all_labels, all_preds, average='weighted')
        f1_by_class      = f1_score(all_labels, all_preds, average=None)

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
        
        # Make table with results: one column
        # with the model properties:
        ir_measures_skel = {'col_header' : ['precision', 'recall', 'f1'], 
                            'row_labels' : ['macro','micro','weighted'],
                            'rows'       : [[prec_macro, recall_macro, f1_macro],
                                           [prec_micro, recall_micro, f1_micro],
                                           [prec_weighted, recall_weighted, f1_weighted]
                                           ]  
                           }
        
        ir_per_class_rows = [[prec_class, recall_class, f1_class]
                            for prec_class, recall_class, f1_class
                            in zip(prec_by_class, recall_by_class, f1_by_class)
                            ]
        ir_by_class_skel =  {'col_header' : ['precision','recall', 'f1'],
                             'row_labels' : self.class_names,
                             'rows'       : ir_per_class_rows
                             }
        
        accuracy_skel = {'col_header' : ['accuracy', 'balanced_accuracy'],
                         'row_labels' : [],
                         'rows'       : [accuracy, balanced_accuracy]
                         }
        ir_measures_tbl  = GithubTableMaker.make_table(ir_measures_skel)
        ir_by_class_tbl  = GithubTableMaker.make_table(ir_by_class_skel)
        accuracy_tbl     = GithubTableMaker.make_table(accuracy_skel)
        
        self.writer.add_text('Information retrieval measures', ir_measures_tbl, global_step=0)
        self.writer.add_text('Per class measures', ir_by_class_tbl, global_step=0)
        self.writer.add_text('Accuracy', accuracy_tbl, global_step=0)

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Run inference on given model"
                                     )

    parser.add_argument('labels',
                        help='optional path to labels for use in measures calcs; default: no measures',
                        default=None)
    parser.add_argument('b', '--batch_size',
                        help='batch size to use; default: parse from model path'
                        )
    parser.add_argument('model',
                        help='path to the saved Pytorch model',
                        default=None)
    parser.add_argument('samples',
                        help='path to samples to run through model',
                        default=None)

    args = parser.parse_args();

    infer = Inferencer(
                 args.model_path, 
                 args.samples_path,
                 batch_size=args.batch_size, 
                 labels_path=args.labels_path
                 )
 
