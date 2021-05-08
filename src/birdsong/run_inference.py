#!/usr/bin/env python3
'''
Created on Mar 12, 2021

@author: paepcke
'''
from _collections import OrderedDict
import argparse
import datetime
import os
from pathlib import Path
import sys

from logging_service.logging_service import LoggingService
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score 
from torch import nn
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.folder import ImageFolder

from birdsong.ml_plotting.classification_charts import ClassificationPlotter
from birdsong.nets import NetUtils
from birdsong.result_tallying import ResultTally, ResultCollection
from birdsong.utils.github_table_maker import GithubTableMaker
from birdsong.utils.learning_phase import LearningPhase
from birdsong.utils.tensorboard_plotter import SummaryWriterPlus, \
    TensorBoardPlotter
from birdsong.utils.utilities import FileUtils, CSVWriterCloseable
import pandas as pd
from result_analysis.charting import Charter


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
                 batch_size=1, 
                 labels_path=None,
                 gpu_id=0
                 ):
        '''
        Given the path to a trained model,
        and the path to the root of a set
        of data, compute predictions.
        
        If labels_path is None, the subdirectory
        names between the samples_path root,
        and the samples themselves are used as
        the ground truth labels.
        
        By default: run batches of size 1,
        because we always have drop_last set
        to True. For small test sets leaving
        out any data at all isn't good. Caller
        can still set batch_size higher to gain
        speed if the testset is very large, so that
        not inferencing on up to batch_size - 1 
        samples is OK
        
        :param model_path:
        :type model_path:
        :param samples_path:
        :type samples_path:
        :param batch_size:
        :type batch_size:
        :param labels_path:
        :type labels_path:
        :param gpu_id: Device number of GPU, in case 
            one is available
        :type gpu_id: int
        '''

        self.model_path = model_path
        self.samples_path = samples_path
        self.labels_path = labels_path
        self.gpu_id = gpu_id
        
        self.IMG_EXTENSIONS = FileUtils.IMG_EXTENSIONS
        
        self.log = LoggingService()
        
        curr_dir          = os.path.dirname(__file__)
        model_fname       = os.path.basename(model_path)
        
        # Extract model properties
        # from the model filename:
        self.model_props  = FileUtils.parse_filename(model_fname)
        
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = self.model_props['batch_size']

        self.csv_dir = os.path.join(curr_dir, 'runs_raw_inferences')
        csv_file_nm = FileUtils.construct_filename(
            self.model_props,
            prefix='inf',
            suffix='.csv', 
            incl_date=True)
        csv_path = os.path.join(self.csv_dir, csv_file_nm)
        
        self.csv_writer = CSVWriterCloseable(csv_path)
        
        tensorboard_dest = os.path.join(curr_dir, 'runs_inferences')
        self.writer = SummaryWriterPlus(log_dir=tensorboard_dest)
        
        transformations = FileUtils.get_image_transforms(
            to_grayscale=self.model_props['to_grayscale'])
        dataset = ImageFolder(self.samples_path,
                              transformations,
                              is_valid_file=lambda file: Path(file).suffix \
                                               in self.IMG_EXTENSIONS
                              )
        self.loader = DataLoader(dataset,
                                 batch_size=self.batch_size, 
                                 shuffle=True, 
                                 drop_last=True 
                                 )
        self.num_classes = len(dataset.classes)
        self.class_names = dataset.classes
        
        # Get the right type of model,
        # Don't bother getting it pretrained,
        # of freezing it, b/c we will overwrite 
        # the weights:
        
        self.model = NetUtils.get_net(
            self.model_props['net_name'],
            num_classes=self.num_classes,
            pretrained=False,
            freeze=0,
            to_grayscale=self.model_props['to_grayscale']
            )
        
    #------------------------------------
    # run_inferencer 
    #-------------------
    
    def run_inference(self):
        '''
        Runs model over dataloader. Along
        the way: creates ResultTally for each
        batch, and maintains dict instance variable
        self.raw_results for later conversion of
        logits to class IDs under different threshold
        assumptions. 
        
        self.raw_results: 
                {'all_outputs' : <arr>,
                 'all_labels'  : <arr>
                 }
        
        Returns a ResultCollection with the
        ResultTally instances of each batch.
        
        :return: collection of tallies, one for each batch,
            or None if something went wrong.
        :rtype: {None | ResultCollection}
        '''
        # Just in case the loop never runs:
        batch_num   = -1

        try:
            try:
                if torch.cuda.is_available():
                    self.model.load_state_dict(torch.load(self.model_path))
                    FileUtils.to_device(self.model, 'gpu', self.gpu_id)
                else:
                    self.model.load_state_dict(torch.load(
                        self.model_path,
                        map_location=torch.device('cpu')
                        ))
            except RuntimeError as e:
                emsg = repr(e)
                if emsg.find("size mismatch for conv1") > -1:
                    emsg += " Maybe model was trained with to_grayscale=False, but local net created for grayscale?"
                    raise RuntimeError(emsg) from e

            loss_fn = nn.CrossEntropyLoss()
    
            result_coll = ResultCollection()
            
            # Save all per-class logits for ability
            # later to use different thresholds for
            # conversion to class IDs:
            
            all_outputs = []
            all_labels  = []

            self.model.eval()
            num_test_samples  = len(self.loader.dataset)
            self.log.info(f"Begin inference ({num_test_samples} test samples)...")

            samples_processed = 0
            
            overall_start_time = datetime.datetime.now()
            loop_start_time    = overall_start_time
            with torch.no_grad():
                
                for batch_num, (batch, targets) in enumerate(self.loader):
                    if torch.cuda.is_available():
                        images = FileUtils.to_device(batch, 'gpu')
                        labels = FileUtils.to_device(targets, 'gpu')
                    else:
                        images = batch
                        labels = targets
                    
                    outputs = self.model(images)
                    loss    = loss_fn(outputs, labels)
                    
                    images  = FileUtils.to_device(images, 'cpu')
                    outputs = FileUtils.to_device(outputs, 'cpu')
                    labels  = FileUtils.to_device(labels, 'cpu')
                    loss    = FileUtils.to_device(loss, 'cpu')
                    
                    # Specify the batch_num in place
                    # of an epoch, which is not applicatble
                    # during testing:
                    tally = ResultTally(batch_num,
                                        LearningPhase.TESTING,
                                        outputs, 
                                        labels, 
                                        loss,
                                        self.num_classes,
                                        self.batch_size)
                    result_coll.add(tally, step=None)
                    
                    all_outputs.append(outputs)
                    all_labels.append(labels)
                    
                    samples_processed += len(labels)
                    
                    del images
                    del outputs
                    del labels
                    del loss

                    torch.cuda.empty_cache()
                    
                    time_now = datetime.datetime.now()
                    # Sign of life every 6 seconds:
                    if (time_now - loop_start_time).seconds >= 5:
                        self.log.info(f"Processed {samples_processed}/{num_test_samples} samples")
                        loop_start_time = time_now 
        finally:
            
            time_now = datetime.datetime.now()
            test_time_duration = overall_start_time - time_now
            # A human readable duration st down to minutes:
            duration_str = FileUtils.time_delta_str(test_time_duration, granularity=4)
            self.log.info(f"Done with inference: {samples_processed} test samples; {duration_str}")
            # Total number of batches we ran:
            num_batches = 1 + batch_num # b/c of zero-base
            
            # If loader delivered nothing, the loop
            # never ran; warn, and get out:
            if num_batches == 0:
                self.log.warn(f"Dataloader delivered no data from {self.samples_path}")
                self.close()
                return None
            
            # Var all_outputs is now:
            #  [tensor([pred_cl0, pred_cl1, pred_cl<num_classes - 1>], # For sample0
            #   tensor([pred_cl0, pred_cl1, pred_cl<num_classes - 1>], # For sample1
            #                     ...
            #   ]
            # Make into one tensor: (num_batches, batch_size, num_classes),
            # unless an exception was raised at some point,
            # throwing us into this finally clause:
            if len(all_outputs) == 0:
                self.log.info(f"No outputs were produced; thus no results to report")
                return None
            
            self.all_outputs_tn = torch.stack(all_outputs)
            # Be afraid...be very afraid:
            assert(self.all_outputs_tn.shape == \
                   torch.Size([num_batches, 
                               self.batch_size, 
                               self.num_classes])
                   )
            
            # Var all_labels is now num-batches tensors,
            # each containing batch_size labels:
            assert(len(all_labels) == num_batches)
            
            # list of single-number tensors. Make
            # into one tensor:
            self.all_labels_tn = torch.stack(all_labels)
            assert(self.all_labels_tn.shape == \
                   torch.Size([num_batches, self.batch_size])
                   )
            # And equivalently:
            assert(self.all_labels_tn.shape == \
                   (self.all_outputs_tn.shape[0], 
                    self.all_outputs_tn.shape[1]
                    )
                   )
            
            self.report_results(result_coll)
            self.close()
            
        return result_coll
    
    #------------------------------------
    # report_results 
    #-------------------
    
    def report_results(self, tally_coll):
        self._report_textual_results(tally_coll, self.csv_dir)
        self._report_charted_results()

    #------------------------------------
    # _report_charted_results 
    #-------------------
    
    def _report_charted_results(self, thresholds=None):
        '''
        Computes and (pyplot-)shows a set of precision-recall
        curves in one plot:
        
        :param thresholds: list of cutoff thresholds
            for turning logits into class ID predictions.
            If None, the default at TensorBoardPlotter.compute_multiclass_pr_curves()
            is used.
        :type thresholds: [float]
        '''

        # Obtain a dict of CurveSpecification instances,
        # one for each class, plus the mean Average Precision
        # across all curves. The dict will be keyed
        # by class ID:

        (all_curves_info, mAP) = \
          Charter.compute_multiclass_pr_curves(
              self.all_labels_tn,
              self.all_outputs_tn,
              thresholds
              )
          
        # Separate out the curves without 
        # ill defined prec, rec, or f1:
        well_defined_curves = filter(
            lambda crv_obj: not(crv_obj['undef_prec'] or\
                                crv_obj['undef_rec'] or\
                                crv_obj['undef_f1']
                                ),
            all_curves_info.values()
            )
        
        # Too many curves are clutter. Only
        # show the best and worst by optimal f1:
        f1_sorted = sorted(well_defined_curves,
                           key=lambda obj: obj['best_op_pt']['f1']
                           )
        curves_to_show = {crv_obj['class_id'] : crv_obj
                          for crv_obj in (f1_sorted[0], f1_sorted[-1])
                          }
        #********** Mixup with objs blurring together
        
        (_num_classes, fig) = \
          ClassificationPlotter.chart_pr_curves(curves_to_show)

        fig.show()

    #------------------------------------
    # _report_textual_results 
    #-------------------
    
    def _report_textual_results(self, tally_coll, res_dir):
        '''
        Give a sequence of tallies with results
        from a series of batches, create long
        outputs, and inputs lists from all tallies
        Then write a CSV file, and create a text
        table with the results. Report the table 
        to tensorboard if possible, and return the
        table text.
        
        :param tally_coll: collect of tallies from batches
        :type tally_coll: ResultCollection
        :param res_dir: directory where all .csv and other 
            result files are to be written
        :type res_dir: str
        :return table of results
        :rtype: str
        '''
        
        all_preds   = []
        all_labels  = []
        
        for tally in tally_coll.tallies(phase=LearningPhase.TESTING):
            all_preds.extend(tally.preds)
            all_labels.extend(tally.labels)
        
        res = OrderedDict({})
        res['prec_macro']       = precision_score(all_labels, 
                                           all_preds, 
                                           average='macro',
                                           zero_division=0)
        res['prec_micro']       = precision_score(all_labels, 
                                           all_preds, 
                                           average='micro',
                                           zero_division=0)
        res['prec_weighted']    = precision_score(all_labels, 
                                           all_preds, 
                                           average='weighted',
                                           zero_division=0)
        res['prec_by_class']    = precision_score(all_labels, 
                                           all_preds, 
                                           average=None,
                                           zero_division=0)
        
        res['recall_macro']     = recall_score(all_labels, 
                                        all_preds, 
                                        average='macro',
                                        zero_division=0)
        res['recall_micro']     = recall_score(all_labels, 
                                        all_preds, 
                                        average='micro',
                                        zero_division=0)
        res['recall_weighted'] = recall_score(all_labels, 
                                       all_preds, 
                                       average='weighted',
                                       zero_division=0)
        res['recall_by_class']  = recall_score(all_labels, 
                                        all_preds, 
                                        average=None,
                                        zero_division=0)

        res['f1_macro']         = f1_score(all_labels, 
                                    all_preds, 
                                    average='macro',
                                    zero_division=0)
        res['f1_micro']         = f1_score(all_labels, 
                                    all_preds, 
                                    average='micro',
                                    zero_division=0)
        res['f1_weighted']      = f1_score(all_labels, 
                                    all_preds, 
                                    average='weighted',
                                    zero_division=0)
        res['f1_by_class']      = f1_score(all_labels, 
                                    all_preds, 
                                    average=None,
                                    zero_division=0)

        res['accuracy']          = accuracy_score(all_labels, all_preds)
        res['balanced_accuracy'] = balanced_accuracy_score(all_labels, all_preds)
        
        res_series = pd.Series(list(res.values()),
                           index=list(res.keys())
                           )
        
        # Write information retrieval type results
        # to a one-line .csv file, using pandas Series
        # as convenient intermediary:
        res_csv_path = os.path.join(res_dir, 'ir_results.csv')
        res_series.to_csv(res_csv_path)
        
        # Get confusion matrix as a tensor,
        # with fields normalized to 1 (i.e. percentages).
        conf_matrix = Charter.compute_confusion_matrix(
            all_labels,
            all_preds,
            self.num_classes,
            normalize=True
            )
        
        # Write confusion matrix to CSV, 
        # using dataframe as convenient intermediary:
        conf_matrix_df = pd.DataFrame(conf_matrix.numpy(),
                                      index=self.class_names,
                                      columns=self.class_names
                                      )
        cm_csv_path = os.path.join(res_dir, 'conf_matrix_results.csv')
        conf_matrix_df.to_csv(cm_csv_path)
        
        # Post to tensorboard as an image:
        TensorBoardPlotter.conf_matrix_to_tensorboard(
            self.writer,
            conf_matrix,
            self.class_names
            )
        
        # Make textual tables using Github flavored
        # markup: Results: one column with the model 
        # properties:
        ir_measures_skel = {'col_header' : ['precision', 'recall', 'f1'], 
                            'row_labels' : ['macro','micro','weighted'],
                            'rows'       : [[res['prec_macro'],    res['recall_macro'],    res['f1_macro']],
                                            [res['prec_micro'],    res['recall_micro'],    res['f1_micro']],
                                            [res['prec_weighted'], res['recall_weighted'], res['f1_weighted']]
                                           ]  
                           }
        
        ir_per_class_rows = [[prec_class, recall_class, f1_class]
                            for prec_class, recall_class, f1_class
                            in zip(res['prec_by_class'], res['recall_by_class'], res['f1_by_class'])
                            ]
        ir_by_class_skel =  {'col_header' : ['precision','recall', 'f1'],
                             'row_labels' : self.class_names,
                             'rows'       : ir_per_class_rows
                             }
        
        accuracy_skel = {'col_header' : ['accuracy', 'balanced_accuracy'],
                         'row_labels' : ['Overall'],
                         'rows'       : [[res['accuracy'], res['balanced_accuracy']]]
                         }

        ir_measures_tbl  = GithubTableMaker.make_table(ir_measures_skel)
        ir_by_class_tbl  = GithubTableMaker.make_table(ir_by_class_skel)
        accuracy_tbl     = GithubTableMaker.make_table(accuracy_skel)
        
        # Write the markup tables to Tensorboard:
        self.writer.add_text('Information retrieval measures', ir_measures_tbl, global_step=0)
        self.writer.add_text('Per class measures', ir_by_class_tbl, global_step=0)
        self.writer.add_text('Accuracy', accuracy_tbl, global_step=0)
        
        return{'info_retrieval_res' : res_series,
               'confusion_matrx'    : conf_matrix_df 
               }

    #------------------------------------
    # close 
    #-------------------
    
    def close(self):
        try:
            self.writer.close()
        except Exception as e:
            self.log.err(f"Could not close tensorboard writer: {repr(e)}")
        try:
            self.csv_writer.close()
        except Exception as e:
            self.log.err(f"Could not close CSV writer: {repr(e)}")


# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Run inference on given model"
                                     )

    parser.add_argument('-l', '--labels_path',
                        help='optional path to labels for use in measures calcs; default: no measures',
                        default=None)
    parser.add_argument('-b', '--batch_size',
                        help='batch size to use; default: 1',
                        default=1
                        )
    parser.add_argument('-d', '--device',
                        help='device number of GPU (zero-origin); only used if GPU available; defaults to 0',
                        default=0
                        )

    parser.add_argument('model_path',
                        help='path to the saved Pytorch model',
                        default=None)
    parser.add_argument('samples_path',
                        help='path to samples to run through model',
                        default=None)

    args = parser.parse_args();

    inferencer = Inferencer(
                    args.model_path, 
                    args.samples_path,
                    batch_size=args.batch_size, 
                    labels_path=args.labels_path,
                    gpu_id=args.device
                    )
    res_coll = inferencer.run_inference()
    if res_coll is None:
        # Something went wrong (and was reported earlier)
        sys.exit(1)
    FileUtils.user_confirm("Hit any key to close images and end...")
