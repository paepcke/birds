#!/usr/bin/env python3
'''
Created on Mar 12, 2021

@author: paepcke

TODO:
    o Catch cnt-C and complete without a stack trace
       after finishing the 'finally' clause
    o Why always: 
      run_inference.py(3254979): 2021-07-06 14:01:54,082;WARNING: For all thresholds, one or more of precision,
              recall or f1 are undefined. No p/r curves to show
    o ir_results.csv: 
        ,0       
        prec_macro,0.0622439088620259
        prec_micro,0.07980347329707624
              ...
      Why the leading zero?
      

'''
from _collections import OrderedDict
import argparse
import datetime
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import traceback as tb

from logging_service.logging_service import LoggingService
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score 
from torch import nn
import torch
from torch.utils.data.dataloader import DataLoader

from skorch import NeuralNet
from skorch.callbacks import EpochScoring

from birdsong.ml_plotting.classification_charts import ClassificationPlotter
from birdsong.nets import NetUtils
from birdsong.result_tallying import ResultTally, ResultCollection
from birdsong.rooted_image_dataset import SingleRootImageDataset
from birdsong.utils.github_table_maker import GithubTableMaker
from birdsong.utils.learning_phase import LearningPhase
from birdsong.utils.tensorboard_plotter import SummaryWriterPlus
from birdsong.utils.utilities import FileUtils, CSVWriterCloseable
from data_augmentation.utils import Utils
import numpy as np
import pandas as pd
from result_analysis.charting import Charter, CELL_LABELING
from seaborn.matrix import heatmap

class Inferencer:
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 model_paths, 
                 samples_path,
                 batch_size=1, 
                 labels_path=None,
                 gpu_ids=0,
                 sampling=None,
                 unittesting=False
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
        
        :param model_paths:
        :type model_paths:
        :param samples_path:
        :type samples_path:
        :param batch_size:
        :type batch_size:
        :param labels_path:
        :type labels_path:
        :param gpu_ids: Device number of GPU, in case 
            one is available
        :type gpu_ids: {int | [int]}
        :param sampling: only use given percent of the samples in
            each class.
        :type sampling: {int | None}
        :param unittesting: if True, returns immediately
        :type unittesting: bool
        '''

        if unittesting:
            return

        self.model_paths  = model_paths
        self.samples_path = samples_path
        self.labels_path  = labels_path
        self.gpu_ids      = gpu_ids if type(gpu_ids) == list else [gpu_ids]
        self.sampling     = sampling
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        
        self.IMG_EXTENSIONS = FileUtils.IMG_EXTENSIONS
        self.log      = LoggingService()
        self.curr_dir = os.path.dirname(__file__)

    #------------------------------------
    # prep_model_inference 
    #-------------------

    def prep_model_inference(self, model_path):
        '''
        1. Parses model_path into its components, and 
            creates a dict: self.model_props, which 
            contains the network type, grayscale or not,
            whether pretrained, etc.
        2. Creates self.csv_writer to write results measures
            into csv files. The destination file is determined
            as follows:
                <script_dir>/runs_raw_inferences/inf_csv_results_<datetime>/<model-props-derived-fname>.csv
        3. Creates self.writer(), a tensorboard writer with destination dir:
                <script_dir>/runs_inferences/inf_results_<datetime>
        4. Creates an ImageFolder classed dataset to self.samples_path
        5. Creates a shuffling DataLoader
        6. Initializes self.num_classes and self.class_names
        7. Creates self.model from the passed-in model_path name
        
        :param model_path: path to model that will be used for
            inference by this instance of Inferencer
        :type model_path: str
        '''
        
        self.model_path = model_path
        model_fname = os.path.basename(model_path)
        
        # Extract model properties
        # from the model filename:
        self.model_props  = FileUtils.parse_filename(model_fname)
        
        csv_results_root = os.path.join(self.curr_dir, 'runs_raw_inferences')
        #self.csv_dir = os.path.join(csv_results_root, f"inf_csv_results_{uuid.uuid4().hex}")
        ts = FileUtils.file_timestamp()
        self.csv_dir = os.path.join(csv_results_root, f"inf_csv_results_{ts}")
        os.makedirs(self.csv_dir, exist_ok=True)

        csv_file_nm = FileUtils.construct_filename(
            self.model_props,
            prefix='inf',
            suffix='.csv', 
            incl_date=True)
        csv_path = os.path.join(self.csv_dir, csv_file_nm)
        
        self.csv_writer = CSVWriterCloseable(csv_path)
        
        ts = FileUtils.file_timestamp()
        tensorboard_root = os.path.join(self.curr_dir, 'runs_inferences')
        tensorboard_dest = os.path.join(tensorboard_root,
                                        f"inf_results_{ts}"
                                        )
                                        #f"inf_results_{ts}{uuid.uuid4().hex}")
        os.makedirs(tensorboard_dest, exist_ok=True)
        
        self.writer = SummaryWriterPlus(log_dir=tensorboard_dest)
        
        dataset = SingleRootImageDataset(
            self.samples_path,
            to_grayscale=self.model_props['to_grayscale'],
            percentage=self.sampling
            )
        
        # Make reproducible:
        Utils.set_seed(42)
        self.loader = DataLoader(dataset,
                                 batch_size=self.batch_size, 
                                 shuffle=True, 
                                 drop_last=True 
                                 )
        self.class_names = dataset.class_names()
        self.num_classes = len(self.class_names)
        
        csv_data_header = self.class_names + ['label']
        self.csv_writer.writerow(csv_data_header)
         


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

        self.log.info(f"Tensorboard info written to {tensorboard_dest}")
        self.log.info(f"Result measurement CSV file(s) written to {csv_path}")
        
    #------------------------------------
    # __call__ 
    #-------------------
    
    def __call__(self, gpu_id_model_path_pair):
        gpu_id, self.model_path = gpu_id_model_path_pair
        self.prep_model_inference(self.model_path)
        self.log.info(f"Begining inference with model {FileUtils.ellipsed_file_path(self.model_path)} on gpu_id {gpu_id}")
        #****************
        return self.run_inference(gpu_to_use=gpu_id)
        # dicts_from_runs = []
        # for _i in range(3):
        #     self.curr_dict = {}
        #     dicts_from_runs.append(self.curr_dict)
        #     self.run_inference(gpu_to_use=gpu_id)
        # print(dicts_from_runs)
        #****************

    #------------------------------------
    # go 
    #-------------------

    def go(self):
        # Pair models to GPUs; example for 
        # self.gpu_ids == [0,4], and three models:
        #    [(gpu0, model0) (gpu4, model1), (gpu0, model3)]
        
        repeats = int(np.ceil(len(self.model_paths) / len(self.gpu_ids)))
        gpu_model_pairings = list(zip(self.gpu_ids*repeats, self.model_paths))
        
        #************* No parallelism for debugging
        result_collection = self(gpu_model_pairings[0])
        return result_collection
        #************* END No parallelism for debugging
        
        with Pool(len(self.gpu_ids)) as inf_pool:
            # Run as many inferences in parallel as
            # there are models to try. The first arg,
            # (self): means to invoke the __call__() method
            # on self.
            result_it = inf_pool.imap(self, 
                                      gpu_model_pairings,
                                      chunksize=len(self.gpu_ids)
                                      )
            results = [res.get() for res in result_it]
            print(f"******Results: {results}")

    #------------------------------------
    # run_inferencer 
    #-------------------
    
    def run_inference(self, gpu_to_use=0):

        
        sko_net = NeuralNet(
            module=self.model,
            criterion=torch.nn.NLLLoss,
            batch_size=self.batch_size,
            train_split=None,    # We won't train, just infer
            #callbacks=[EpochScoring('f1')]
            device=f"cuda:{gpu_to_use}" if torch.cuda.is_available() else "cpu"
            )

        sko_net.initialize()  # This is important!
        sko_net.load_params(f_params=self.model_path)
        truth = []
        pred_logits = []
        for batch_num, (X,y) in enumerate(self.loader):
            pred_logits.append(sko_net.predict(X))
            truth.append(y)
            #********
            #if batch_num > 5:
            #    break
            #********
            
        # Preds are now a list of np-arrays, each
        # of which has shape (batch_size, num_classes).
        # The length of the list is the number of batches
        # retrieved from the dataset. Use map() to
        # turn the np arrays into tensors, and concatenate
        # the rows (i.e. all the batches) into one tensor:
        pred_logits_tn = torch.cat(list(map(torch.tensor, pred_logits)))
        
        # Get a tensor:
        #    sample0  [prob_class0, prob_class1, ..., prob_class_n]
        #    sample1  [prob_class0, prob_class1, ..., prob_class_n]
        #                  ...
        
        pred_probs = torch.softmax(pred_logits_tn, dim=1)
        
        # Probabilities across one row must add to 1:
        # But due to rounding they add to 0.99999999xyz
        #assert(sum(pred_probs[0,:]) == 1)
        
        # Truth is a list of tensors, each of batch_size
        # length. The length of the list (i.e. then number
        # of tensors is the number of batches extracted from
        # the dataset. Turn into a 1-D tensor of truth:
        truth_tn = torch.cat(truth)
        
        # For each row in pred_probs, get the 
        # most likely class; the position of the max
        # probability is synonymous with the class ID.
        # Get:
        #    sample0 class3
        #    sample1 class1
        #       ...
        pred_classes = pred_probs.argmax(dim=1)

        # For evaluation the whole outcome, use
        # Pandas data structures:
        
        truth_series  = pd.Series(truth_tn)
        pred_probs_df = pd.DataFrame(pred_probs)
        # Turn each of the df's one-element tensors
        # into floats:
        pred_probs_df = pred_probs_df.applymap(lambda el: el.item())
        
        self.report_results(truth_series, pred_probs_df, pred_classes)

    #------------------------------------
    # run_inferencerOLD 
    #-------------------
    
    def run_inferenceOLD(self, gpu_to_use=0):
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

        :param gpu_to_use: which GPU to deploy to (if it is available)
        :type gpu_to_use: int
        :return: collection of tallies, one for each batch,
            or None if something went wrong.
        :rtype: {None | ResultCollection}
        '''
        # Just in case the loop never runs:
        batch_num   = -1
        overall_start_time = datetime.datetime.now()

        # Load the previously created model
        # with the learned weights:
        
        try:
            try:
                if torch.cuda.is_available():
                    self.model.load_state_dict(torch.load(self.model_path))
                    self.model = FileUtils.to_device(self.model, 'gpu', gpu_to_use)
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

                    #**********
                    # max_logit = outputs[0].max().item()
                    # max_idxes = (outputs.squeeze() == max_logit).nonzero(as_tuple=False)
                    # max_idx = max_idxes.amax().item()
                    # smpl_id = torch.utils.data.dataloader.sample_id_seq[-1]
                    # lbl     = labels[0].item()
                    # pred_cl = max_idx
                    #
                    # try:
                        # self.curr_dict[smpl_id] = (smpl_id, lbl, pred_cl)
                    # except AttributeError:
                        # self.curr_dict = {smpl_id : (smpl_id, lbl, pred_cl)}
                    # #**********
                    
                    
                    # Specify the batch_num in place
                    # of an epoch, which is not applicatble
                    # during testing:
                    tally = ResultTally(batch_num,
                                        LearningPhase.TESTING,
                                        outputs, 
                                        labels, 
                                        loss,
                                        self.class_names,
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
                        self.log.info(f"GPU{gpu_to_use} processed {samples_processed}/{num_test_samples} samples")
                        loop_start_time = time_now 
        except Exception as e:
            # Print stack trace:
            tb.print_exc()
            self.log.err(f"Error during inference loop: {repr(e)}")
            return
        finally:
            
            time_now = datetime.datetime.now()
            test_time_duration = time_now - overall_start_time
            # A human readable duration st down to minutes:
            duration_str = Utils.time_delta_str(test_time_duration, granularity=4)
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
    
    def report_results(self, truth_series, pred_probs, predicted_classes):
        #self._report_textual_results(tally_coll, self.csv_dir)
        Inferencer.conf_matrix_fig = self._report_conf_matrix(truth_series, predicted_classes, show_in_tensorboard=True)
        Inferencer.pr_curve_fig = self._report_charted_results(truth_series, pred_probs, predicted_classes)
        

    #------------------------------------
    # _report_conf_matrix 
    #-------------------
    
    def _report_conf_matrix(self, 
                            truth_series, 
                            predicted_classes, 
                            show=True, 
                            show_in_tensorboard=None):
        '''
        Computes the confusion matrix CM from the truth/prediction
        series.
        
        Creates an image from CM, and displays it via matplotlib, 
        if show arg is True. If show_in_tensorboard is a Tensorboard
        SummaryWriter instance, the figure is posted to tensorboard,
        no matter the value of the show arg.  
        
        Returns the Figure object.
        
        :param truth_series: true labels
        :type truth_series: pd.Series
        :param predicted_classes: the classes predicted by the 
            classifier
        :type predicted_classes: pd.Series
        :param show: whether or not to call show() on the
            confusion matrix figure, or only return the Figure instance
        :type show: bool
        :param show_in_tensorboard: whether or not to post the image
            to tensorboard
        :type show_in_tensorboard: bool
        :return: Figure instance containing confusion matrix heatmap
            with color legend.
        :rtype: matplotlib.pyplot.Figure
        '''

        conf_matrix = Charter.compute_confusion_matrix(truth_series,
                                                       predicted_classes,
                                                       self.class_names,
                                                       normalize=True
                                                       )
        
        # Normalization in compute_confusion_matrix() is
        # to 0-1. Turn those values into percentages:
        # conf_matrix_perc = (100 * conf_matrix).astype(int)
        
        # Decide whether or not to write 
        # confusion cell values into the cells.
        # The decision depends on how many species
        # are represented in the conf matrix; too many,
        # and having numbers in all cells is too cluttered:
        
        if len(self.class_names) > CELL_LABELING.CONF_MATRIX_CELL_LABEL_LIMIT.value:
            write_in_fields=CELL_LABELING.DIAGONAL
        else:
            write_in_fields=CELL_LABELING.ALWAYS
            
        fig = Charter.fig_from_conf_matrix(conf_matrix,
                                           supertitle='Confusion Matrix\n',
                                           subtitle='Normalized to percentages',
                                           write_in_fields=write_in_fields
                                           )
        if show_in_tensorboard:
            self.writer.add_figure('Inference Confusion Matrix', 
                                   fig,
                                   global_step=0)

        if show:
            # Something above makes fig lose its
            # canvas manager. Add that back in:
            Utils.add_pyplot_manager_to_fig(fig)
            fig.show()
        return fig
        

    #------------------------------------
    # _report_charted_results 
    #-------------------
    
    def _report_charted_results(self, truth_series, pred_probs_df, predicted_classes):
        '''
        Computes and (pyplot-)shows a set of precision-recall
        curves in one plot. If precision and/or recall are 
        undefined (b/c of division by zero) for all curves, then
        returns False, else True. If no curves are defined,
        logs a warning.
        
        :param thresholds: list of cutoff thresholds
            for turning logits into class ID predictions.
            If None, the default at Charters.compute_multiclass_pr_curves()
            is used.
        :type thresholds: [float]
        :return: True if curves were computed and show. Else False
        :rtype: bool
        '''

        # Obtain a dict of CurveSpecification instances,
        # one for each class, plus the mean Average Precision
        # across all curves. The dict will be keyed
        # by class ID:
        
        mAP, pr_curves = Charter.visualize_testing_result(truth_series, pred_probs_df)
        
        # Separate out the curves without 
        # ill defined prec, rec, or f1. Prec and
        # rec should have none, b/c NaNs were interpolated
        # out earlier.
        well_defined_curves = list(filter(
                    lambda crv_obj: crv_obj.undef_prec() + \
                                    crv_obj.undef_rec() + \
                                    crv_obj.undef_f1() \
                                    == 0,
                    pr_curves.values()
                    )
            )
        
        if len(well_defined_curves) == 0:
            self.log.warn(f"For all thresholds, one or more of precision, recall or f1 are undefined. No p/r curves to show")
            return False
        
        # Too many curves are clutter. Only
        # show the best, worst, and median by optimal f1;
        f1_sorted_curves = sorted(well_defined_curves,
                           key=lambda curve: curve['best_op_pt']['f1']
                           )
        curves_to_show = self.pick_pr_curve_classes(f1_sorted_curves)

        fig = ClassificationPlotter.chart_pr_curves(curves_to_show)

        # Get human readable name of hardest
        # class (lowest average precision for 1-against-all),
        # and easiest
        # [the try/except/else, and legend texts
        #    should be done way more elegantly!]:
        try:
            hardest_cl_id, median_cl_id, easiest_cl_id = [crv['class_id'] 
                                                          for crv 
                                                          in curves_to_show]
        except ValueError:
            # Happens when less than 3 classes are at play:
            hardest_cl_id = curves_to_show[0]['class_id']
            hardest_cl_nm = self.class_names[hardest_cl_id]
            if len(curves_to_show) == 2:
                easiest_cl_id = curves_to_show[1]['class_id']
                easiest_cl_nm = self.class_names[easiest_cl_id]
            else:
                # Only one curve:
                easiest_cl_nm = None
            median_cl_nm = None
        else:
            hardest_cl_nm = self.class_names[hardest_cl_id]
            median_cl_nm  = self.class_names[median_cl_id]
            easiest_cl_nm = self.class_names[easiest_cl_id]
        
        legend_cl_names = list(filter(lambda el: el is not None,
                                      [hardest_cl_nm, 
                                       median_cl_nm, 
                                       easiest_cl_nm]))
        legend_txts = []
        if len(legend_cl_names) > 1:
            legend_txts.append(f"Hardest: {hardest_cl_nm}")
        else:
            # Only one curve; just list the species name:
            legend_txts.append(hardest_cl_nm)
        if len(legend_cl_names) == 2:
            legend_txts.append(f"Easiest: {easiest_cl_nm}")
        elif len(legend_cl_names) > 2:
            # Got three curves:
            legend_txts.append(f"Medium: {median_cl_nm}")
            legend_txts.append(f"Easiest: {easiest_cl_nm}")
        
        legend = fig.axes[0].get_legend()
        # Get text lines:
        #    'class 0'
        #    'class 1'
        #    'Optimal operation'
        existing_legend_texts  = legend.get_texts()
        for i, txt in enumerate(legend_txts):
            existing_legend_texts[i].set_text(txt)

        fig.show()
        return True

    #------------------------------------
    # _report_textual_results 
    #-------------------
    
    def _report_textual_results(self, tally_coll, res_dir):
        '''
        Give a sequence of tallies with results
        from a series of batches, create long
        outputs, and inputs lists from all tallies
        
        Computes information retrieval type values:
             precision (macro/micro/weighted/by-class)
             recall    (macro/micro/weighted/by-class)
             f1        (macro/micro/weighted/by-class)
             acuracy
             balanced_accuracy
        
        Combines these results into a Pandas series, 
        and writes them to a csv file. That file is constructed
        from the passed-in res_dir, appended with 'ir_results.csv'.
        
        Finally, constructs Github flavored tables from the
        above results, and posts them to the 'text' tab of 
        tensorboard.
        
        Returns the results measures Series 
        
        :param tally_coll: collect of tallies from batches
        :type tally_coll: ResultCollection
        :param res_dir: directory where all .csv and other 
            result files are to be written
        :type res_dir: str
        :return results of information retrieval-like measures
        :rtype: pandas.Series
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
        # The series looks like:
        #    prec_macro,0.04713735383721669
        #    prec_micro,0.0703125
        #    prec_weighted,0.047204446832196136
        #    prec_by_class,"[0.         0.09278351 0.         0.         0.05747126 0.08133971
        #        0.03971119 0.18181818 0.07194245 0.0877193  0.         0.
        #        0.        ]"
        #    recall_macro,0.07039151324865611
        #    recall_micro,0.0703125
        #
        # We want a df like:
        #   prec_macro,prec_micro,...prec_GLDH, prec_VASE, ..., rec_GLDH, rec_VASE,...
        #
        # Expand the per-class-pred:
        
        
        
        
        # Write information retrieval type results
        # to a one-line .csv file, using pandas Series
        # and df as convenient intermediary:
        res_csv_path = os.path.join(res_dir, 'ir_results.csv')
        res_df = self.res_measures_to_df(res_series)
        res_df.to_csv(res_csv_path, index=False)

        # Next, construct Tensorboard markup tables
        # for the same results:
        res_rnd = {}
        for meas_nm, meas_val in res.items():
            
            # Measure results are either floats (precision, recall, etc.),
            # or np arrays (e.g. precision-per-class). For both
            # cases, round each measure to one digit:
            
            res_rnd[meas_nm] = round(meas_val,1) if type(meas_val) == float \
                                                 else meas_val.round(1)
        
        ir_measures_skel = {'col_header' : ['precision', 'recall', 'f1'], 
                            'row_labels' : ['macro','micro','weighted'],
                            'rows'       : [[res_rnd['prec_macro'],    res_rnd['recall_macro'],    res_rnd['f1_macro']],
                                            [res_rnd['prec_micro'],    res_rnd['recall_micro'],    res_rnd['f1_micro']],
                                            [res_rnd['prec_weighted'], res_rnd['recall_weighted'], res_rnd['f1_weighted']]
                                           ]  
                           }
        
        ir_per_class_rows = [[prec_class, recall_class, f1_class]
                            for prec_class, recall_class, f1_class
                            in zip(res_rnd['prec_by_class'], res_rnd['recall_by_class'], res_rnd['f1_by_class'])
                            ]
        ir_by_class_skel =  {'col_header' : ['precision','recall', 'f1'],
                             'row_labels' : self.class_names,
                             'rows'       : ir_per_class_rows
                             }
        
        accuracy_skel = {'col_header' : ['accuracy', 'balanced_accuracy'],
                         'row_labels' : ['Overall'],
                         'rows'       : [[res_rnd['accuracy'], res_rnd['balanced_accuracy']]]
                         }

        ir_measures_tbl  = GithubTableMaker.make_table(ir_measures_skel, sep_lines=False)
        ir_by_class_tbl  = GithubTableMaker.make_table(ir_by_class_skel, sep_lines=False)
        accuracy_tbl     = GithubTableMaker.make_table(accuracy_skel, sep_lines=False)
        
        # Write the markup tables to Tensorboard:
        self.writer.add_text('Information retrieval measures', ir_measures_tbl, global_step=0)
        self.writer.add_text('Per class measures', ir_by_class_tbl, global_step=0)
        self.writer.add_text('Accuracy', accuracy_tbl, global_step=0)
        
        return res_series

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


# --------------------- Utilities -------------------

    #------------------------------------
    # pick_pr_curve_classes 
    #-------------------

    def pick_pr_curve_classes(self, f1_sorted_curves):
        '''
        Given a list of CurveSpecification instances
        that are already sorted by increasing f1 values
        of their best operating point.
        
        From among those instances, find three 'interesting' 
        curves to plot in a precision-recall chart. 
        
        We pick the two extremes, and the curve with best-op-pt 
        having the median f1. If the median f1 is the same
        as one of the other two, move along the list of
        curves to find one with a different f1. 
        
        :param f1_sorted_curves: list of CurveSpecification
            instances that are sorted by the f1 values of
            their best operating point.
        :type f1_sorted_curves: [CurveSpecification]
        :return three CurveSpecification instances
        :rtype [CurveSpecification]
        '''
        
        if len(f1_sorted_curves) <= 3:
            # Use all curves:
            return f1_sorted_curves
        
        min_best_f1_crv = f1_sorted_curves[0]
        max_best_f1_crv = f1_sorted_curves[-1]
        med_f1_crv      = f1_sorted_curves[len(f1_sorted_curves) // 2]
        
        min_f1 = min_best_f1_crv['best_op_pt']['f1']
        max_f1 = max_best_f1_crv['best_op_pt']['f1']
        med_f1 = med_f1_crv['best_op_pt']['f1']
        
        if med_f1 not in (min_f1, max_f1):
            return [min_best_f1_crv, med_f1_crv, max_best_f1_crv] 
        
        # Median of the best op points' f1 scores is
        # same as one of the extremes:
        
        f1_scores = [crv['best_op_pt']['f1'] 
                     for crv in f1_sorted_curves]
        med_f1_idx = f1_sorted_curves.index(med_f1_crv)
        if med_f1 == min_f1:
            # Fallback: if we won't find
            # a curve with a best op pt's f1 other
            # than the curve with the lowest best-op-pt f1,
            # use the curve above the lowest f1
            middle_idx = 1
            for idx in range(med_f1_idx, len(f1_scores)):
                middle_f1 = f1_scores[idx]
                if middle_f1 != min_f1:
                    middle_idx = idx
                    break
        else:
            # Median f1 is same as max f1; search backward:
            # Fallback: if we won't find
            # a curve with a best op pt's f1 other
            # than the curve with the highest best-op-pt f1,
            # use the curve below the lowest f1
            middle_idx = -2
            for idx in range(med_f1_idx, 0, -1):
                middle_f1 = f1_scores[idx]
                if middle_f1 != max_f1:
                    middle_idx = idx
                    break

        return [min_best_f1_crv,
                f1_sorted_curves[middle_idx],
                max_best_f1_crv]

    #------------------------------------
    # res_measures_to_df 
    #-------------------
    
    def res_measures_to_df(self, measure_series):
        '''
        Given a pandas series like:
        
            prec_macro,0.04713735383721669
            prec_micro,0.0703125
            prec_weighted,0.047204446832196136
                  ...
            prec_by_class,"[0.         0.09278351 0.         0.         0.05747126 0.08133971
                0.03971119 0.18181818 0.07194245 0.0877193  0.         0.
                0.        ]"
              ...
            recall_macro,0.07039151324865611
            recall_micro,0.0703125
        
        create a dataframe with the series' index as column
        names. However, for list-valued entries, expand,
        giving each list element its own column. Col names
        for those expansions will be <SPECIES_ABBREV>_prec
        or <SPECIES_ABBREV>_rec, as appropriate.
        
        :param measure_series:
        :type measure_series:
        :return: dataframe with no special index, and
           new column names
        :rtype: pd.DataFrame
        '''
        col_names = []
        col_vals  = []
        for meas_nm in measure_series.index:
            if type(measure_series[meas_nm]) != np.ndarray:
                col_names.append(meas_nm)
                col_vals.append(measure_series[meas_nm])
                continue
            
            # Have array-valued entry like prec_per_class:
            list_values = measure_series[meas_nm]
            
            # Shorten 'prec_by_class' and 'recall_by_class'
            # to 'prec' and 'rec', b/c we will append the
            # species name:
            if meas_nm == 'prec_by_class':
                col_nm_prefix = 'prec'
            elif meas_nm == 'recall_by_class':
                col_nm_prefix = 'rec'
            else:
                col_nm_prefix = meas_nm
                
            for class_id, class_nm in enumerate(self.class_names):
                col_names.append(f"{col_nm_prefix}_{class_nm}")
                col_vals.append(list_values[class_id])
        
        df = pd.DataFrame([col_vals], columns=col_names)
        return df

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
                        type=int,
                        help='batch size to use; default: 1',
                        default=1
                        )
    parser.add_argument('-d', '--device',
                        type=int,
                        nargs='+',
                        help='device number of GPU(s) (zero-origin); repeatable; only used if GPU available; defaults to ID 0',
                        default=0
                        )
    parser.add_argument('-s', '--sampling',
                        type=int,
                        help='optional: only use given percentage of samples in each class; default: all',
                        default=None 
                        )
    
    parser.add_argument('--model_paths',
                        required=True,
                        nargs='+',
                        help='path(s) to the saved Pytorch model(s); repeatable, if more than one, composites of results from all models. ',
                        default=None)
    parser.add_argument('--samples_path',
                        required=True,
                        help='path to samples to run through model',
                        default=None)

    args = parser.parse_args();

    if type(args.device) != list:
        args.device = [args.device]
    
    # Expand Unix wildcards, tilde, and env 
    # vars in the model paths:
    
    if args.model_paths is None:
        print("Must provide --model_paths: path(s) to models over which to inference")
        sys.exit(1)
    
    if type(args.model_paths) != list:
        model_paths_raw = [args.model_paths]
    else:
        model_paths_raw = args.model_paths
        
    model_paths = []
    # For each model path, determine whether 
    # it is a dir, has tilde or $HOME, etc.
    # and create a flat list of full paths
    # to models:
    
    for fname in model_paths_raw:
        # If fname is a file name (i.e. in our context has an extension):
        if len(Path(fname).suffix) > 0:
            # Resolve tilde and env vars:
            model_paths.append(os.path.expanduser(os.path.expandvars(fname)))
            continue
        # fname is a directory:
        for root, dirs, files in os.walk(fname):
            # For files in this dir, expand them,...
            expanded_fnames = [os.path.expanduser(os.path.expandvars(one_fname))
                                for one_fname
                                in files
                                ]
            #... and prepend the root before adding
            # as a full model path:
            model_paths.extend([os.path.join(root, fname)
                                for fname in
                                expanded_fnames
                                ])

    # Ensure the all model paths exist:
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Cannot find model {model_path}")
            sys.exit(1)

    # Same for samples path, though we only allow
    # one of those paths.
    if args.samples_path is None:
        print(f"Must provide --samples_path: directory with subdirs of spectrogram snippets")
        sys.exit(1)
    samples_path = os.path.expanduser(os.path.expandvars(args.samples_path))
    if not os.path.exists(samples_path):
        print(f"Samples path {samples_path} does not exist")
        sys.exit(1)
    
    # Ensure that the file arrangements are as required by
    # the ImageFolder class: 
    #                        <root_dir>
    #        img_folder_1   img_folder_2     ...   img_folder_n
    #         img_file        img_file                  img_file
    #         img_file        img_file                  img_file
    #                   ...                  ...

    dir_struct_desc = f"Samples must be in *sub*directories with image files under {samples_path}"
    for root, dirs, _files in os.walk(samples_path):
        if len(dirs) == 0:
            # No subdirectories:
            print(dir_struct_desc)
            sys.exit(1)
        # Go one level deeper to ensure
        # there are image files in the first-tier subdirs:
        for subdir in dirs:
            full_subdir_path = os.path.join(root, subdir)
            files = os.listdir(full_subdir_path)
            # Find at least one image file to be
            # satisfied:
            found_at_least_one_img = False
            for maybe_img_fname in files:
                if Path(maybe_img_fname).suffix in FileUtils.IMG_EXTENSIONS:
                    found_at_least_one_img = True
                    break
            if not found_at_least_one_img:
                print(f"Subdirectory '{subdir}' does not include any image files.")
                print(dir_struct_desc)
                sys.exit(1)
        # No need to walk deeper:
        break

    inferencer = Inferencer(model_paths,
                            samples_path,
                            labels_path=None,
                            gpu_ids=args.device if type(args.device) == list else [args.device],
                            sampling=args.sampling,
                            batch_size=args.batch_size
                            )
    result_collection = inferencer.go()
    #*************
    input("Hit Enter to quit")
    #print(f"Result collection: {result_collection}")
    #*************
