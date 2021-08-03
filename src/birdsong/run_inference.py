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
from multiprocessing import Pool
import os
from pathlib import Path
import sys

from logging_service.logging_service import LoggingService
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, classification_report 
from skorch import NeuralNet
from torch.utils.data.dataloader import DataLoader

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
import torch
from result_analysis.charting import Charter, CELL_LABELING, CurveSpecification
#import traceback as tb


class Inferencer:
    '''
    classdocs
    '''

    REPORT_EVERY = 50
    #REPORT_EVERY = 4
    '''How many inferences to make before writing to tensorboard and disk'''

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

        self.log.info(f"Tensorboard info will be written to {tensorboard_dest}")
        self.log.info(f"Result measurement CSV file(s) will be written to {csv_path}")
        
    #------------------------------------
    # __call__ 
    #-------------------
    
    def __call__(self, gpu_id_model_path_pair):
        gpu_id, self.model_path = gpu_id_model_path_pair
        self.prep_model_inference(self.model_path)
        self.log.info(f"Beginning inference with model {FileUtils.ellipsed_file_path(self.model_path)} on gpu_id {gpu_id}")
        #****************
        return self.run_inference(gpu_to_use=gpu_id)
        # Parallelism:
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
    # run_inference 
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

        res_coll = ResultCollection()
        display_counter = 0
        # For keeping track what's been written to disk:
        start_idx = 0
        for batch_num, (X,y) in enumerate(self.loader):
            outputs = sko_net.predict(X)
            tally = ResultTally(batch_num,
                                LearningPhase.TESTING,
                                torch.tensor(outputs),
                                y,  # Labels
                                torch.tensor(0),  # Don't track loss
                                self.class_names,
                                self.batch_size
                                )
            res_coll.add(tally, batch_num)

            if batch_num > 0 and batch_num % self.REPORT_EVERY == 0:
                tmp_coll = ResultCollection(res_coll[start_idx:])
                self.report_intermittent_results(tmp_coll,
                                                 display_counter=display_counter) # Overlay the charts in tensorboard
                start_idx += len(res_coll) - start_idx
                display_counter += 1

            #************
            #if batch_num >= 7:
            #    break
            #************
        try:
            self.report_results(res_coll)
        finally:
            self.writer.close()

    #------------------------------------
    # report_intermittent_results 
    #-------------------
    
    def report_intermittent_results(self, 
                                    res_coll, 
                                    show_in_tensorboard=True,
                                    display_counter=0):
        '''
        Writes a portion of the given tally collection to disk.
        Updates tensorboard with bar charts of IR values.
        Each call results in an updated bar chart that reflects
        the entire collection. Tenforboard will overlay them, and
        provide a slider.
        
        The start_idx/end_idx define a slice of tallies whose
        (partial) content is added to the csv file manage by
        self.csv_writer. 
        
        :param res_coll: collection of result tallies
        :type res_coll: ResultCollection
        :param show_in_tensorboard: whether or not to transmit
            the computed barchart to tensorboard.
        :type show_in_tensorboard: bool
        :param display_counter: if calling this method repeatedly, incrementing
            this counter causes tensorflow to create a slider for overlays
        :type display_counter: int
            
        '''

        # Update the precision/recall/accuracy bar chart:
        
        # Create one long Pandas series from all 
        # predictions, and another from all the corresponding
        # truths:
        
        preds_arr  = res_coll.flattened_predictions(phase=LearningPhase.TESTING)
        labels_arr = res_coll.flattened_labels(phase=LearningPhase.TESTING)

        res_dict   = self._compute_ir_values(preds_arr, labels_arr)
        
        res_series = pd.Series(res_dict)

        # Specify bar colors for related quantities:
        color_groups = {
            'steelblue' : ['prec_macro', 'prec_micro', 'prec_weighted'],
            'teal': ['recall_macro', 'recall_micro', 'recall_weighted'],
            'slategrey' : ['f1_macro', 'f1_micro', 'f1_weighted'],
            'saddlebrown' : ['accuracy', 'balanced_accuracy']
            }
        ax = Charter.barchart(res_series, 
                              rotation=45,
                              ylabel='Performance Measure', 
                              color_groups=color_groups)
        
        if show_in_tensorboard:
            self.writer.add_figure('Performance Measures', 
                                   ax.get_figure(),
                                   global_step=display_counter)


        # Now add just the new preds/labels to the CSV file:
        self._write_probability_rows(res_coll)
        
 
    #------------------------------------
    # _write_probability_rows 
    #-------------------
    
    def _write_probability_rows(self, tally_coll):
        '''
        Given a list of TallyResult, extract the probabilites
        and labels. Write them to .csv. Each tally contains
        results from one batch of inference results. Thus
        we have one list of probabilities of length num_classes
        for each inference result in that batch:
        
                                 NumClasses
                     
        res0    prob_0_class0  prob_0_class1, ... prob_0_class_(num_classes - 1)  
        res1    prob_1_class0  prob_1_class1, ... prob_1_class_(num_classes - 1)  
        res3               ...
        res_(batch_size - 1)
        
        Assumption: self.csv_writer contains an open csv.Writer instance
                    with header already written:
                    
                        Class0,Class1,...Classn,Label
        
        :param tally_coll: list of results from batches
        :type tally_coll: ResultCollection
        '''

        # Get the probabilities for each class for
        # the entire batch for each tally as a list of data frames:
        probs_tns_list = [pd.DataFrame(tally.probs, columns=tally.class_names)
                          for tally
                          in tally_coll
                          ]
        
        # Get a list of label arrays. Each arr
        # is of length batch_size:
        labels_arr = [tally.labels
                     for tally
                     in tally_coll
                     ]
        
        for batch_results, truths in zip(probs_tns_list, labels_arr):
            # Each batch_results is a dataframe of
            # probability results. Each row is the probs
            # for each class of one sample. Like this:
            #
            #              BANA       PATY    ...      YCEU
            #  sample0    prob_0_0  prob_0_1  ...  prob_0_numClasses 
            #  sample1             ...
            #
            # Where each prob is a one-element float tensor.
            # The df shape is (batch_size, num_classes).
            # Each truths is an arr of length batch_size.
             
            # Add the truth labels as a column on the right
            # of the df:
            batch_results = batch_results.astype(float)
            batch_results['label'] = pd.Series(truths)

            # Write rows to CSV as floats:
        
            for i in range(len(batch_results)):
                one_sample_probs = batch_results.iloc[i,:].values
                
                # Have a pd.Series  with the probabilities of 
                # one sample for each class. Values are tensors
                # of individual floats.
                self.csv_writer.writerow(one_sample_probs)

    #------------------------------------
    # report_results 
    #-------------------
    
    def report_results(self, result_coll):
        predicted_classes = result_coll.flattened_predictions(phase=LearningPhase.TESTING)
        truths = result_coll.flattened_labels(phase=LearningPhase.TESTING)
        
        # Confusion matrix to tensorboard and local figure, if possible:
        conf_matrix_fig = self._report_conf_matrix(truths, 
                                                   predicted_classes, 
                                                   show_in_tensorboard=True)
        self.log.info(f"Saving confusion matrix fig 'inference_conf_matrix.pdf' in {self.csv_dir}")
        conf_matrix_fig.savefig(os.path.join(self.csv_dir, 'inference_conf_matrix.pdf'),
                                format='pdf',
                                dpi=150
                                )

        pred_probs_df = result_coll.flattened_probabilities(phase=LearningPhase.TESTING)
        # Get mean average precision and the number
        # of classes from which the value was computed.
        # I.e. the classes for which AP was not NaN:
        (mAP, num_classes), pr_curve_fig = self._report_charted_results(truths, 
                                                                        pred_probs_df, 
                                                                        show_in_tensorboard=True
                                                                        )
        
        # Save pr_curve figure:
        self.log.info(f"Saving PR curve pr_curve.pdf in {self.csv_dir}")
        pr_curve_fig.savefig(os.path.join(self.csv_dir, 'pr_curve.pdf'),
                             format='pdf',
                             dpi=150
                             )

        # Text tables results to tensorboard:
        self._report_textual_results(predicted_classes, 
                                     truths, 
                                     (mAP,num_classes), 
                                     self.csv_dir)
        


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

        self.log.info(f"Writing conf_matrix.csv to {self.csv_dir}")
        dst_path = os.path.join(self.csv_dir, 'conf_matrix.csv')
        conf_matrix.to_csv(dst_path, index_label='True Class')

        fig = Charter.fig_black_white_from_conf_matrix(conf_matrix)
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
    
    def _report_charted_results(self, 
                                truth_series, 
                                pred_probs_df, 
                                show_in_tensorboard=True):
        '''
        Computes a set of precision-recall curves in one plot. 
        Writes figure to tensorboard writer in self.writer.
        Returns the mean average precision, and the figure. 
        
        :param truth_series: truth labels
        :type truth_series: pd.Series
        :param pred_probs_df: probability for each class 
            for each sample 
        :type pred_probs_df: pd.DataFrame
        :param show_in_tensorboard: whether or not to post
            the computed figure to Tensorboard
        :type show_in_tensorboard: bool
        :return: a tuple and the figure. The tuple holds the
            mAP, and the number of classes for which the avg prec
            (AP) was well defined. I.e. that were included in the
            mAP calculation
        '''

        # Obtain a dict of CurveSpecification instances,
        # one for each class, plus the mean Average Precision
        # across all curves. The dict will be keyed
        # by class ID:
        
        mAP_and_num_APs, pr_curves = Charter.visualize_testing_result(truth_series, pred_probs_df)
        
        # Separate out the curves without 
        # ill defined prec, rec, or f1. Prec and
        # rec should have none, b/c NaNs were interpolated
        # out earlier.
        well_defined_curves = list(filter(CurveSpecification.well_defined, pr_curves.values())) 
        
        if len(well_defined_curves) == 0:
            self.log.warn(f"For all thresholds, one or more of precision, recall or f1 are undefined. No p/r curves to show")
            return False
        
        # Too many curves are clutter. Only
        # show the best, worst, and median by optimal f1;
        f1_sorted_curves = sorted(well_defined_curves,
                           key=lambda curve: curve['best_op_pt']['f1']
                           )
        curves_to_show = self.pick_pr_curve_classes(f1_sorted_curves)

        fig = ClassificationPlotter.chart_pr_curves(curves_to_show, self.class_names)

        if show_in_tensorboard:
            self.writer.add_figure('Precision Recall Curves for Selected Classes', 
                                   fig,
                                   global_step=0)

        return mAP_and_num_APs, fig

    #------------------------------------
    # _report_textual_results 
    #-------------------


    def _report_textual_results(self, 
                                all_preds, 
                                all_labels, 
                                mAP_info, 
                                res_dir):
        '''
        Given sequences of predicted class IDs and 
        corresponding truth labels, computes information 
        retrieval type values:

             precision (macro/micro/weighted/by-class)
             recall    (macro/micro/weighted/by-class)
             f1        (macro/micro/weighted/by-class)
             acuracy
             balanced_accuracy
             mAP       # the mean average precision
             number of well_defined_APs
             number of classes
             
        The well_defined_APs is the number of classes for
        which average precision (AP) was not NaN, i.e. the 
        number of classes included in the mAP
        
        Writes these results into three .csv files:

           <res_dir>/accuracy_mAP.csv
           <res_dir>/performance_per_class.csv
           <res_dir>/ir_results.csv
        
        Finally, constructs Github flavored tables from the
        above results, and posts them to the 'text' tab of 
        tensorboard.

        :param all_preds: list of all predictions made
        :type all_preds: [int]
        :param all_labels: list of all truth labels
        :type all_labels: [int]
        :param mAP_info: the previously calculated mean average
            precision, and the number of classes from which the
            quantity was calculated
        :type mAP: (float, int)
        :param res_dir: directory where all .csv and other 
            result files are to be written
        :type res_dir: str
        '''
        
        res = self._compute_ir_values(all_preds, all_labels)
        
        # Get a Series looks like:
        #    prec_macro,0.04713735383721669
        #    prec_micro,0.0703125
        #    prec_weighted,0.047204446832196136
        #    recall_macro,0.07039151324865611
        #    recall_micro,0.0703125
        
        res_series = pd.Series(list(res.values()),
                               index=list(res.keys())
                               )
        
        res_df = self.res_measures_to_df(res_series)
        
        # Add the mAP and well_defined_APs values:
        mAP, num_well_defined_APs = mAP_info
        res_df['mAP'] = mAP
        res_df['well_defined_APs'] = num_well_defined_APs
        res_df['num_classes_total'] = len(self.class_names)
        
        # Write to csv file:
        self._ir_tbl_to_csv(res_dir, 'ir_results.csv', res_df)

        # Get the per-class prec/rec/f1/support:
        per_class_report = classification_report(all_labels, 
                                                 all_preds, 
                                                 output_dict=True,
                                                 zero_division=0)
        # Turn into a df:
        #
        #             precision  recall  f1-score  support
        #    label 1        0.5     1.0      0.67      1.0
        #    label 2        0.6     0.2      0.88      1.0
        #          ...                    ...
        ir_per_cl_df = pd.DataFrame.from_dict(per_class_report)
        # Have prec/rec/f1/support as rows/ want those as cols:
        ir_per_cl_df = ir_per_cl_df.T
        # Drop the accuracy and averages we already covered above:
        ir_per_cl_df = ir_per_cl_df.drop(['accuracy', 'macro avg', 'weighted avg'])
        # Replace the numeric species class IDs with
        # their species. Note that the df may not contain
        # data for all species, so we cannot count on using
        # range():
        new_row_labels = []
        for index_entry in ir_per_cl_df.index:
            new_row_labels.append(self.class_names[int(index_entry)])
        ir_per_cl_df.index = new_row_labels

        # Write perfomance per class to file:
        self._ir_tbl_to_csv(res_dir, 
                            'performance_per_class.csv', 
                            ir_per_cl_df,
                            index_label='species')


        res_rnd = {}
        for meas_nm, meas_val in res.items():
            
            # Measure results are either floats (precision, recall, etc.),
            # Round each measure to one digit:
            res_rnd[meas_nm] = round(meas_val,2)

        # Next, construct Tensorboard markup tables
        # for the same results:

        ir_measures_skel = {'col_header' : ['precision', 'recall', 'f1'], 
                            'row_labels' : ['macro','micro','weighted'],
                            'rows'       : [[res_rnd['prec_macro'],    res_rnd['recall_macro'],    res_rnd['f1_macro']],
                                            [res_rnd['prec_micro'],    res_rnd['recall_micro'],    res_rnd['f1_micro']],
                                            [res_rnd['prec_weighted'], res_rnd['recall_weighted'], res_rnd['f1_weighted']]
                                           ]  
                           }
        
        ir_by_class_skel =  {'col_header' : ir_per_cl_df.columns.to_list(),
                             'row_labels' : ir_per_cl_df.index.to_list(),
                             'rows'       : round(ir_per_cl_df,2).values.tolist()
                             }
        
        acc_mAP_df = pd.DataFrame.from_records([{
            'accuracy' : res_rnd['accuracy'],
            'balanced_accuracy' : res_rnd['balanced_accuracy'],
            'mAP' : round(mAP,2),
            'classes_in_mAP' : num_well_defined_APs,
            'num_classes'    : len(self.class_names)
            }]) 

        self._ir_tbl_to_csv(res_dir, 'accuracy_mAP.csv', acc_mAP_df)
        
        accuracy_skel = {'col_header' : ['accuracy', 
                                         'balanced_accuracy', 
                                         'mean avg precision (mAP)',
                                         ],
                         'row_labels' : ['Overall'],
                         'rows'       : [[acc_mAP_df['accuracy'].item(), 
                                          acc_mAP_df['balanced_accuracy'].item(), 
                                          f"{round(mAP,2)} from {num_well_defined_APs}/{len(self.class_names)} classes"
                                          ]]
                         }

        ir_measures_tbl  = GithubTableMaker.make_table(ir_measures_skel, sep_lines=False)
        ir_by_class_tbl  = GithubTableMaker.make_table(ir_by_class_skel, sep_lines=False)
        accuracy_tbl     = GithubTableMaker.make_table(accuracy_skel, sep_lines=False)
        
        self.log.info("Writing IR measures to tensorflow Text tab")
        
        # Write the markup tables to Tensorboard:
        self.writer.add_text('Information retrieval measures', ir_measures_tbl, global_step=0)
        self.writer.add_text('Per class measures', ir_by_class_tbl, global_step=0)
        self.writer.add_text('Accuracy', accuracy_tbl, global_step=0)
        
        return res_series

    #------------------------------------
    # _ir_tbl_to_csv 
    #-------------------
    
    def _ir_tbl_to_csv(self, res_dir, fname, df, index_label=None):
        '''
        Check whether fname exists in res_dir. If
        so, append the dataframe df's values to the
        file. Else, create the file, and write the df,
        using columns as header.
        
        :param res_dir: destination directory
        :type res_dir: str
        :param fname: destination file name
        :type fname: str
        :param df: dataframe with column names set
        :type df: pd.DataFrame
        :param index_label: if the df's row labels are
            to be included, set index_label to the column
            name to use for those labels. If None, no
            row labels are included
        :type index_label: {None | str}
        '''

        res_csv_path = os.path.join(res_dir, fname)
        
        # Should we append or create a new file?
        try:
            if os.path.getsize(res_csv_path) > 0:
                append_to_csv = True
        except FileNotFoundError:
                append_to_csv = False
        else:
            append_to_csv = False
            
        if index_label is None:
            include_index = False
        else:
            include_index = True
        
        self.log.info(f"Writing {fname} to {res_dir}")

        if append_to_csv:
            df.to_csv(res_csv_path, 
                      index=include_index,
                      index_label=index_label, 
                      mode='a', 
                      header=False,
                      )
        else:
            df.to_csv(res_csv_path, 
                      index=include_index,
                      index_label=index_label, 
                      mode='w', 
                      header=True)


    #------------------------------------
    # _compute_ir_values
    #-------------------

    def _compute_ir_values(self, all_preds, all_labels):
        '''
        Given class predictions and labels, compute the
        typical information retrieval values: macro/micro
        precision, recall, etc. Returns a dict with the 

        :param all_preds: series of truth labels
        :type all_preds: pd.Series(int)
        :param all_labels: predictions
        :type all_labels: pd.Series(int)
        :return dict with measures
        :rtype {src : float}
        '''
        with np.errstate(divide='ignore',invalid='ignore'):
            res = OrderedDict({})
            res['prec_macro'] = precision_score(all_labels, all_preds, 
                average='macro', 
                zero_division=0)
            res['prec_micro'] = precision_score(all_labels, 
                all_preds, 
                average='micro', 
                zero_division=0)
            res['prec_weighted'] = precision_score(all_labels, 
                all_preds, 
                average='weighted', 
                zero_division=0)
            res['recall_macro'] = recall_score(all_labels, 
                all_preds, 
                average='macro', 
                zero_division=0)
            res['recall_micro'] = recall_score(all_labels, 
                all_preds, 
                average='micro', 
                zero_division=0)
            res['recall_weighted'] = recall_score(all_labels, 
                all_preds, 
                average='weighted', 
                zero_division=0)
            res['f1_macro'] = f1_score(all_labels, 
                all_preds, 
                average='macro', 
                zero_division=0)
            res['f1_micro'] = f1_score(all_labels, 
                all_preds, 
                average='micro', 
                zero_division=0)
            res['f1_weighted'] = f1_score(all_labels, 
                all_preds, 
                average='weighted', 
                zero_division=0)
            res['accuracy'] = accuracy_score(all_labels, all_preds)
            res['balanced_accuracy'] = balanced_accuracy_score(all_labels, all_preds)
        return res

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
        
        # Only consider curves that have at least
        # the median in the number of curve points.
        # Else we end up with curves of only 2 points,
        # or such (num of recalls and precisions points
        # are the same, of course):
        
        crv_lengths = [len(cs['recalls']) for cs in f1_sorted_curves]
        # Veer on the 'more-points-is-better' side by
        # rounding up:
        med_len = round(np.median(crv_lengths) + 0.5)
        
        # Keep only the longer point series, and the curves
        # where the average precision is not extreme:
        crv_candidates = list(filter(lambda crv, med_len=med_len: 
                                          len(crv['recalls']) >= med_len and \
                                          crv['avg_prec'] > 0.0 and \
                                          crv['avg_prec'] < 1.0,
                                     f1_sorted_curves))

        # Since curves are f1-sorted, easily ID
        # the lowest, highest, and median:
        min_best_f1_crv = crv_candidates[0]
        max_best_f1_crv = crv_candidates[-1]
        med_f1_crv      = crv_candidates[len(crv_candidates) // 2]
        
        # The corresponding f1 values at the best
        # operating point for each of the three curves:
        min_f1 = min_best_f1_crv['best_op_pt']['f1']
        max_f1 = max_best_f1_crv['best_op_pt']['f1']

        med_f1 = med_f1_crv['best_op_pt']['f1']
        
        # Only be happy if the median isn't identical
        # to the min or max:
        if med_f1 not in (min_f1, max_f1):
            return [min_best_f1_crv, med_f1_crv, max_best_f1_crv] 
        
        # Median of the best op points' f1 scores is
        # same as one of the extremes:
        
        f1_scores = [crv['best_op_pt']['f1'] 
                     for crv in crv_candidates]
        med_f1_idx = crv_candidates.index(med_f1_crv)
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
                crv_candidates[middle_idx],
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
    #input("Hit Enter to quit")
    #print(f"Result collection: {result_collection}")
    #*************
