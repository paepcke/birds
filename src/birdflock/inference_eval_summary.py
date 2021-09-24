'''
Created on Sep 20, 2021

@author: paepcke
'''
import argparse
import os
from pathlib import Path
import sys

from experiment_manager.experiment_manager import ExperimentManager, Datatype

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from result_analysis.charting import Charter


class BinaryInferenceEvaluator:
    '''
    classdocs
    '''


    def __init__(self, common_experiments_timestamp, experiments_root=None):
        '''
        Constructor
        '''
        self.cur_dir = os.path.dirname(__file__)
        if experiments_root is None:
            self.experiments_root = os.path.join(self.cur_dir, 'Experiments')
        else:
            self.experiments_root = experiments_root

        # Create an ExperimentManager for all the
        # summarization results:
        summary_exp_path = os.path.join(experiments_root, 
                                        f"InferenceSummary_{common_experiments_timestamp}")
        test_summary_exp = ExperimentManager(summary_exp_path)
            
        # Get map from species name to experiment root
        # dir name in alpha order by species:
        exp_dir_dict = ExperimentManager.collect_experiment_roots(experiments_root,
                                                                  common_experiments_timestamp
                                                                  )
        ir_results = self._collect_ir_results(exp_dir_dict)
#******
        # Barchart of all balanced accuracies:
        fig_balanced_acc = self._make_balanced_accuracy_chart(ir_results)
        test_summary_exp.save('balanced_accs', fig_balanced_acc)
#******        
        
        # ***** MAYBE A GRID OF IR CHARTS?
        # MAYBE THE EXTREMES (TOP 5 and Bottom 5)
        #fig_recall_prec_f1 = self._make_ir_results_panel(ir_results)
        
    #------------------------------------
    # _make_balanced_accuracy_chart
    #-------------------

    def _make_balanced_accuracy_chart(self, ir_results):
        
        # Get a df like:
        #
        #    Species   accuracy    macro_f1  ...
        #     WTROC      0.5         0.6
        #     VASEG      ...         ...
        
        
        balanced_accs = ir_results.balanced_accuracy
        ax = Charter.barchart(balanced_accs, 
                              ylabel='Balanced Accuracy',
                              rotation=45)
        fig = ax.figure

        return fig

    #------------------------------------
    # _make_ir_results_panel
    #-------------------
    
    def _make_ir_results_panel(self, ir_results):
        
        charts_per_row = 5
        num_charts, _num_measures = ir_results.shape
        num_plot_rows  = int(np.ceil(num_charts / charts_per_row))

        #fig, axes = plt.subplots(nrows=num_plot_rows, ncols=charts_per_row)
        #fig = plt.figure()

        color_groups = {'green' : ['recall', 'prec', 'f1'],
                        'blue'  : ['bal-acc']
                        }
        for species, values_ser in ir_results.iterrows():
            vals_to_plot = pd.Series([
                values_ser['balanced_accuracy'],
                values_ser['recall_macro'],
                values_ser['prec_macro'],
                values_ser['f1_macro']
                ], 
                name=species,
                index = ['bal-acc', 'recall', 'prec', 'f1']
            )
            ax = Charter.barchart(vals_to_plot,
                                  ylabel = species,
                                  rotation=45,
                                  color_groups=color_groups
                                  )
            #**************8
            # if fig is None:
            #     fig = ax.figure
            # fig.axes.append(ax)
            #**************8
                        
        #print(recall)

    #------------------------------------
    # _collect_ir_results
    #-------------------

    def _collect_ir_results(self, exp_dir_dict):
        '''
        Go through every inference experiment, and
        collect the IR-like results, such as balanced accuracy,
        f1, recall, etc.
        
        Return a dataframe like:
        
           Species   accuracy    macro_f1  ...
            WTROC      0.5         0.6
            VASEG      ...         ...

        :param exp_dir_dict: root of all inference experiments
        :type exp_dir_dict: str
        :return information retrieval type results, one row
            for each species
        :rtype pd.DataFrame
        '''

        all_ir_res_df = pd.DataFrame()
        for species, exp_dir in exp_dir_dict.items():
            # Materialize the inference ExperimentManager
            # from the full path of the inference exeriment:
            dir_path = os.path.join(self.experiments_root, exp_dir)
            exp = ExperimentManager(dir_path) # Get a 1-line dataframe with IR results:
            
            # Get all information retrieval related results
            # from the exp manager as a one-line pd.DataFrame
            ir_results = exp.read('ir_results', Datatype.tabular) 
            row = ir_results.iloc[0, :]
            # Create a (future) row labels from the species name:
            row.name = species
            # Add new row:
            all_ir_res_df = all_ir_res_df.append(row)
        
        return all_ir_res_df

# ------------------------ Main ------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Create summary charts from inference experiments."
                                     )

    parser.add_argument('experiment_root',
                        help='directory with Classifier_<species>_<date>_inference subdirectories',
                        default=None)

    args = parser.parse_args()
    
    dir_root = args.experiment_root
    if not os.path.isdir(dir_root):
        print(f"Must pass an existing directory, not {dir_root}")
        sys.exit(1)
    # Count number of inference experiments under
    # experiment_root
    
    all_subdirs = os.listdir(dir_root)
    
    inf_subdirs = []
    for subdir in all_subdirs:
        if subdir.endswith('_inference') \
           and ExperimentManager.timestamp_from_exp_path(subdir) is not None:
            inf_subdirs.append(subdir)
        
    num_inf_exps = len(inf_subdirs)
    if num_inf_exps == 0:
        print(f"No inference experiments under {dir_root}; doing nothing")
        sys.exit(0)
    
    print(f"Number of inference experiments is {len(inf_subdirs)}")
    
    # For now, assume that under experiment_root are
    # only inference experiments that are related by a
    # single datestamp. So, find the stamp from the first
    # inference subdir:
    
    inf_subdir = inf_subdirs[0]
    timestamp = ExperimentManager.timestamp_from_exp_path(inf_subdir)
    
    BinaryInferenceEvaluator(timestamp, dir_root)

    input("Press any key to close the chart: ")
    