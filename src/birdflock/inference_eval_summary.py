'''
Created on Sep 20, 2021

@author: paepcke
'''
import os
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt

from experiment_manager.experiment_manager import ExperimentManager, Datatype

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
            
        # Get map from species name to experiment root
        # dir name in alpha order by species:
        exp_dir_dict = ExperimentManager.collect_experiment_roots(experiments_root,
                                                                  common_experiments_timestamp
                                                                  )
        
        fig = self._make_balanced_accuracy_chart(exp_dir_dict)
        
    #------------------------------------
    # _make_balanced_accuracy_chart
    #-------------------

    def _make_balanced_accuracy_chart(self, exp_dir_dict):
        
        # Build a df from all the one-line ir_results.csv
        # files---one in each experiment:
        
        all_ir_res_df = pd.DataFrame()
        for species, exp_dir in exp_dir_dict.items():
            dir_path = os.path.join(self.experiments_root, exp_dir)
            exp = ExperimentManager(dir_path)
            # Get a 1-line dataframe with IR results:
            ir_results = exp.read('ir_results', Datatype.tabular)
            # Add a Species column at the front:
            row = ir_results.iloc[0,:]
            species_ser = pd.Series([species], index=['species'])
            res_ser = species_ser.append(row)
            all_ir_res_df.append(res_ser, ignore_index=True)
            all_ir_res_df = all_ir_res_df.append(ir_results)
                                                 

        print(ir_results)

