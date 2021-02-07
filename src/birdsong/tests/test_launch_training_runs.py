'''
Created on Feb 5, 2021

@author: paepcke
'''
import os
import socket
import unittest

import numpy as np

from birdsong.launch_training_runs import TrainScriptRunner
from birdsong.utils.neural_net_config import NeuralNetConfig


#*****TEST_ALL = True
TEST_ALL = False

class Test(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        super(Test, cls).setUpClass()

        curr_dir = os.path.dirname(__file__)
        conf_path = os.path.join(curr_dir,
                                 'bird_trainer_tst.cfg'
                                 )
        cls.initial_config = NeuralNetConfig(conf_path)

        cls.hparms_spec = {'lr' : [0.001],
                           'optimizer'  : ['Adam', 'RMSprop', 'SGD'],
                           'batch_size' : [32,64,128],
                           'kernel'     : [3,7]
                           }
        cls.this_hostname = socket.getfqdn()

    def setUp(self):

        self.launcher = TrainScriptRunner(self.hparms_spec, 
                                         self.initial_config,
                                         unittesting=True)

    def tearDown(self):
        pass

# --------------------- Tests ------------
#*********
#         self.gpu_landscape = self.obtain_world_map(starting_config)
#         
#         the_run_dicts   = self.get_runs_hparm_specs(hparms_spec)
#         the_run_configs = self.gen_configurations(starting_config,
#********* 

    #------------------------------------
    # test_obtain_world_map 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_obtain_world_map(self):
        
        gpu_landscape = self.launcher.obtain_world_map(self.initial_config)
        # Because this test must be able to
        # run on all kinds of machines with and
        # w/o GPUs, we cannot test much:
        
        if not self.this_hostname in gpu_landscape.keys():
            self.assertFalse("Must have an entry for this hostname in <proj-root>.world_map.json")

        world_entry = gpu_landscape[self.this_hostname]
        self.assertIn('num_gpus', world_entry.keys())
        self.assertIn('gpu_device_ids', world_entry.keys())
        

    #------------------------------------
    # test_get_runs_hparm_specs 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_get_runs_hparm_specs(self):
        
        run_dicts = self.launcher.get_runs_hparm_specs(self.hparms_spec)
        
        num_combinations = self.get_num_of_combos()
        
        self.assertEqual(len(run_dicts), num_combinations)
        print(run_dicts)

    #------------------------------------
    # test_gen_configurations
    #-------------------
    
    #****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_gen_configurations(self):
        
        config_dicts = self.launcher.get_runs_hparm_specs(self.hparms_spec)
        config_list = self.launcher.gen_configurations(self.initial_config, 
                                                       config_dicts)
        
        num_combinations = self.get_num_of_combos()
        self.assertEqual(len(config_list), num_combinations)
        
        for conf in config_list:
            self.assertEqual(conf.sections(),
                             ['Paths', 'Training', 'Parallelism']
                             )
        
        print(config_list)
        
# ------------------ Utils -----------

    def get_num_of_combos(self):
        
        val_len_each_hparm = [len(hparm_vals) 
                              for hparm_vals
                              in self.hparms_spec.values()]
        num_combinations = np.product(np.array(val_len_each_hparm))
        return num_combinations
    
# ------------------ Main ------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()