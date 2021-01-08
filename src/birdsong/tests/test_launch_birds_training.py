'''
Created on Jan 8, 2021

@author: paepcke
'''
from _collections import OrderedDict
import os
import unittest

from birdsong.launch_birds_training import TrainScriptLauncher
from logging_service import LoggingService


TEST_ALL = True
#TEST_ALL = False

class TestLauncher(unittest.TestCase):

    def setUp(self):
        self.curr_dir = os.path.dirname(__file__)


    def tearDown(self):
        pass

    #------------------------------------
    # test_read_world_map 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_read_world_map(self):
        
        launcher = TrainScriptLauncher(unittesting=True)
        tst_world_map_path = os.path.join(self.curr_dir,
                                          'world_map_for_testing.json'
                                          )
        world_map = launcher.read_world_map(tst_world_map_path)
        
        expected = {'quintus.stanford.edu' : {
                            "master" : 'Yes',
                            "foo"    : 1,
                            "gpus"   : 2
                            },
                    'quatro.stanford.edu'  : {
                            "gpus" : 2,
                            "devices" : [1,2]
                            }
                    }
        self.assertDictEqual(world_map, expected)

    #------------------------------------
    # test_training_script_start_cmd 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_training_script_start_cmd(self):
        
        launcher = TrainScriptLauncher(unittesting=True)

        launcher.log = LoggingService()
        config_path = os.path.join(self.curr_dir, '../../../config.cfg')
        rank = 0
        local_rank = 0
        launcher.WORLD_SIZE = 4
        script_args = {'MASTER_ADDR' : '127.0.0.1',
                       'MASTER_PORT' : 5678,
                       'RANK'        : rank,
                       'LOCAL_RANK'  : local_rank,
                       'WORLD_SIZE'  : 4,
                       'config'      : config_path 
                       }
        
        script_path = os.path.join(self.curr_dir, '../birds_train_parallel')
        launch_args = {'training_script' : script_path}
        
        
        start_cmd = launcher.training_script_start_cmd(rank, 
                                                       local_rank,
                                                       launch_args,
                                                       script_args
                                                       )
                                                       
        print(start_cmd)
        

    #------------------------------------
    # test_build_gpu_landscape 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_build_gpu_landscape(self):
        
        launcher = TrainScriptLauncher(unittesting=True)
        
        # Pretend we are on one of the hostnames
        # that are mentioned in the testing world map:
        launcher.hostname = 'quintus.stanford.edu'

        
        tst_world_map_path = os.path.join(self.curr_dir,
                                          'world_map_for_testing.json'
                                          )
        world_map = launcher.read_world_map(tst_world_map_path)
        
        gpu_landscape = launcher.build_compute_landcape(world_map)
        
        expected = OrderedDict([('quatro.stanford.edu', 
                                 {'num_gpus': 2, 
                                  'gpu_device_ids': [1, 2], 
                                  'rank_range': [2, 3]}), 
                                ('quintus.stanford.edu', 
                                 {'num_gpus': 2, 
                                  'gpu_device_ids': [0, 1], 
                                  'rank_range': [0, 1]})])

        self.assertDictEqual(gpu_landscape, expected)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()