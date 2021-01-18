'''
Created on Jan 8, 2021

@author: paepcke
'''
from _collections import OrderedDict
import os
import unittest
import socket 

from logging_service import LoggingService

from birdsong.launch_birds_training import TrainScriptLauncher
from birdsong.utils.dottable_config import DottableConfigParser


TEST_ALL = True
#TEST_ALL = False

class TestLauncher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_host_name = socket.getfqdn()

    def setUp(self):
        self.curr_dir    = os.path.dirname(__file__)
        self.config_path = os.path.join(self.curr_dir, 'bird_trainer_tst.cfg')
        self.config      = DottableConfigParser(self.config_path)
        
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
        tst_world_map_path = os.path.join(self.curr_dir,
                                          'world_map_for_testing.json'
                                          )
        world_map = launcher.read_world_map(tst_world_map_path)
        world_map = self.substitute_local_hostname(world_map)
        launcher.config = self.config
        launcher.build_compute_landscape(world_map)
        
        rank = 0
        local_rank = 0
        min_rank_this_machine = 0
        gpus_to_use = 2 # The entry of this machine in the world map
        
        # Set some instance vars that would
        # normally be set in the the launcher's
        # constructor:
        
        launcher.WORLD_SIZE = 4
        launcher.MASTER_PORT = self.config.getint('Parallelism', 'master_port')
        launcher.log = LoggingService()
        
        script_args = {'MASTER_ADDR' : '127.0.0.1',
                       'MASTER_PORT' : 5678,
                       'RANK'        : rank,
                       'LOCAL_RANK'  : local_rank,
                       'WORLD_SIZE'  : 4,
                       'MIN_RANK_THIS_MACHINE' : min_rank_this_machine,
                       'config'      : self.config_path 
                       }
        
        script_path = os.path.join(self.curr_dir, '../birds_train_parallel')
        launch_args = {'training_script' : script_path}
        
        
        start_cmd = launcher.training_script_start_cmd(rank, 
                                                       local_rank,
                                                       gpus_to_use,
                                                       min_rank_this_machine,
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
        
        gpu_landscape = launcher.build_compute_landscape(world_map)
        
        expected = OrderedDict([('quatro.stanford.edu', 
                                 {'num_gpus': 2, 
                                  'gpu_device_ids': [1, 2], 
                                  'rank_range': [2, 3]}), 
                                ('quintus.stanford.edu', 
                                 {'num_gpus': 2, 
                                  'gpu_device_ids': [0, 1], 
                                  'rank_range': [0, 1]})])

        self.assertDictEqual(gpu_landscape, expected)

# ------------------ Utils ---------------

    def substitute_local_hostname(self, world_map):
        '''
        Replace the hard-coded name 'quintus.stanford.edu'
        in the test world map with the name of the current
        machine. Else some of the methods being tested will
        complain that the local machine is not represented
        in the map.
        
        @param world_map: a world_map that contains the
            hard coded key 'quintus.stanford.edu'
        @type world_map: {str : {str : Any}}
        @return: modified world map
        @rtype: {str : {str : Any}}
        '''
        
        if self.full_host_name == 'quintus.stanford.edu':
            # We happen to be on the quintus machine...
            return world_map
        world_map[self.full_host_name] = world_map['quintus.stanford.edu']
        del world_map['quintus.stanford.edu']
        return world_map
        

# ------------------ Main ---------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
