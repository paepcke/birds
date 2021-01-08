'''
Created on Jan 8, 2021

@author: paepcke
'''
import unittest
import os

from birdsong.launch_birds_training import TrainScriptLauncher

TEST_ALL = True
# TEST_ALL = False

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


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()