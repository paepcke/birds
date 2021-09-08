'''
Created on Sep 4, 2021

@author: paepcke
'''
import os
from pathlib import Path
import shutil
import unittest

from data_augmentation.recording_length_inventory import RecordingsInventory


TEST_ALL = True
#TEST_ALL = False


class RecortingsInventoryTester(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        
        cls.cur_dir = os.path.dirname(__file__)
        cls.tst_data = os.path.join(cls.cur_dir, 'data/augmentation_tst_data/')
        
        tst_data_path = Path(cls.tst_data)
        # Path to manifest directory that will
        # be created:
        cls.manifest_dir = str(Path(cls.tst_data)
                               .parent
                               .joinpath(Path(f"Audio_Manifest_{tst_data_path.stem}")))

    def setUp(self):
        pass


    def tearDown(self):
        shutil.rmtree(self.manifest_dir, ignore_errors=True)

# ---------------------- Tests ---------------------

    #------------------------------------
    # test_constructor 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_constructor(self):
        
        ri_obj = RecordingsInventory(self.tst_data, 
                                     message=None, 
                                     chart_result=False)
        self.assertEqual(ri_obj.manifest_dir_path, self.manifest_dir)

        # Check whether the files in the manifest directory
        # exist:
        
        # Manifest must exist and be non-empty:
        manifest_dir_p = Path(self.manifest_dir)
        manifest_file_p = manifest_dir_p.joinpath('manifest.json')
        self.assertGreater(manifest_file_p.stat().st_size, 0)
        
        # Get the inventory, and make a new RecordingsInventory
        # instance; this time including a message and the precomputed
        # inventory. Also, ask for a bar chart that should show
        # up in the inventory dir:
        
        inventory = ri_obj.inventory
        _ri_obj1   = RecordingsInventory(self.tst_data, 
                                         message="A test message", 
                                         chart_result=True,
                                         inventory=inventory)
        # Manifest file should still be there:
        self.assertGreater(manifest_file_p.stat().st_size, 0)
        
        # README file:
        with open(manifest_dir_p.joinpath('README.txt'), 'r') as fd:
            msg_retrieved = fd.read()
            self.assertEqual(msg_retrieved, "A test message")
            
        # And a .pdf file with the barchart should be there now:
        barchart_file_p = manifest_dir_p.joinpath('audio_recording_distribution.pdf')
        self.assertGreater(barchart_file_p.stat().st_size, 0)

        print('foo')
# --------------------- Main ---------------------

if __name__ == "__main__":
    
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()