'''
Created on Sep 4, 2021

@author: paepcke
'''
import os
from pathlib import Path
import unittest

from data_augmentation.recording_length_inventory import RecordingsInventory


TEST_ALL = True
#TEST_ALL = False


class RecortingsInventoryTester(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        
        cls.cur_dir = os.path.dirname(__file__)
        cls.tst_data = os.path.join(cls.cur_dir, 'data/augmentation_tst_data/')

    def setUp(self):
        pass


    def tearDown(self):
        pass

# ---------------------- Tests ---------------------

    #------------------------------------
    # test_constructor 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_constructor(self):
        
        ri_obj = RecordingsInventory(self.tst_data, message=None, chart_result=False)
        tst_data_path = Path(self.tst_data)
        res_dir = Path(self.tst_data).parent.joinpath(Path(f"self.Audio_Manifest_{tst_data_path.stem}"))
        
        print(res_dir)


if __name__ == "__main__":
    
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()