'''
Created on May 2, 2021

@author: paepcke

NOTE: this is a pretty slim set of unittests.
      Needs more.

'''
import os
import unittest

from birdsong.rooted_image_dataset import SingleRootImageDataset
from torch.utils.data.dataloader import DataLoader

TEST_ALL = True
#TEST_ALL = False


class TestRootedImageDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir  = os.path.dirname(__file__)
        cls.data_dir = os.path.join(cls.cur_dir, 'data_other/TestSnippets')
        # Count number of AMADECs:
        amadec_root = os.path.join(cls.data_dir, 'AMADEC')
        melrub_root = os.path.join(cls.data_dir, 'MELRUB')
        cls.num_amadecs = len(os.listdir(amadec_root))
        cls.num_melrub  = len(os.listdir(melrub_root))

    def setUp(self):
        pass


    def tearDown(self):
        pass

# ------------------ Tests -------------------

    def test_dataset_setup(self):
        ds = SingleRootImageDataset(self.data_dir)
        self.assertEqual(len(ds), 21)
        
        self.assertEqual(ds.class_names(), ['AMADEC', 'MELRUB'])
        
        self.assertListEqual(list(ds.sample_ids()), list(range(21)))

        # Map from class ID to species name:
        clid2clnm = {cls_id : cls_nm
                     for (cls_id, cls_nm)
                     in zip(ds.class_id_list(), ds.class_names())
                     }
        # Get only a percentage:
        ds1perc = SingleRootImageDataset(self.data_dir, percentage=50)
        
        self.assertEqual(len(ds1perc), 10)
        # Expect sample IDs still to be consecutive ints,
        # not a broken chain:
        self.assertListEqual(list(ds1perc.sample_ids()), list(range(10)))

        dl =  DataLoader(ds1perc,
                         batch_size=1,
                         shuffle=False, 
                         drop_last=True 
                        )
        self.assertEqual(len(dl), 10)
        num_seen_species = {'AMADEC' : 0, 'MELRUB' : 0}
        for _batch_num, (_batch, targets) in enumerate(dl):
            target_species_nm = clid2clnm[targets.item()]
            num_seen_species[target_species_nm] += 1
            
        # Expected AMADECs: 50% of self.num_amadecs:
        expected_amadec = int(self.num_amadecs / 2)
        expected_melrub = int(self.num_melrub / 2)
        self.assertEqual(num_seen_species['AMADEC'], expected_amadec)
        self.assertEqual(num_seen_species['MELRUB'], expected_melrub)
        
        
        
# ---------------------- Main --------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()