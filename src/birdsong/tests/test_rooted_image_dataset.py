'''
Created on May 2, 2021

@author: paepcke

NOTE: this is a pretty slim set of unittests.
      Needs more.

'''
import unittest

from birdsong.rooted_image_dataset import SingleRootImageDataset


TEST_ALL = True
#TEST_ALL = False


class TestRootedImageDataset(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass

# ------------------ Tests -------------------

    def test_cull_samples(self):
        ds = SingleRootImageDataset('/dummy/filename.txt')
        sample_id_to_class = {
         100 : 0,
         101 : 0,
         102 : 0,
         103 : 0,
         104 : 0,
         105 : 0,
         200 : 1,
         201 : 1,
         202 : 1,
         203 : 1,
         204 : 1,
         205 : 1
         }
        
        sample_id_to_path = {
         100 : '/tmp/foo/img0',
         101 : '/tmp/foo/img1',
         102 : '/tmp/foo/img2',
         103 : '/tmp/foo/img3',
         104 : '/tmp/foo/img4',
         105 : '/tmp/foo/img5',
         200 : '/tmp/foo/img6',
         201 : '/tmp/foo/img7',
         202 : '/tmp/foo/img8',
         203 : '/tmp/foo/img9',
         204 : '/tmp/foo/img10',
         205 : '/tmp/foo/img11'
         }
        
        reduced_sid2cid, reduced_sid2path = ds.cull_samples(
            sample_id_to_class,
            sample_id_to_path,
            20
            )
        self.assertEqual(len(reduced_sid2cid), 2)
        self.assertEqual(len(reduced_sid2path), 2)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()