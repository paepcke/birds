'''
Created on Jun 8, 2021

@author: paepcke
'''
import os
import tempfile
import unittest

from birdsong.utils.testset_creator import TestsetCreator
from data_augmentation.utils import Utils


TEST_ALL = True
#TEST_ALL = False

class TestsetCreationTester(unittest.TestCase):


    def setUp(self):
        self.mover = TestsetCreator(None, None, None, unittesting=True)

    def tearDown(self):
        pass

# ------------------ Tests --------------


    #------------------------------------
    # test_get_file_distrib 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_get_file_distrib(self):
        
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='testset_creation') as tmp_dir_nm:
            self.make_dir_tree(tmp_dir_nm)
            file_distrib = self.mover.get_file_distrib(tmp_dir_nm)
            
            subdir1_path = os.path.join(tmp_dir_nm, 'subdir1')
            subdir2_path = os.path.join(tmp_dir_nm, 'subdir2')
            truth = {f"{subdir1_path}" : 10,
                     f"{subdir2_path}" : 2,
                     }
            self.assertDictEqual(file_distrib, truth)

    #------------------------------------
    # test_move
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_move(self):
        
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='testset_creation') as tmp_dir_nm:
            
            # Source will be <tmp-dir>/Training,
            # with subdirectories subdir1 and subdir2.
            # Subdir1 will have 10 files, subdir2 will
            # have 2:
            src_root = os.path.join(tmp_dir_nm, 'Training')
            os.mkdir(src_root)
            
            self.make_dir_tree(src_root)

            # A dict of subdir-name to number 
            # of files in that dir:
            file_distrib = self.mover.get_file_distrib(src_root)
            
            # Dest will be <tmp-dir>/Testset
            to_dir = os.path.join(tmp_dir_nm, 'Testset')
            
            # Move 10% from the src tree to the dst tree.
            # Since a minimum of 1 is moved, subdir2, with 
            # its 2 files will have 1 file moved:
            self.mover.move_files(file_distrib, to_dir, 10)
            
            # Get new number of files in src and dst dirs:
            new_src_distrib = self.mover.get_file_distrib(src_root)
            new_dst_distrib = self.mover.get_file_distrib(to_dir)

            src_subdir1_path = os.path.join(src_root, 'subdir1')
            src_subdir2_path = os.path.join(src_root, 'subdir2')
            
            dst_subdir1_path = os.path.join(to_dir, 'subdir1')
            dst_subdir2_path = os.path.join(to_dir, 'subdir2')
            
            # What we hope for
            new_truth_src = {f"{src_subdir1_path}" : 9,
                             f"{src_subdir2_path}" : 1,
                             }
            new_truth_dst = {f"{dst_subdir1_path}" : 1,
                             f"{dst_subdir2_path}" : 1,
                             }
            self.assertDictEqual(new_src_distrib, new_truth_src)
            self.assertDictEqual(new_dst_distrib, new_truth_dst)

    #------------------------------------
    # test_move_too_few_src_files
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_move_too_few_src_files(self):
        
        # Two few sample files to take
        # a sample. Effectively: an empty
        # directory as source:
        
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='testset_creation') as tmp_dir_nm:
            
            src_root = os.path.join(tmp_dir_nm, 'Training')
            os.mkdir(src_root)
            dst_root = os.path.join(tmp_dir_nm, 'Testing')
            
            self.make_dir_tree(src_root)
            
            # Remove one file from subdir2, leaving
            # only 1 file:
            subdir2_path = os.path.join(src_root, 'subdir2')
            for path in Utils.listdir_abs(subdir2_path):
                os.remove(path)
                # Remove just 1 file:
                break
                
            file_distrib = self.mover.get_file_distrib(src_root)
            
            self.assertRaises(ValueError,
                              self.mover.move_files,
                              file_distrib, dst_root, 10 
                              )

#---------------- Utilities ----------------

    #------------------------------------
    # make_dir_tree
    #-------------------
    
    def make_dir_tree(self, root_dir):
        
        subdir1 = os.path.join(root_dir, 'subdir1')
        subdir2 = os.path.join(root_dir, 'subdir2')
        
        os.makedirs(subdir1)
        os.makedirs(subdir2)
        
        # Create 10 files in subdir1:
        for i in range(10):
            with open(os.path.join(subdir1, f"file{i}"), 'w'):
                pass
            
        # ... but only 2 files in subdir2:
        for i in range(2):
            with open(os.path.join(subdir2, f"file{i}"), 'w'):
                pass

#---------------- Main ----------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()