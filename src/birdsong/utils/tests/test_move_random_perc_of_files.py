'''
Created on Jan 15, 2021

@author: paepcke
'''
import os
import unittest

from birdsong.utils.move_random_perc_of_files import FileMover


TEST_ALL = True
#TEST_ALL = False

class Test(unittest.TestCase):
    '''
    Assumes the following file structure
    below this script's dir (10 files total)
    
		                 <curr_dir>
		                     |
		                  mv_tst_dir
		                /    |       \
            dir_file1 ... dir_file4   mv_tst_dir1
                                           |
                                           |     
                                     /            \
                                dir1_file1 ... dir1_file6
    '''

    @classmethod
    def setUpClass(cls):
        curr_dir = os.path.dirname(__file__)
        cls.root = os.path.join(curr_dir, 'mv_tst_dir')
        cls.dir_num_files  = 4
        cls.dir1_num_files = 6
        

    def setUp(self):
        pass


    def tearDown(self):
        pass


    #------------------------------------
    # test_move
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_move(self):
        
        # Pretend to move 20 percent
        # of the 10 files:
        perc = 20
        mover = FileMover(self.root, '/tmp', perc)
        
        # Get list of files in the 
        # order that os.walk() visits them:
        
        all_files = []
        for root_path, _dirs, files in os.walk(self.root):
            full_paths = [os.path.join(root_path, file)
                             for file in files
                          ]
            all_files.extend(full_paths)
            
        self.assertEqual(len(all_files), 10)

        # Only pretend to move the files 
        # in the test directory tree:
        
        num_moved = mover.move(lambda src, dst: print(f"From {src} to {dst}"),
                               testing=True
                               )
        self.assertEqual(num_moved, 2)
        
        # List of indices into file list that
        # random generator created in move():
        
        self.assertEqual(len(mover.file_indices), 2)
        
        # Correct list of the files that should have
        # been moved:
        
        true_files_to_move = [all_files[file_idx] 
                                 for file_idx 
                                 in mover.file_indices
                                 ]

        # To match the move() method list:
        true_files_to_move.reverse()
        
        self.assertListEqual(mover.files_moved, true_files_to_move)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
