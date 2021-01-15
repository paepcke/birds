'''
Created on Jan 15, 2021

@author: paepcke
'''
import unittest

from birdsong.utils.move_random_perc_of_files import FileMover


TEST_ALL = True
#TEST_ALL = False

class Test(unittest.TestCase):
    '''
    Assumes the following file structure
    below this script's dir:
    
		                 <curr_dir>
		                     |
		                  mv_tst_dir
		                /    |       \
               dir_file1  dir_file1   mv_tst_dir1
                                           |
                                    /      |        \
                            dir1_file1  dir1_file2 dir1_file3
    '''


    def setUp(self):
        pass


    def tearDown(self):
        pass


    #------------------------------------
    # test_move
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_move(self):
        mover = FileMover()
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()