'''
Created on Apr 26, 2021

@author: paepcke
'''
import os
import subprocess
import tempfile
import unittest

from data_augmentation.create_spectrograms import SpectrogramCreator
from data_augmentation.utils import Utils, WhenAlreadyDone
from IPython.utils.capture import capture_output


#******TEST_ALL = True
TEST_ALL = False

class TestSpectrogramCreator(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.cur_dir     = os.path.dirname(__file__)
        cls.sound_root  = os.path.join(cls.cur_dir, 'sound_data')

    def setUp(self):
        pass


    def tearDown(self):
        pass

# -------------------------- Tests -------------

    #------------------------------------
    # test_create_spectro 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_create_spectro_one_dir(self):

        # A directory that itself contains audio files:
        src_dir = os.path.join(self.sound_root, 'DYSMEN_S/')
        self.run_spectrogrammer(src_dir)

    #------------------------------------
    # test_create_spectro_multiple_dirs 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_create_spectro_multiple_dirs(self):
        self.run_spectrogrammer(self.sound_root)


    #------------------------------------
    # test_from_commandline 
    #-------------------
    
    #*******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_commandline(self):
        cmd_file = os.path.join(self.cur_dir, '../create_spectrograms.py')
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
        
            completed_proc = subprocess.run([cmd_file, self.sound_root, dst_dir],
                                            capture_output=True,
                                            env=os.environ,
                                            shell=True
                                            )
            self.assertEqual(completed_proc.returncode, 0)
            self.check_spectro_sanity(os.listdir(dst_dir))

# -------------------- Utilities ---------------


    #------------------------------------
    # run_spectrogrammer
    #-------------------


    def run_spectrogrammer(self, src_dir):
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            dirs_filled = SpectrogramCreator.create_spectrograms(src_dir, 
                                                                 dst_dir, 
                                                                 num=2,
                                                                 overwrite_policy=WhenAlreadyDone.OVERWRITE
                                                                 )

            # Check that each spectro is of
            # reasonable size:
            self.check_spectro_sanity(dirs_filled)

    #------------------------------------
    # check_spectro_sanity 
    #-------------------
    
    def check_spectro_sanity(self, dirs_filled):
        '''
        Raises assertion error if any file in
        the passed-in list of directories is less than
        5000 bytes long
        
        :param dirs_filled: list of directories whose content
            files to check for size
        :type dirs_filled: [str]
        '''
        # Check that each spectro is of
        # reasonable size:
        for species_dst_dir in dirs_filled:
            for spec_file in Utils.listdir_abs(species_dst_dir):
                self.assertTrue(os.stat(spec_file).st_size > 5000)
        


# ------------------ Main -------------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()