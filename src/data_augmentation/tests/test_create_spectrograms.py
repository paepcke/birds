'''
Created on Apr 26, 2021

@author: paepcke
'''
import os
import tempfile
import unittest

from data_augmentation.create_spectrograms import SpectrogramCreator
from data_augmentation.utils import Utils, WhenAlreadyDone


#*****TEST_ALL = True
TEST_ALL = False


# --------------------- Arguments Class -----------------

class Arguments:
    '''
    Used to create a fake argparse args data structure.
    '''
    pass

# --------------------- TestSpectrogramCreator Class -----------------

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
    
    #*****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_commandline(self):
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            
            args = Arguments()
            args.input   = self.sound_root
            args.outdir  = dst_dir
            args.workers = None
                                                
            SpectrogramCreator.run_workers(
                args,
                overwrite_policy=WhenAlreadyDone.ASK
                )
            
            dirs_filled = [os.path.join(dst_dir, species_dir) 
                           for species_dir 
                           in os.listdir(dst_dir)]
            self.check_spectro_sanity(dirs_filled)

            # DO IT AGAIN to ensure that overwrite will
            # create new files.
            
            # Remember the creation times:
            file_times = self.record_creation_times(dirs_filled)

            # Run again, asking to skip already existing
            # spectros:
            SpectrogramCreator.run_workers(
                args,
                overwrite_policy=WhenAlreadyDone.SKIP
                )

            dirs_filled = [os.path.join(dst_dir, species_dir) 
                           for species_dir 
                           in os.listdir(dst_dir)]

            new_file_times = self.record_creation_times(dirs_filled)
            self.assertDictEqual(new_file_times, file_times)

            
            # File times must be *different* from previous
            # run because we asked to overwrite:
            
                
            

#**************                            
            # SpectrogramCreator.run_workers(
                # args,
                # overwrite_policy=WhenAlreadyDone.OVERWRITE
                # )
                #
            # dirs_filled = [os.path.join(dst_dir, species_dir) 
                           # for species_dir 
                           # in os.listdir(dst_dir)]
                           #
            # self.check_spectro_sanity(dirs_filled)
            #
            # # File times must be *different* from previous
            # # run because we asked to overwrite:
            #
            # new_file_times = self.record_creation_times(dirs_filled)
            # for fname in file_times.keys():
                # self.assertTrue(new_file_times[fname] != file_times[fname])
                #
#**************                
            # And ONE MORE TIME, this time forcing program
            # to ask permission to overwrite:
            
            # Remember the creation times:
            #*****file_times = self.record_creation_times(dirs_filled)
            
            SpectrogramCreator.run_workers(
                args,
                overwrite_policy=WhenAlreadyDone.ASK,
                )
            new_file_times = self.record_creation_times(dirs_filled)
            # Spectrograms should all be what they used to be:
            self.assertDictEqual(new_file_times, file_times)


# -------------------- Utilities ---------------

    #------------------------------------
    # record_creation_times 
    #-------------------
    
    def record_creation_times(self, dirs_filled):
        '''
        Given list of absolute file paths, return 
        a dict mapping each path to a Unix modification time
        in fractional epoch seconds
        
        :param dirs_filled: list of absolute file paths
        :type dirs_filled: [str]
        :return dict of modification times
        :rtype {str : float}
        '''
        
        file_times = {}
        for species_dst_dir in dirs_filled:
            for spec_fname in Utils.listdir_abs(species_dst_dir):
                file_times[spec_fname] = os.path.getmtime(spec_fname)

        return file_times
    
    
    #------------------------------------
    # run_spectrogrammer
    #-------------------


    def run_spectrogrammer(self, src_dir):
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            dirs_filled = SpectrogramCreator.create_spectrogram(src_dir, 
                                                                 dst_dir, 
                                                                 num=2,
                                                                 overwrite_policy=WhenAlreadyDone.OVERWRITE
                                                                 )

            # Check that each spectro is of
            # reasonable size:
            self.check_spectro_sanity(dirs_filled)
            
            dirs_filled = SpectrogramCreator.create_spectrogram(src_dir, 
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