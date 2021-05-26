'''
Created on Apr 29, 2021

@author: paepcke
'''
import os, sys
from pathlib import Path
import tempfile
import unittest

import multiprocessing as mp

from data_augmentation.chop_spectrograms import SpectrogramChopper
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import WhenAlreadyDone, Utils

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

TEST_ALL = True
#TEST_ALL = False

# --------------------- Arguments Class -----------------

class Arguments:
    '''
    Used to create a fake argparse args data structure.
    '''
    pass

# --------------------- TestChopSpectrograms Class -----------------

class TestChopSpectrograms(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestChopSpectrograms, cls).setUpClass()
        
        cls.skip_size = 2 # sec
        
        cls.cur_dir  = os.path.dirname(__file__)
        cls.spectro_root = os.path.join(cls.cur_dir, 
                                       'spectro_data_long')
        cls.spectro_file = os.path.join(cls.spectro_root, 'DOVE/dove_long.png')
        
        cls.num_spectro_files = len(Utils.find_in_dir_tree(
            cls.spectro_root, 
            pattern='*.png', 
            entry_type='file'))

        _spectro, metadata = SoundProcessor.load_spectrogram(cls.spectro_file)
        try:
            cls.duration      = float(metadata['duration'])
        except KeyError:
            raise AssertionError(f"Spectrogram test file {os.path.basename(cls.spectro_file)} has no duration metadata")
        
        cls.default_win_len = 5 # seconds
        

    def setUp(self):
        pass


    def tearDown(self):
        pass

# -------------- Test Methods ---------------

    #------------------------------------
    # test_chop_one_spectrogram_file
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_chop_one_spectrogram_file(self):
        
        with tempfile.TemporaryDirectory(dir='/tmp',
                                         prefix='chopping', 
                                         ) as dir_nm:
            chopper = SpectrogramChopper(
                self.spectro_root,
                dir_nm,
                overwrite_policy=WhenAlreadyDone.OVERWRITE
                )
            species = Path(self.spectro_file).parent.stem
            outdir  = os.path.join(dir_nm, species)
            true_snippet_time_width = chopper.chop_one_spectro_file(
                self.spectro_file,
                outdir,
                skip_size=self.skip_size
                )
            snippet_names = os.listdir(outdir)
            num_expected_snippets = 0
            cur_time = true_snippet_time_width
            while cur_time < self.duration:
                num_expected_snippets += 1
                cur_time += self.skip_size

            self.assertEqual(len(snippet_names), num_expected_snippets)
            
            # Check embedded metadata of one snippet:
            
            _spectro, metadata = SoundProcessor.load_spectrogram(Utils.listdir_abs(outdir)[0])
            self.assertEqual(round(float(metadata['duration(secs)']), 3),
                             round(true_snippet_time_width, 3)
                             )
            
    #------------------------------------
    # test_compute_worker_assignments_empty_dest 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_compute_worker_assignments_empty_dest(self):

        # Virginal scenario: no spectro was every
        # created for any of the 12 audio samples:
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:

            # Since no spectros are already done,
            # overwrite_policy should make no difference;
            # try it for all three:

            self.verify_worker_assignments(self.spectro_root,
                                           dst_dir, 
                                           WhenAlreadyDone.ASK, 
                                           self.num_spectro_files)
            self.verify_worker_assignments(self.spectro_root,
                                           dst_dir, 
                                           WhenAlreadyDone.SKIP, 
                                           self.num_spectro_files)
            self.verify_worker_assignments(self.spectro_root,
                                           dst_dir, 
                                           WhenAlreadyDone.OVERWRITE,
                                           self.num_spectro_files)


    #------------------------------------
    # test_compute_worker_assignments_one_spectro_done
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_compute_worker_assignments_one_spectro_done(self):

        # Scenario: one spectro was already done:
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            
            # Fake-create an existing spectrogram: 
            os.mkdir(os.path.join(dst_dir, 'DOVE'))
            done_spectro_path = os.path.join(dst_dir, 
                                             'DOVE/')
            Path(done_spectro_path).touch()
            
            num_tasks_done = len(Utils.find_in_dir_tree(
                dst_dir,
                pattern='*.png', 
                entry_type='file'))

            true_num_assignments = self.num_spectro_files - num_tasks_done

            self.verify_worker_assignments(self.spectro_root,
                                           dst_dir, 
                                           WhenAlreadyDone.SKIP, 
                                           true_num_assignments)
            
            # We are to overwrite existing files, 
            # all sound files will need to be done:
            
            true_num_assignments = self.num_spectro_files
            self.verify_worker_assignments(self.spectro_root,
                                           dst_dir, 
                                           WhenAlreadyDone.OVERWRITE, 
                                           true_num_assignments)

    #------------------------------------
    # test_from_commandline 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_commandline(self):
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            
            args = Arguments()
            args.input   = self.spectro_root
            args.outdir  = dst_dir
            args.workers = None
            
            # Number of spectrogram .png files
            # in source tree:
            spectros_to_chop = Utils.find_in_dir_tree(self.spectro_root, '*.png')

            manager = mp.Manager()
            global_info = manager.dict()
            global_info['jobs_status'] = manager.list()

            # ------ Chop spectrograms:
            SpectrogramChopper.run_workers(
                args,
                global_info,
                overwrite_policy=WhenAlreadyDone.OVERWRITE
                )
                
            dirs_filled = [os.path.join(dst_dir, species_dir) 
                           for species_dir 
                           in os.listdir(dst_dir)]
            
            num_spectros_done = sum([len(Utils.find_in_dir_tree(one_filled_dir, '*.png'))
                                     for one_filled_dir
                                     in dirs_filled])

            self.assertTrue(num_spectros_done > len(spectros_to_chop))
            
            self.check_spectro_sanity(dirs_filled)
            
            # Remember the creation times:
            file_times = self.record_creation_times(dirs_filled)

            # ------ SKIP the existing spectrograms:
            # Run again, asking to skip already existing
            # spectros:
            global_info = manager.dict()
            global_info['jobs_status'] = manager.list()
            
            SpectrogramChopper.run_workers(
                args,
                global_info,
                overwrite_policy=WhenAlreadyDone.SKIP
                )

            dirs_filled = [os.path.join(dst_dir, species_dir) 
                           for species_dir 
                           in os.listdir(dst_dir)]

            # Mod times of png files must NOT have changed,
            # b/c of skipping
            new_file_times = self.record_creation_times(dirs_filled)
            self.assertDictEqual(new_file_times, file_times)
            
            # ------ Force RECREATION of spectrograms:
            # Run again with OVERWRITE, forcing the 
            # spectros to be done again:
            global_info = manager.dict()
            global_info['jobs_status'] = manager.list()

            SpectrogramChopper.run_workers(
                args,
                global_info,
                overwrite_policy=WhenAlreadyDone.OVERWRITE
                )
                
            dirs_filled = [os.path.join(dst_dir, species_dir) 
                           for species_dir 
                           in os.listdir(dst_dir)]
                           
            self.check_spectro_sanity(dirs_filled)
            
            # File times must be *different* from previous
            # run because we asked to overwrite:

            new_file_times = self.record_creation_times(dirs_filled)
            for fname in file_times.keys():
                try:
                    self.assertTrue(new_file_times[fname] != file_times[fname])
                except KeyError as e:
                    print(repr(e))


# --------------- Utilities --------------

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
    # verify_worker_assignments
    #-------------------
    
    def verify_worker_assignments(self, 
                                  spectro_root,
                                  outdir,
                                  overwrite_policy,
                                  true_num_assignments,
                                  num_workers_to_use=None
                                  ):
        '''
        Called to compute worker assignments under different
        overwrite_policy decisions and with no or some spectrograms
        already created.
        
        :param spectro_root: root of species/recorder subdirectories
        :type spectro_root: src
        :param outdir: root of where spectrograms are to be placed
        :type outdir: str
        :param overwrite_policy: what to do if dest spectro exists
        :type overwrite_policy: WhenAlreadyDone
        :param true_num_assignments: expected number of tasks to
            be done: number of sound files minus num of spectros
            already done
        :type true_num_assignments: int
        :param num_workers_to_use: number of cores to use
        :type num_workers_to_use: int
        '''

        (worker_assignments, _num_workers) = SpectrogramChopper.compute_worker_assignments(
            spectro_root,
            outdir,
            overwrite_policy=overwrite_policy,
            num_workers=num_workers_to_use
            )
        
        # Only check for num workers used if
        # a particular number of requested:
        if num_workers_to_use is not None:
            self.assertEqual(len(worker_assignments), num_workers_to_use)
        # Must have true_num_assignments assignments, no matter 
        # how they are distributed across CPUs:
        num_tasks = sum([len(assignments) for assignments in worker_assignments])
        self.assertEqual(num_tasks, true_num_assignments)

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

# ------------------------ Main ------------
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()