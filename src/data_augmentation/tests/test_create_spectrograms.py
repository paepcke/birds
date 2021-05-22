'''
Created on Apr 26, 2021

@author: paepcke
'''
import os
from pathlib import Path
import tempfile
import unittest

from data_augmentation.create_spectrograms import SpectrogramCreator
from data_augmentation.utils import Utils, WhenAlreadyDone
import multiprocessing as mp
import numpy as np
from logging_service.logging_service import LoggingService


TEST_ALL = True
#TEST_ALL = False

# --------------------- Arguments Class -----------------

class Arguments:
    '''
    Used to create a fake argparse args data structure.
    '''
    pass

# --------------------- TestSpectrogramCreator Class -----------------

class TestSpectrogramCreator(unittest.TestCase):
    '''
    Tests use 12 audio files distributed evenly over two
    subdirectories: HENLES_S and DYSMEN_S. For the worker
    distribution tests the test outcomes depend on the 
    local machine's number of CPUs. SpectrogramCreator
    by default uses 50% of available CPUs. An example outcome 
    for a 12-CPU machine is:
    
        assignments = np.array(
		     [[('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259380.mp3'),
		      ('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259381.mp3')
		      ],
		     [('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259383.mp3'),
		      ('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259379.mp3')
		      ],
		     [('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259378.mp3'),
		      ('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259384.mp3')
		      ],
		     [('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc513.mp3'),
		      ('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc518466.mp3')],
		    
		     [('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc531750.mp3'),
		      ('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc50519.mp3')],
		    
		     [('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc511477.mp3'),
		      ('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc548015.mp3')
              ]
             ])

    Thus, a normal outcome in this case has 12 assignments, spread 
    across 6 CPUs, with two tasks for each CPU. A task consists of a 
    tuple: species-subdir-name, and audio file to process. 
    '''

    @classmethod
    def setUpClass(cls):
        cls.cur_dir     = os.path.dirname(__file__)
        cls.sound_root  = os.path.join(cls.cur_dir, 'sound_data')
        # Number of cores to use:
        num_cores = mp.cpu_count()
        cls.num_workers = round(num_cores * Utils.MAX_PERC_OF_CORES_TO_USE  / 100)
        
        cls.num_sound_files = len(Utils.find_in_dir_tree(
            cls.sound_root, 
            pattern='*.mp3', 
            entry_type='file'))
        
        cls.assignments = np.array(
		     [[('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259380.mp3'),
		      ('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259381.mp3')
		      ],
		     [('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259383.mp3'),
		      ('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259379.mp3')
		      ],
		     [('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259378.mp3'),
		      ('HENLES_S', 'SONG_Henicorhinaleucosticta_xc259384.mp3')
		      ],
		     [('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc513.mp3'),
		      ('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc518466.mp3')],
		    
		     [('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc531750.mp3'),
		      ('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc50519.mp3')],
		    
		     [('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc511477.mp3'),
		      ('DYSMEN_S', 'SONG_Dysithamnusmentalis_xc548015.mp3')
              ]
             ])        

    def setUp(self):
        pass


    def tearDown(self):
        pass

# -------------------------- Tests -------------


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

            self.verify_worker_assignments(self.sound_root,
                                           dst_dir, 
                                           WhenAlreadyDone.ASK, 
                                           self.num_sound_files)
            self.verify_worker_assignments(self.sound_root,
                                           dst_dir, 
                                           WhenAlreadyDone.SKIP, 
                                           self.num_sound_files)
            self.verify_worker_assignments(self.sound_root,
                                           dst_dir, 
                                           WhenAlreadyDone.OVERWRITE,
                                           self.num_sound_files)

    #------------------------------------
    # test_compute_worker_assignments_one_spectro_done
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_compute_worker_assignments_one_spectro_done(self):

        # Scenario: one spectro was already done:
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            
            # Fake-create an existing spectrogram: 
            os.mkdir(os.path.join(dst_dir, 'HENLES_S'))
            done_spectro_path = os.path.join(dst_dir, 
                                             'HENLES_S/SONG_Henicorhinaleucosticta_xc259378.png')
            Path(done_spectro_path).touch()
            
            num_tasks_done = len(Utils.find_in_dir_tree(
                dst_dir,
                pattern='*.png', 
                entry_type='file'))

            true_num_assignments = self.num_sound_files - num_tasks_done

            self.verify_worker_assignments(self.sound_root,
                                           dst_dir, 
                                           WhenAlreadyDone.SKIP, 
                                           true_num_assignments)
            
            # We are to overwrite existing files, 
            # all sound files will need to be done:
            
            true_num_assignments = self.num_sound_files
            self.verify_worker_assignments(self.sound_root,
                                           dst_dir, 
                                           WhenAlreadyDone.OVERWRITE, 
                                           true_num_assignments)

    #------------------------------------
    # test_assignment_imbalance
    #-------------------

    # MORE TROUBLE THAN WORTH: does not happen:
    
    # @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    # def test_assignment_imbalance(self):

    #     # Scenario: some cores must accomplish
    #     # more tasks than others:
    #     with tempfile.TemporaryDirectory(dir='/tmp', 
    #                                      prefix='test_spectro') as dst_dir:

    #         self.verify_worker_assignments(self.sound_root,
    #                                        dst_dir, 
    #                                        WhenAlreadyDone.ASK,
    #                                        self.num_sound_files,
    #                                        num_workers_to_use=5)

    #------------------------------------
    # test_from_commandline 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_commandline(self):
        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as dst_dir:
            
            args = Arguments()
            args.input   = self.sound_root
            args.outdir  = dst_dir
            args.workers = None

            # ------ Create spectrograms:
            SpectrogramCreator.run_workers(
                args,
                overwrite_policy=WhenAlreadyDone.OVERWRITE
                )
                
            dirs_filled = [os.path.join(dst_dir, species_dir) 
                           for species_dir 
                           in os.listdir(dst_dir)]
            self.check_spectro_sanity(dirs_filled)
            
            # Remember the creation times:
            file_times = self.record_creation_times(dirs_filled)

            # ------ SKIP the existing spectrograms:
            # Run again, asking to skip already existing
            # spectros:
            SpectrogramCreator.run_workers(
                args,
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

            SpectrogramCreator.run_workers(
                args,
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
                self.assertTrue(new_file_times[fname] != file_times[fname])

    #------------------------------------
    # test_bad_wav_file 
    #-------------------
    
    #*****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_bad_wav_file(self):

        with tempfile.TemporaryDirectory(dir='/tmp', 
                                         prefix='test_spectro') as in_dir:
    
            with tempfile.TemporaryDirectory(dir='/tmp', 
                                             prefix='test_spectro') as out_dir:
    
    
                log_file = os.path.join(out_dir, 'err_log.txt')
                SpectrogramCreator.log = LoggingService()
                SpectrogramCreator.log.log_file=log_file
    
                bad_species_path = os.path.join(in_dir, 'BADBIRD')
                os.mkdir(bad_species_path)
                bad_bird_fname   = 'bad_audio.wav'
                assignments = ([('BADBIRD', bad_bird_fname)])
                bad_bird_path = os.path.join(bad_species_path, bad_bird_fname)
                # Create a 0-length file:
                Path(bad_bird_path).touch()
                
                ret_value_slot = mp.Value("b", False)
                
                # Ensure that an error is logged, though
                # none is raised:

                SpectrogramCreator.create_from_file_list(
                    assignments,
                    in_dir,
                    out_dir,
                    WhenAlreadyDone.OVERWRITE,
                    ret_value_slot,
                    env=None)
                # Read the log file:
                with open(log_file, 'r') as fd:
                    log_entry = fd.read()
                
                # The log msg should include:
                # "ERROR: One file could not be processed ... AudioLoadException('Audio file to load is empty ..."
                self.assertTrue(log_entry.find('to load is empty') > -1)

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
        
    #------------------------------------
    # verify_worker_assignments
    #-------------------
    
    def verify_worker_assignments(self, 
                                  sound_root,
                                  outdir,
                                  overwrite_policy,
                                  true_num_assignments,
                                  num_workers_to_use=None
                                  ):
        '''
        Called to compute worker assignments under different
        overwrite_policy decisions and with no or some spectrograms
        already created.
        
        :param sound_root: root of species/recorder subdirectories
        :type sound_root: src
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

        (worker_assignments, _num_workers) = SpectrogramCreator.compute_worker_assignments(
            sound_root,
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

# ------------------ Main -------------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()