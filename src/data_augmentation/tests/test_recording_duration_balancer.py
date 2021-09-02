'''
Created on Sep 1, 2021

@author: paepcke
'''
import os
import shutil
import unittest

from data_augmentation.recordings_duration_balancer import DurationsBalancer
from data_augmentation.sound_processor import SoundProcessor


TEST_ALL = True
#TEST_ALL = False


class RecordingDurationBalancerTester(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        cls.cur_dir = os.path.dirname(__file__)
        cls.tst_recordings_2files = os.path.join(cls.cur_dir, 'data/augmentation_tst_data/YCEUG')
        cls.tst_recordings_3files = os.path.join(cls.cur_dir, 'data/augmentation_tst_data/WTROS')
        
        cls.dur_df_2files = SoundProcessor.find_recording_lengths(cls.tst_recordings_2files)
        cls.dur_df_3files = SoundProcessor.find_recording_lengths(cls.tst_recordings_3files)
        
        # Destination of 'move' type recording removal:
        cls.mv_dst_dir = '/tmp/duration_balancing'  

    def setUp(self):
        pass


    def tearDown(self):
        # For the 'move' type removal tests, which
        # create the destination directory:
        shutil.rmtree(self.mv_dst_dir, ignore_errors=True)

# ---------------- Tests --------------    

    #------------------------------------
    # test_max_to_remove
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_max_to_remove(self):
        
        balancer = DurationsBalancer(self.tst_recordings_3files, 
                                     duration_goal=10,
                                     excess_dest_dir=None,
                                     dry_run=True)
        
        # Get the df of recordings, and ensure
        # it is sorted by decreasing recording time:
        
        durations = balancer.dur_df['recording_length_secs'].to_list()
        self.assertListEqual(durations, [45,29,14])
        
        balancer.balance()
        
        bird3_path = os.path.join(self.tst_recordings_3files, 'wtros_bird3.mp3')
        bird1_path = os.path.join(self.tst_recordings_3files, 'wtros_bird1.mp3')
        
        expected_actions = [f"delete,{bird3_path}",
                            f"delete,{bird1_path}"
                            ]
        log = balancer.dry_run_log
        # Now have like:
        #    ['delete,<full-path>/WTROS/wtros_bird3.mp3',
        #     'delete,<full-path>/WTROS/wtros_bird1.mp3']
        self.assertListEqual(log, expected_actions)
        

        # Do the same, but with moving the excess
        # recordings, rather than deleting them:
        
        balancer = DurationsBalancer(self.tst_recordings_3files, 
                                     duration_goal=10,
                                     excess_dest_dir=self.mv_dst_dir,
                                     dry_run=True)
        
        balancer.balance()
        expected_actions = [f"mv,{bird3_path} {self.mv_dst_dir}",
                            f"mv,{bird1_path} {self.mv_dst_dir}"
                            ]
        log = balancer.dry_run_log
        # Now have like:
        #    ['delete,<full-path>/WTROS/wtros_bird3.mp3',
        #     'delete,<full-path>/WTROS/wtros_bird1.mp3']
        self.assertListEqual(log, expected_actions)

    #------------------------------------
    # test_remove_nothing
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_remove_nothing(self):

        balancer = DurationsBalancer(self.tst_recordings_3files, 
                                     duration_goal=100,
                                     excess_dest_dir=self.mv_dst_dir,
                                     dry_run=True)
        balancer.balance()
        
        self.assertEqual(len(balancer.dry_run_log), 0)



# -------------------- Main ---------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()