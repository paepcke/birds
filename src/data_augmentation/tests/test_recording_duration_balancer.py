'''
Created on Sep 1, 2021

@author: paepcke
'''
import os
import shutil
import unittest

from data_augmentation.recordings_duration_balancer import DurationsBalancer
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils

import pandas as pd


TEST_ALL = True
#**********TEST_ALL = False

class RecordingDurationBalancerTester(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        cls.cur_dir = os.path.dirname(__file__)
        cls.tst_recordings_2files = os.path.join(cls.cur_dir, 'data/augmentation_tst_data/YCEUG')
        cls.tst_recordings_3files = os.path.join(cls.cur_dir, 'data/augmentation_tst_data/WTROS')
        
        cls.tst_manifest_dir  = os.path.join(cls.cur_dir,
                                             'data/rebalancing_Audio_manifest_augmentation_tst_data')
        # File with summary of all species durations:
        cls.tst_manifest_file = os.path.join(cls.tst_manifest_dir, 'manifest.json')
        cls.tst_wtros_json_file = os.path.join(cls.tst_manifest_dir,'WTROS_manifest.json.gz') 
        
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
                                     duration_goal_seconds=10,
                                     excess_dest_dir=None,
                                     dry_run=True)
        
        # Get the df of recordings, and ensure
        # it is sorted by decreasing recording time:
        
        durations = balancer.dur_df['recording_length_secs'].to_list()
        self.assertListEqual(durations, [45.53,29.88,14.55])
        
        balancer.balance()
        
        bird1_path = os.path.join(self.tst_recordings_3files, 'wtros_bird1.mp3')
        bird2_path = os.path.join(self.tst_recordings_3files, 'wtros_bird2.mp3')
        bird3_path = os.path.join(self.tst_recordings_3files, 'wtros_bird3.mp3')
                
        expected_actions = set([f"delete,{bird1_path}",
                                f"delete,{bird2_path}",
                                f"delete,{bird3_path}",
                                ])
        log = set(balancer.dry_run_log)
        # Now have like:
        #    ['delete,<full-path>/WTROS/wtros_bird3.mp3',
        #     'delete,<full-path>/WTROS/wtros_bird1.mp3']
        self.assertSetEqual(log, expected_actions)
        

        # Do the same, but with moving the excess
        # recordings, rather than deleting them:
        
        balancer = DurationsBalancer(self.tst_recordings_3files, 
                                     duration_goal_seconds=10,
                                     excess_dest_dir=self.mv_dst_dir,
                                     dry_run=True)
        
        balancer.balance()
        expected_actions = set([f"mv,{bird1_path} {self.mv_dst_dir}",
                                f"mv,{bird2_path} {self.mv_dst_dir}",
                                f"mv,{bird3_path} {self.mv_dst_dir}",
                                ])
        log = set(balancer.dry_run_log)
        # Now have like:
        #    ['delete,<full-path>/WTROS/wtros_bird3.mp3',
        #     'delete,<full-path>/WTROS/wtros_bird1.mp3']
        self.assertSetEqual(log, expected_actions)

    #------------------------------------
    # test_remove_nothing
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_remove_nothing(self):

        # Test on the WTROS species (3 recordings):        
        #
        #                  recording_length_secs recording_length_hhs_mins_secs
        # wtros_bird3.mp3                  45.53                 0:00:45.530000
        # wtros_bird1.mp3                  29.88                 0:00:29.880000
        # wtros_bird2.mp3                  14.55                 0:00:14.550000        
        
        # Ask for a cap of 100 secs for the sum or
        # recordings in wtros:
        balancer = DurationsBalancer(self.tst_recordings_3files, 
                                     duration_goal_seconds=100,
                                     excess_dest_dir=self.mv_dst_dir,
                                     dry_run=True)
        balancer.balance()
        
        # Should have done nothing, b/c sum of 
        # WTROS recordings is 89.96:
        
        self.assertEqual(len(balancer.dry_run_log), 0)
        
        # Now ask for cap of 80sec; should
        # report removing bird2:
        balancer = DurationsBalancer(self.tst_recordings_3files, 
                                     duration_goal_seconds=80,
                                     excess_dest_dir=self.mv_dst_dir,
                                     dry_run=True)
        balancer.balance()
        
        # The dry-run log should have a single
        # enty, for moving wtros_bird3.mp3 to the default
        # /tmp/duration/balancing directory:

        self.assertEqual(len(balancer.dry_run_log), 1)
        action = balancer.dry_run_log[0]
        expected = f"mv,{self.tst_recordings_3files}/wtros_bird3.mp3 /tmp/duration_balancing" 
        
        self.assertEqual(action, expected)

        # Lower threshold to where two
        # files must be moved:
        balancer = DurationsBalancer(self.tst_recordings_3files, 
                                     duration_goal_seconds=15,
                                     excess_dest_dir=self.mv_dst_dir,
                                     dry_run=True)
        balancer.balance()

        self.assertEqual(len(balancer.dry_run_log), 2)
        action0 = balancer.dry_run_log[0]
        action1 = balancer.dry_run_log[1]
        expected0 = f"mv,{self.tst_recordings_3files}/wtros_bird3.mp3 /tmp/duration_balancing" 
        expected1 = f"mv,{self.tst_recordings_3files}/wtros_bird1.mp3 /tmp/duration_balancing" 
        
        self.assertEqual(action0, expected0)
        self.assertEqual(action1, expected1)

    #------------------------------------
    # test_inventory_available
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_inventory_available(self):
        
        balancer = DurationsBalancer(self.tst_recordings_3files, 
                                     duration_goal_seconds=10,
                                     excess_dest_dir=None,
                                     inventory=self.tst_manifest_dir,
                                     dry_run=True)
        expected = pd.DataFrame(
                     [[45.53,'0:00:45.530000'],
                      [29.88,'0:00:29.880000'],
                      [14.55,'0:00:14.550000']
                      ],
                      columns=['recording_length_secs', 'recording_length_hhs_mins_secs'],
                      index=['wtros_bird3.mp3', 'wtros_bird1.mp3', 'wtros_bird2.mp3']
                      )

        Utils.assertDataframesEqual(balancer.dur_df, expected)
        print('foo')

# -------------------- Main ---------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()