'''
Created on Feb 12, 2021

@author: paepcke
'''

TEST_ALL = True
#TEST_ALL = False

import unittest
import tempfile

from birdsong.utils.tensorboard_manager import TensorBoardManager


class ShowInBrowserTester(unittest.TestCase):


    def setUp(self):
        self.tensorboard_root = tempfile.TemporaryDirectory(
            prefix='tensorboard_event_root', 
            dir='/tmp'
            ) 
        self.tb_man = TensorBoardManager(self.tensorboard_root.name)

    def tearDown(self):
        self.tb_man.close()
        
        # Destroying the temp directory
        # removes it from the file systems:
        self.tensorboard_root = None
    
    #------------------------------------
    # test_show_in_browser 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_show_in_browser(self):
        
        # Use default port, which is *not*
        # the standard tensorboard port of
        # 6006; so testing shouldn't interfere
        # with another running tensorboard instance:
        
        self.assertTrue(self.tb_man.tensorboard_server is not None)
        
        # None return from poll() indicates subprocess running:
        self.assertTrue(self.tb_man.tensorboard_server.poll() is None)

        self.assertTrue(self.tb_man.web_browser is not None)
        
        # None return from poll() indicates subprocess running:
        self.assertTrue(self.tb_man.web_browser.poll() is None)
        
        # Happens fast enough that browser window won't
        # come up, unless we pause here. When input returns,
        # tearDown() will close the browser:
        
        input("Browser window should have popped up with empty tensorboard; hit any key: ")
        
# --------------- Main -----------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()