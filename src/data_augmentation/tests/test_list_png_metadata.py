'''
Created on May 31, 2021

@author: paepcke
'''
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import warnings

from data_augmentation.list_png_metadata import PNGMetadataManipulator


TEST_ALL = True
#TEST_ALL = False


class PNGMetadataTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)

        # PNG File with known initial metadata:
        #      Shape: (128, 1123)
        #      sr : 22050
        #      duration : 26.1
        #      identifier : Unknown
        
        cls.clean_dove_fpath = os.path.join(cls.cur_dir,
                                            'data/list_metadata_tst_files/dove.png'
                                            )
        cls.clean_elephant_fpath = os.path.join(cls.cur_dir,
                                                'data/list_metadata_tst_files/elephant.png'
                                                )
        warnings.filterwarnings("ignore",
                                category=ResourceWarning,
                                         message='Implicitly cleaning'
                                )
        
        cls.initial_dove_metadata = {'sr': '22050', 'duration': '26.1', 'identifier': 'Unknown'}

    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory(dir='/tmp', 
                                                      prefix='png_metadata_tests')
        # Fresh copy of png file
        shutil.copy(self.clean_dove_fpath, self.tmp_dir_obj.name)
        shutil.copy(self.clean_elephant_fpath, self.tmp_dir_obj.name)
        self.dove_path = os.path.join(self.tmp_dir_obj.name, 'dove.png')
        self.elephant_path = os.path.join(self.tmp_dir_obj.name, 'elephant.png')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir_obj.name)

# --------------------- Tests ---------------

    #------------------------------------
    # test_extract_metadata
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_extract_metadata(self):
        
        # Individual file:
        fname = os.path.join(self.tmp_dir_obj.name, 'dove.png')
        metadata = PNGMetadataManipulator.extract_metadata(fname)
        self.assertDictEqual(metadata, self.initial_dove_metadata)
        
        # Directory:
        metadata_list = PNGMetadataManipulator.extract_metadata(self.tmp_dir_obj.name)
        self.assertEqual(len(metadata_list), 2)
        for md in metadata_list:
            self.assertEqual(type(md), dict)
            
        for metadata_dict in metadata_list:
            if metadata_dict != {}:
                self.assertDictEqual(metadata, self.initial_dove_metadata)

    #------------------------------------
    # test_metadata_list
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_metadata_list(self):
        for fpath, metadata in PNGMetadataManipulator.metadata_list(self.tmp_dir_obj.name):
            fname = Path(fpath).stem
            if fname == 'elephant.png':
                self.assertDictEqual(metadata, {})
            elif fname == 'dove.png':
                self.assertDictEqual(metadata, self.initial_dove_metadata)

    #------------------------------------
    # test_clear_metadata 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_clear_metadata(self):
        empty_dove_path = os.path.join(self.tmp_dir_obj.name, 'dove_copy.png')
        PNGMetadataManipulator.clear_metadata(self.dove_path, outfile=empty_dove_path
                                              )
        # Does the copy exist, and has empty metadata?
        self.assertEqual(PNGMetadataManipulator.extract_metadata(empty_dove_path), {})
        
        # The original should still have the metadata:
        self.assertDictEqual(PNGMetadataManipulator.extract_metadata(self.dove_path), 
                             self.initial_dove_metadata)

        # Now do it in place:
        PNGMetadataManipulator.clear_metadata(self.dove_path)
        self.assertDictEqual(PNGMetadataManipulator.extract_metadata(self.dove_path), 
                             {}
                             )


# ----------------------- Main -----------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()