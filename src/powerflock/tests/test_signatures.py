'''
Created on Dec 23, 2021

@author: paepcke
'''
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import Signature, SpectralTemplate


#*******TEST_ALL = True
TEST_ALL = False

class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        
        cls.xc_sound_data = os.path.join(cls.cur_dir, 'signal_processing_sounds/XenoCanto')
        
        # Xeno Canto 
        cls.sel_tbl_cmto_xc1 = os.path.join(cls.cur_dir, 'selection_tables/XenoCanto/sel_tbl_Kelley_SONG_XC274155-429_CMTO_KEL.txt')
        cls.sel_rec_cmto_xc1 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_XC274155-429_Chestnut-mandibled_Toucan_2_song.mp3')
        
        cls.sel_tbl_cmto_xc2 = os.path.join(cls.cur_dir, 'selection_tables/XenoCanto/cmto2.selections.txt')
        cls.sel_rec_cmto_xc2 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_XC591812-Ramphastos_ambiguus.mp3')
    
        cls.sel_tbl_cmto_xc3 = os.path.join(cls.cur_dir, 'selection_tables/XenoCanto/cmto3.selections.txt')
        cls.sel_rec_cmto_xc3 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_Black-mandibled_Toucan2011-1-24-1.mp3')

        # Create a signature template to work with:
        cls.recordings = [cls.sel_rec_cmto_xc1, cls.sel_rec_cmto_xc2, cls.sel_rec_cmto_xc3]
        cls.sel_tbls   = [cls.sel_tbl_cmto_xc1, cls.sel_tbl_cmto_xc2, cls.sel_tbl_cmto_xc3]
        
        #cls.templates = SignalAnalyzer.compute_species_templates('CMTOG', 
        #                                                         cls.recordings, 
        #                                                         cls.sel_tbls)
        
        
    def setUp(self):
        pass


    def tearDown(self):
        pass

    # --------------- Tests -----------

    #------------------------------------
    # test_dumps_signature
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_dumps_signature(self):
        sig_values = pd.DataFrame([[1,2,3,4],
                                   [0.1, 0.2, 0.3, 0.4]],
                                   index=[0.0, 0.2],
                                   columns=['flatness',
                                            'continuity',
                                            'pitch',
                                            'freq_mod']
                                   )
        scale_factors = pd.Series([1,2,3,4],
                                  index=['flatness', 'continuity', 'pitch', 'freq_mod'])
        sig = Signature('CMTOG',
                        sig_values,
                        scale_factors
                        )

        jstr = sig.json_dumps()
        #****sig1 = Signature.from_json(jstr)
        sig1 = Signature.from_json(jstr)
        
        self.assertTrue(sig1 == sig)
        
        # Now writing and retrieving from file:
        with tempfile.NamedTemporaryFile(dir='/tmp') as wrapper:
            fname = wrapper.name
            sig.json_dump(fname)
            sig1 = Signature.from_json_file(fname)
            
            self.assertTrue(sig == sig1)

    #------------------------------------
    # test_dumps_template
    #-------------------
    
    #******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_dumps_template(self):
        sig1_values = pd.DataFrame([[1,2,3,4],
                                    [0.1, 0.2, 0.3, 0.4]],
                                    index=[0.0, 0.2],
                                    columns=['flatness',
                                             'continuity',
                                             'pitch',
                                             'freq_mod']
                                    )
        
        sig2_values = pd.DataFrame([[5,6,7,8],
                                    [0.5, 0.7, 0.7, 0.8]],
                                    index=[0.0, 0.2],
                                    columns=['flatness',
                                             'continuity',
                                             'pitch',
                                             'freq_mod']
                                    )
        
        scale_factors1 = pd.Series([1,2,3,4],
                                   index=['flatness', 'continuity', 'pitch', 'freq_mod'])
        
        scale_factors2 = pd.Series([5,6,7,8],
                                   index=['flatness', 'continuity', 'pitch', 'freq_mod'])
        
        sig1 = Signature('CMTOG',
                         sig1_values,
                         scale_factors1
                         )
        sig2 = Signature('CMTOG',
                         sig2_values,
                         scale_factors2
                         )
        
        template = SpectralTemplate([sig1, sig2])

        jstr = template.json_dumps()

        template1 = SpectralTemplate.from_json(jstr)
        
        self.assertTrue(template == template1)
        
        # Now writing and retrieving from file:
        with tempfile.NamedTemporaryFile(dir='/tmp') as wrapper:
            fname = wrapper.name
            template.json_dump(fname)
            template1 = SpectralTemplate.from_json_file(fname)
            
            self.assertTrue(template == template1)

    #------------------------------------
    # test_compute_species_templates
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_compute_species_templates(self):

        # Now done once in setUpClass() 
        # recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2, self.sel_rec_cmto_xc3]
        # sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2, self.sel_tbl_cmto_xc3]
        # templates = SignalAnalyzer.compute_species_templates('CMTOG', 
        #                                                      recordings, 
        #                                                      sel_tbls)
        self.assertEqual(len(self.templates), 3)
        self.assertListEqual([len(template) for template in self.templates], [11,8,11])
        
        for tmpl_num, template in enumerate(self.templates):
            for call_num, sig in enumerate(template.signatures):
                rec_fname = Path(self.recordings[tmpl_num]).name
                self.assertTrue(sig.name == f"{rec_fname}_call{call_num}")

        expected_sig_lengths = [
            [85, 72, 133, 60, 59, 45, 48, 42, 37, 87, 107], 
            [46, 46, 41, 41, 46, 68, 48, 46], 
            [58, 54, 53, 62, 58, 54, 54, 56, 58, 56, 51]
            ]
        observed_sig_lengths = [template.sig_lengths for template in self.templates]
        self.assertListEqual(observed_sig_lengths, expected_sig_lengths)
        
        # While we are in here, test signature averaging in
        # templates:
        
        template = self.templates[0]
        mean_sig = template.mean_sig

        # The mean sig should be as long as the
        # longest among the sigs:
        longest = pd.DataFrame(expected_sig_lengths).max().max()
        self.assertEqual(len(template.mean_sig), longest)
        
        # Compute what the first element of the 
        # mean sig should be: the mean of the first
        # element of all sigs in the template:
        
        mean_first_els = np.mean([sig[0] for sig in template.signatures])
        self.assertAlmostEqual(mean_sig[0], mean_first_els, places=3)

    def testName(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()