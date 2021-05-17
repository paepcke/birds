'''
Created on May 13, 2021

@author: paepcke
'''
import os
from tempfile import TemporaryDirectory
import unittest

from birdsong.utils.match_snippets_to_selection_table import SnippetSelectionTableMapper
from data_augmentation.utils import Utils


TEST_ALL = True
#TEST_ALL = False


class TestSnippetMatcher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir        = os.path.dirname(__file__)
        cls.data_dir       = os.path.join(cls.cur_dir, 'data')
        cls.sel_tbl_path   = os.path.join(cls.data_dir, 'raven_sel_tbl.csv')
        cls.tst_snips_path = os.path.join(cls.data_dir, 'fld_snippets')

        # Get start-time sorted list of dicts,
        # each dict containing info of one selection
        # table:
        cls.sel_tbl_lst    = Utils.read_raven_selection_table(cls.sel_tbl_path)

    def setUp(self):
        self.tmp_outdir = TemporaryDirectory(dir='/tmp', prefix='sel_tbl_matching').name
        self.mapper = SnippetSelectionTableMapper(
            self.sel_tbl_path,
            self.tst_snips_path, 
            self.tmp_outdir
            )

    def tearDown(self):
        # Destroy the out directory
        try:
            os.remove(self.tmp_outdir)
        except Exception:
            pass

# ----------------------- Tests ------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_match_snippets(self):
        self.mapper.match_snippets(self.sel_tbl_lst, self.tst_snips_path, self.tmp_outdir)
        # new Species subdirs we should see under
        # the tmp dir:
        subdirs_truth = set(os.listdir(self.tmp_outdir))
        self.assertSetEqual(subdirs_truth, set(os.listdir(self.tmp_outdir)))
        
        species_rep = {
            'vase' : 4,
            'bgta' : 3,
            'rcwp' : 3,
            'rbps' : 4,
            'howp' : 1,
            'noise': 26
            }
        for species in subdirs_truth:
            num_snippets = len(os.listdir(os.path.join(self.tmp_outdir, species)))
            self.assertEqual(species_rep[species], num_snippets)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()