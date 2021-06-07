'''
Created on May 13, 2021

@author: paepcke
'''
from _collections import OrderedDict
from _collections_abc import Iterable
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import tempfile
import unittest
import warnings

import multiprocessing as mp

from birdsong.match_snippets_to_selection_table import SelTblSnipsAssoc
from birdsong.match_snippets_to_selection_table import SnippetSelectionTableMapper
from data_augmentation.list_png_metadata import PNGMetadataManipulator
from data_augmentation.utils import Utils, Interval


TEST_ALL = True
#TEST_ALL = False

'''
The following are the correct results for matching test 28 test snippets at 
<proj-root>/src/birdsong/utils/tests/data/fld_snippets
to the selection table DS_AM01_20190711_170000.Table.1.selections.txt
in <proj-root>/src/birdsong/utils/tests/data/selection_tables

Each row below describes one of the output snippets that
are generated by test_whole_thing():

    <species>, <filename> <start-time> <end-time>
    
If all went well, the species in the snippets' metadata
key species' will contain <species>. And the files will
be in subdirectories of the temporary directory created
in test_whole_think(). Those subdirs will be named after
the species.
'''


# The following truth was created by pausing 
# test_whole_thing() just before it ends (and
# destroys the tmp dir). Then:
#
#     from data_augmentation.list_png_metadata import PNGMetadataManipulator as PMM
#     # Get list of (full-path, metadata_dict) elements:
#     res = list(PMM.metadata_list('/tmp/snip_matchingi0qywb0h'))
#     # Convert the full paths to just the fnames, and make a dict
#     # fname : md-dict:
#     d = {}
#     for fname, md in res:
#         d[Path(fname).stem] = md

#     # Only keep species, and start/end times of metadata:
#     for fname, md in d.items():
#         d[fname] = {'species'    : md['species'],
#                     'start_time' : float(md['start_time(secs)']),
#                     'end_time'   : float(md['end_time(secs)'])
#                     }
#     # Sort by start time of snippets:
#     sorted_d = OrderedDict(sorted(d.items(),
#                            key=lambda fname_dict_tuple: fname_dict_tuple[1]['start_time']))

#     # Pretty print without sorting the metadata dicts:                
#     pprint.pprint(sorted_d, sort_dicts=False)

snippet_matching_truth = \
OrderedDict([('AM01_20190711_170000_sw-start0_wcpa',
              {'species': 'wcpa',
               'start_time': 0.0,
               'end_time': 5.944272445820434}),
             ('AM01_20190711_170000_sw-start2_wcpa',
              {'species': 'wcpa',
               'start_time': 1.996904024767802,
               'end_time': 7.9411764705882355}),
             ('AM01_20190711_170000_sw-start4_wcpa',
              {'species': 'wcpa',
               'start_time': 3.993808049535604,
               'end_time': 9.938080495356038}),
             ('AM01_20190711_170000_sw-start6',
              {'species': 'noise',
               'start_time': 5.9907120743034055,
               'end_time': 11.93498452012384}),
             ('AM01_20190711_170000_sw-start8',
              {'species': 'noise',
               'start_time': 7.987616099071208,
               'end_time': 13.931888544891642}),
             ('AM01_20190711_170000_sw-start10',
              {'species': 'noise',
               'start_time': 9.98452012383901,
               'end_time': 15.928792569659443}),
             ('AM01_20190711_170000_sw-start12',
              {'species': 'noise',
               'start_time': 11.981424148606811,
               'end_time': 17.925696594427244}),
             ('AM01_20190711_170000_sw-start14',
              {'species': 'noise',
               'start_time': 13.978328173374614,
               'end_time': 19.922600619195048}),
             ('AM01_20190711_170000_sw-start16',
              {'species': 'noise',
               'start_time': 15.975232198142416,
               'end_time': 21.91950464396285}),
             ('AM01_20190711_170000_sw-start18',
              {'species': 'noise',
               'start_time': 17.97213622291022,
               'end_time': 23.916408668730654}),
             ('AM01_20190711_170000_sw-start20',
              {'species': 'noise',
               'start_time': 19.96904024767802,
               'end_time': 25.913312693498455}),
             ('AM01_20190711_170000_sw-start22_shwc',
              {'species': 'shwc',
               'start_time': 21.96594427244582,
               'end_time': 27.910216718266255}),
             ('AM01_20190711_170000_sw-start24_shwc',
              {'species': 'shwc',
               'start_time': 23.962848297213622,
               'end_time': 29.907120743034056}),
             ('AM01_20190711_170000_sw-start26_shwc',
              {'species': 'shwc',
               'start_time': 25.959752321981426,
               'end_time': 31.90402476780186}),
             ('AM01_20190711_170000_sw-start26_unk1',
              {'species': 'unk1',
               'start_time': 25.959752321981426,
               'end_time': 31.90402476780186}),
             ('AM01_20190711_170000_sw-start28',
              {'species': 'shwc',
               'start_time': 27.956656346749227,
               'end_time': 33.90092879256966}),
             ('AM01_20190711_170000_sw-start28_unk1',
              {'species': 'unk1',
               'start_time': 27.956656346749227,
               'end_time': 33.90092879256966}),
             ('AM01_20190711_170000_sw-start30_wcpa',
              {'species': 'wcpa',
               'start_time': 29.953560371517028,
               'end_time': 35.89783281733746}),
             ('AM01_20190711_170000_sw-start30_shwc',
              {'species': 'shwc',
               'start_time': 29.953560371517028,
               'end_time': 35.89783281733746}),
             ('AM01_20190711_170000_sw-start30_unk1',
              {'species': 'unk1',
               'start_time': 29.953560371517028,
               'end_time': 35.89783281733746}),
             ('AM01_20190711_170000_sw-start32',
              {'species': 'unk1',
               'start_time': 31.950464396284833,
               'end_time': 37.89473684210527}),
             ('AM01_20190711_170000_sw-start34',
              {'species': 'noise',
               'start_time': 33.94736842105263,
               'end_time': 39.89164086687306}),
             ('AM01_20190711_170000_sw-start36',
              {'species': 'noise',
               'start_time': 35.94427244582044,
               'end_time': 41.88854489164087}),
             ('AM01_20190711_170000_sw-start38',
              {'species': 'noise',
               'start_time': 37.94117647058824,
               'end_time': 43.88544891640867}),
             ('AM01_20190711_170000_sw-start40',
              {'species': 'noise',
               'start_time': 39.93808049535604,
               'end_time': 45.88235294117647}),
             ('AM01_20190711_170000_sw-start42',
              {'species': 'noise',
               'start_time': 41.93498452012384,
               'end_time': 47.87925696594427}),
             ('AM01_20190711_170000_sw-start44',
              {'species': 'noise',
               'start_time': 43.93188854489164,
               'end_time': 49.87616099071207}),
             ('AM01_20190711_170000_sw-start46',
              {'species': 'noise',
               'start_time': 45.92879256965944,
               'end_time': 51.873065015479874}),
             ('AM01_20190711_170000_sw-start48',
              {'species': 'noise',
               'start_time': 47.925696594427244,
               'end_time': 53.869969040247675}),
             ('AM01_20190711_170000_sw-start50',
              {'species': 'noise',
               'start_time': 49.92260061919505,
               'end_time': 55.86687306501548}),
             ('AM01_20190711_170000_sw-start52',
              {'species': 'noise',
               'start_time': 51.91950464396285,
               'end_time': 57.863777089783284}),
             ('AM01_20190711_170000_sw-start54',
              {'species': 'wcpa',
               'start_time': 53.916408668730654,
               'end_time': 59.860681114551085})])

'''
The above in summary:
'species': 'wcpa',
'species': 'wcpa',
'species': 'wcpa',
'species': 'wcpa',

'species': 'shwc',
'species': 'shwc',
'species': 'shwc',
'species': 'shwc',

'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',
'species': 'noise',

'species': 'unk1',
'species': 'unk1',
'species': 'unk1',
'species': 'unk1'
'''

class TestSnippetMatcher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir        = os.path.abspath(os.path.dirname(__file__))
        cls.data_dir       = os.path.join(cls.cur_dir, 'data')
        cls.sel_tbl_dir    = os.path.join(cls.data_dir, 'selection_tables')
        cls.sel_tbl_path1  = os.path.join(cls.sel_tbl_dir, 
                                          'DS_AM01_20190711_170000.Table.1.selections.txt')
        cls.sel_tbl_path2  = os.path.join(cls.sel_tbl_dir, 
                                          'DS_AM01_20190712_050000.Table.1.selections.txt')
        cls.tst_snips_dir = os.path.join(cls.data_dir, 'fld_snippets')

        # Get start-time sorted list of dicts,
        # each dict containing info of one selection
        # table:
        cls.sel_tbl_lst1    = Utils.read_raven_selection_table(cls.sel_tbl_path1)
        cls.sel_tbl_lst2    = Utils.read_raven_selection_table(cls.sel_tbl_path2)
        
        # For testing SelTblSnipsAssoc iteration:
        cls.tbl_snips_assoc_iter_test_dir = os.path.join(cls.data_dir, 
                                                         'tbl_snippet_assoc_iter_data')
        
        cls.sel1 = {
            'Begin Time (s)' : 10,
            'End Time (s)'   : 20,
            'species'        : 'dog',
            'mix'            : None
            }
        
        cls.sel2 = {
            'Begin Time (s)' : 25,
            'End Time (s)'   : 30,
            'species'        : 'dog',
            'mix'            : ['species1']
            }
        
        cls.sel3 = {
            'Begin Time (s)' : 30,
            'End Time (s)'   : 40,
            'species'        : 'cat',
            'mix'            : ['species1']
            }
        cls.sel4 = {
            'Begin Time (s)' : 50,
            'End Time (s)'   : 60,
            'species'        : 'dog',
            'mix'            : ['species1', 'species2']
            }
        
        # Case 1: left of all sels:
        cls.iv1 = Interval(5,6)
        # Case 2 end reaches into sel1:
        cls.iv2 = Interval(6,12)
        # Case 3 entirely enclosed in sel1:
        cls.iv3 = Interval(12,16)
        # Case 4: only start is in sel1:
        cls.iv4 = Interval(14,22)
        # Case 5: in no selection:
        cls.iv5 = Interval(22,24)
        # Case 6: straddles two selections sel 2/3:
        cls.iv6 = Interval(28,35)
        # Case 7: to the right of all sels (sel 4):
        cls.iv7 = Interval(65,70)
        
        cls.sels = [cls.sel1, cls.sel2, cls.sel3, cls.sel4]
        
        warnings.filterwarnings("ignore",
                                category=ResourceWarning,
                                         message='Implicitly cleaning'
                                )
        warnings.filterwarnings("ignore",
                                category=ResourceWarning,
                                         message='unclosed file'
                                )
        
        # A communication dict visible to 
        # All processes:
        # For nice progress reports, create a shared
        # dict quantities that each process will report
        # on to the console as progress indications:
        #    o Which job-completions have already been reported
        #      to the console (a list)
        #    o The most recent number of already created
        #      output images 
        # The dict will be initialized in run_workers,
        # but all Python processes on the various cores
        # will have read/write access:
        manager = mp.Manager()
        cls.global_info = manager.dict()
        cls.global_info['jobs_status'] = manager.list()

    #------------------------------------
    # setUP 
    #-------------------

    def setUp(self):
        self.tmp_outdir_obj = TemporaryDirectory(dir='/tmp', 
                                                 prefix='sel_tbl_matching')
        self.tmp_outdir = self.tmp_outdir_obj.name
        # self.mapper = SnippetSelectionTableMapper(
        #    self.sel_tbl_path1,
        #    self.tst_snips_dir, 
        #    self.tmp_outdir
        #    )
        self.mapper = SnippetSelectionTableMapper(None,None,None,None,unittesting=True)

    #------------------------------------
    # tearDown 
    #-------------------

    def tearDown(self):
        # Destroy the out directory
        try:
            shutil.rmtree(self.tmp_outdir)
        except Exception as _e:
            pass

# ----------------------- Tests ------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_create_spectro_gen_for_sel_tbls(self):
        
        # Given one snippet directory path, and 
        # directory with Raven selection tables, 
        # create a dict whose keys are recording
        # IDs gleaned from the file names or recordings
        # and selection tables. Each value is one 
        # object, and instance of SelTblSnipsAssoc.
        # It is an iterator that provides paths to 
        # snippets that are covered by the table.
        
        # Get dict:
        #   {<recording-ID>  : SelTblSnipsAssoc-instance}
        tbl_snips_mapping = self.mapper.create_snips_gen_for_sel_tbls(
            self.tst_snips_dir,
            self.sel_tbl_dir)
        
        # Ensure that there is an entry in the dict
        # for both of our two test selection tables:
        
        self.assertEqual(len(tbl_snips_mapping), 2)

        rec_id1 = self.mapper.extract_recording_id(self.sel_tbl_path1)
        rec_id2 = self.mapper.extract_recording_id(self.sel_tbl_path2)
        
        self.assertSetEqual(set(list(tbl_snips_mapping.keys())), 
                            set([rec_id1, rec_id2])
                            )

    #------------------------------------
    # test_match_snippet
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_match_snippet(self):

        # Get list of dicts; each dict will contain
        # info from one row in the selection table
        # at self.sel_tbl_path1:
        
        sel_dicts = Utils.read_raven_selection_table(self.sel_tbl_path1)
        
        # The list is expected to look like this:
        #
        #      [{'Selection': '5',
        #         'View': 'Spectrogram 1',
        #         'Channel': '1',
        #         'Begin Time (s)': 0.014506764,
        #         'End Time (s)': 4.656671354,
        #         'Low Freq (Hz)': 345.6,
        #         'High Freq (Hz)': 22050.0,
        #         'Delta Time (s)': '4.6422',
        #         'species': 'no bird',
        #         'type': '',
        #         'number': '',
        #         'mix': [],
        #         'time_interval': {'low_val': 0.014506764,
        #         'high_val': 4.656671354},
        #         'freq_interval': {'low_val': 345.6,
        #                           'high_val': 22050.0}
        #       },
        #      ...
        #      ]    
        #
        # and there should be as many entries as there
        # are rows in the table:
        
        with open(self.sel_tbl_path1, 'r') as fd:
            all_lines = fd.readlines()
            # The '-1' subtracts the sel table's
            # column header line:
            self.assertEqual(len(sel_dicts), len(all_lines)-1)

        # Create a dict:
        #    {<recording-id>  : SelectionTblSnipsAssoc-instance}
        #
        # Each SelectionTblSnipsAssoc instance is a generator of snippets
        # from one recording. The recording-id is the part of 
        # field recording and selection table file names like:
        #    AM01_20190712_050000

        rec_id_to_sel_tbl_snips_gens = self.mapper.create_snips_gen_for_sel_tbls(
            self.tst_snips_dir, 
            self.sel_tbl_dir)

        # We have two test select tables, so the 
        # number of SelectionTblSnipsAssoc instances
        # should match:
        
        num_sel_tbls = len(os.listdir(self.sel_tbl_dir))
        self.assertEqual(len(rec_id_to_sel_tbl_snips_gens),
                         num_sel_tbls)

        # Examine the rec-id-->snippet-generator for 
        # recording AM01_20190711_170000: 
        assoc170k = rec_id_to_sel_tbl_snips_gens['AM01_20190711_170000']
        
        self.assertEqual(assoc170k.rec_id, 'AM01_20190711_170000')
        self.assertTrue(isinstance(assoc170k.snip_dir, Iterable))
        
        # Get number of snippets with recording id 
        # of 'AM01_20190711_170000' that we have among 
        # the test snippets:
        snip_metadata_dicts = list(assoc170k)
        snip_fnames = [snip_metadata_dict['snip_path']
                       for snip_metadata_dict
                       in snip_metadata_dicts
                       ]
        snips_fname_filter = filter(self.mapper.extract_recording_id, snip_fnames)
        snips_list = list(snips_fname_filter)
        num_tst_snips  = len(snips_list)
        
        num_snips_in_generator = len(snip_metadata_dicts)
        self.assertEqual(num_snips_in_generator, num_tst_snips)

    #------------------------------------
    # test_snips_iterator 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_snips_iterator(self):
        
        sel_snip_assoc = SelTblSnipsAssoc(self.sel_tbl_path1, 
                                          self.tbl_snips_assoc_iter_test_dir)
        snip_nums = []
        for snip_md in sel_snip_assoc:
            # Get <loooong-dir-path>/AM01_20190711_170000_sw-start4.png
            # Get the integer after 'start':
            snip_path = snip_md['snip_path']
            start_n = Path(snip_path).stem.split('-')[1]
            n       = int(start_n[len('start'):])
            snip_nums.append(n)
        self.assertListEqual(snip_nums, [0,2,4,6,5])

    #------------------------------------
    # test_extract_recording_id 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_extract_recording_id(self):
        
        tst_path = '/foo/bar/AM01_20190719_063242.png'
        rec_id = self.mapper.extract_recording_id(tst_path)
        self.assertEqual(rec_id, 'AM01_20190719_063242')
        
        tst_path = '/foo/barAM01_20190719_063242/fum.png'
        rec_id = self.mapper.extract_recording_id(tst_path)
        self.assertTrue(rec_id is None)

    #------------------------------------
    # test_interval_matching 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_interval_matching(self):
        
        # Case 1: left of all sels:
        self.assertIsNone(self.mapper.find_covering_sel(self.sels, self.iv1))
        # Case 2 end reaches into sel1:
        self.assertDictEqual(self.mapper.find_covering_sel(self.sels, self.iv2),
                             self.sel1)
        # Case 3 entirely enclosed in sel1:
        self.assertDictEqual(self.mapper.find_covering_sel(self.sels, self.iv3),
                             self.sel1)
        # Case 4 start is in sel
        self.assertDictEqual(self.mapper.find_covering_sel(self.sels, self.iv4),
                             self.sel1)
        # Case 6: straddling multiple selections (sel2 and sel3):
        res_sel_dict = self.mapper.find_covering_sel(self.sels, self.iv6)
        
        truth_sel = self.sel2.copy()
        truth_sel['mix'] = list(set(['cat', 'species1']))
        
        for key,val in res_sel_dict.items():
            if key != 'mix':
                self.assertEqual(val, truth_sel[key])
            else:
                # Compare the mix lists regardless of
                # order. Contrary to the method name, 
                # the test does test the individual list
                # members:
                self.assertCountEqual(val, truth_sel['mix'])
        
        # Case 7: Interval entirely to the right of all sels:
        self.assertIsNone(self.mapper.find_covering_sel(self.sels, self.iv7))

    #------------------------------------
    # test_whole_thing
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_whole_thing(self):
        # Integration test:

        with tempfile.TemporaryDirectory(dir='/tmp', prefix='snip_matching') as tmp_dir_name:
            _mapper = SnippetSelectionTableMapper(
                self.sel_tbl_dir,
                self.tst_snips_dir,
                tmp_dir_name,
                self.global_info
                )
            
            #for fpath, metadata in PNGMetadataManipulator.metadata_list(tmp_dir_name):
            #    print(fpath, metadata)
            
            # Get a list of metadata from all snippets:
            #all_snip_metadata = list(PNGMetadataManipulator.metadata_list(tmp_dir_name))
            # Sort by time:
            #sorted_md = sorted(all_snip_metadata, 
            #                   key=lambda one_fnm_dict: float(one_fnm_dict[1]['start_time(secs)']))

            #for (fpath, md_dict) in sorted_md:
            #    fname = Path(fpath).stem
            #    print(f"{md_dict['species']} {fname}    : {md_dict['start_time(secs)']}")
            
            #print(sorted_md)
            self.check_whole_thing_outcome(tmp_dir_name)
                
            
# ---------------------- Utilities -----------------

    #------------------------------------
    # check_whole_thing_outcome 
    #-------------------
    
    def check_whole_thing_outcome(self, root_dir):

        # Set of species represented in the 
        # selections:
        all_species = {truth_dict['species']
                       for truth_dict
                       in snippet_matching_truth.values()
                       }
        
        # First: does the given dir include one subdir
        # for each species in table snippet_matching_truth?

        subdirs = os.listdir(root_dir)
        self.assertEqual(len(all_species), len(subdirs))
        
        self.assertSetEqual(set(subdirs), all_species)

        # Get a list of metadata from all snippets:
        # We will receive:
        #    [(<absolute-snippet-path>, <metadata>),
        #     (<absolute-snippet-path>, <metadata>),
        #       ...
        #    ]
        all_snip_metadata = list(PNGMetadataManipulator.metadata_list(root_dir))
        # Get dict
        #   {<snippet-file-name> : <metadata>,
        #    <snippet-file-name> : <metadata>,
        #         ...
        #   }
        snip_md_dict = {Path(snip_path).stem : metadata_dict
                        for (snip_path, metadata_dict)
                        in all_snip_metadata
                        }
        
        noise_start_times = [md['start_time(secs)']
                             for md
                             in snip_md_dict.values()
                             if md['species'] == 'noise'
                             ]
        noise_end_times = [md['end_time(secs)']
                             for md
                             in snip_md_dict.values()
                             if md['species'] == 'noise'
                             ]
        
        shwc_start_times = [md['start_time(secs)']
                             for md
                             in snip_md_dict.values()
                             if md['species'] == 'shwc'
                             ]
        shwc_end_times = [md['end_time(secs)']
                             for md
                             in snip_md_dict.values()
                             if md['species'] == 'shwc'
                             ]
        
        wcpa_start_times = [md['start_time(secs)']
                             for md
                             in snip_md_dict.values()
                             if md['species'] == 'wcpa'
                             ]
        wcpa_end_times = [md['end_time(secs)']
                             for md
                             in snip_md_dict.values()
                             if md['species'] == 'wcpa'
                             ]

        unk1_start_times = [md['start_time(secs)']
                             for md
                             in snip_md_dict.values()
                             if md['species'] == 'unk1'
                             ]
        unk1_end_times = [md['end_time(secs)']
                             for md
                             in snip_md_dict.values()
                             if md['species'] == 'unk1'
                             ]
        
        for _fname, md in PNGMetadataManipulator.metadata_list(root_dir):
            species = md['species']
            if species == 'noise':
                self.assertIn(md['start_time(secs)'], noise_start_times)
                self.assertIn(md['end_time(secs)'], noise_end_times)
            elif species == 'shwc':
                self.assertIn(md['start_time(secs)'], shwc_start_times)
                self.assertIn(md['end_time(secs)'], shwc_end_times)
            elif species == 'wcpa':
                self.assertIn(md['start_time(secs)'], wcpa_start_times)
                self.assertIn(md['end_time(secs)'], wcpa_end_times)
            elif species == 'unk1':
                self.assertIn(md['start_time(secs)'], unk1_start_times)
                self.assertIn(md['end_time(secs)'], unk1_end_times)

        num_noise_snips = len(os.listdir(os.path.join(root_dir, 'noise')))
        self.assertEqual(num_noise_snips, 18)
        self.assertEqual(len(noise_start_times), 18)
        self.assertEqual(len(noise_end_times), 18)
        
        num_shwc_snips = len(os.listdir(os.path.join(root_dir, 'shwc')))
        self.assertEqual(num_shwc_snips, 4)
        self.assertEqual(len(shwc_start_times), 4)
        self.assertEqual(len(shwc_end_times), 4)
        
        num_wcpa_snips = len(os.listdir(os.path.join(root_dir, 'wcpa')))
        self.assertEqual(num_wcpa_snips, 4)
        self.assertEqual(len(wcpa_start_times), 4)
        self.assertEqual(len(wcpa_end_times), 4)

        num_unk1_snips = len(os.listdir(os.path.join(root_dir, 'unk1')))
        self.assertEqual(num_unk1_snips, 4)
        self.assertEqual(len(unk1_start_times), 4)
        self.assertEqual(len(unk1_end_times), 4)
        

# ---------------------- Main ------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()