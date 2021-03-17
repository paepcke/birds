'''
Created on Mar 1, 2021

@author: paepcke
'''
import json5
import os
import unittest
import tempfile

from logging_service.logging_service import LoggingService

from birdsong.xeno_canto_manager import XenoCantoCollection, XenoCantoRecording


TEST_ALL = True
#TEST_ALL = False

class XenoCantoProcessorTester(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.tst_dir = os.path.join(cls.curr_dir,
                                   'data_other/xeno_canto_tst_data'
                                   )
        cls.tst_file = os.path.join(cls.tst_dir, 'xeno_canto_coll_small.pickle')

    def setUp(self):
        pass 
    
    def tearDown(self):
        pass

# ------------------- Tests --------------

    #------------------------------------
    # test_load 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_load(self):
        coll = XenoCantoCollection.load(self.tst_file)
        phylo_names = list(coll.keys())
        
        # For each list of recordings, every
        # member's phylo_name should be the same
        # as the collection key:
        for phylo_name in phylo_names:
            rec_instances = coll[phylo_name]
            for recording_inst in rec_instances:
                self.assertEqual(recording_inst.phylo_name, phylo_name)

    #------------------------------------
    # test_to_json_xc_recording 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_recording_to_json_xc_recording(self):
        coll = XenoCantoCollection.load(self.tst_file)
        rec  = coll['Tangaragyrola'][0]
        
        jstr = rec.to_json()
        
        # Make Python dict from json:
        recovered_dict = eval(jstr)
        
        # The recovered dict must reflect
        # the instance vars of the XenoCantoRecording:
        
        for inst_var_nm, inst_var_val in recovered_dict.items():
            self.assertEqual(rec.__getattribute__(inst_var_nm), inst_var_val)
        
    #------------------------------------
    # test_from_json_xc_recording 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_json_xc_recording(self):
        
        coll = XenoCantoCollection.load(self.tst_file)

        rec  = coll['Tangaragyrola'][0]
        jstr = rec.to_json()
        
        rec_recovered = XenoCantoRecording.from_json(jstr)
        
        for inst_var_nm, inst_var_val in rec_recovered.__dict__.items():
            if type(inst_var_val) == str:
                self.assertEqual(inst_var_val, rec.__getattribute__(inst_var_nm))
            elif inst_var_nm == 'log':
                # Inst var 'log' should be a LoggingService:
                self.assertEqual(type(inst_var_val), LoggingService)
        
        num_inst_vars_rec_orig      = len(rec.__dict__.keys())
        num_inst_vars_rec_recovered = len(rec_recovered.__dict__.keys())

        # Ensure all inst vars are recovered:
        self.assertEqual(num_inst_vars_rec_orig, num_inst_vars_rec_recovered)

    #------------------------------------
    # test_to_json_collection 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_to_json_collection(self):
        
        coll = XenoCantoCollection.load(self.tst_file)
        
        jstr = coll.to_json()
        
        try:
            # Should be able to get a dict
            # back from the json string without
            # error:
            coll_as_dict = json5.loads(jstr)
        except Exception as e:
            self.fail(f"Could not read back collection json: {repr(e)}")
            
            
        # Now have:
        #    {
        #      "phylo_nm1" : [{recording_dict1}, {recording_dict2}, ...],
        #      "phylo_nm2" : [{recording_dict1}, {recording_dict2}, ...],
        #                       ...
        #    }
        
        # Separately econstitute each recording into
        # a XenoCantoRecording, and ensure that its
        # phylo_name equals the collection entry's phil_name<n>:
        
        for phylo_nm, rec_dict_jstr_list in coll_as_dict.items():
            for rec_jstr in rec_dict_jstr_list:
                rec_obj = XenoCantoRecording.from_json(rec_jstr)
                self.assertEqual(rec_obj.phylo_name, phylo_nm)

    #------------------------------------
    # test_coll_json_to_file
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_coll_json_to_file(self):
        coll = XenoCantoCollection.load(self.tst_file)
        
        tmp_obj = tempfile.NamedTemporaryFile(suffix='.json', 
                                              prefix='xeno_canto_tst', 
                                              dir='/tmp', 
                                              delete=False)
        fname = tmp_obj.name
        
        new_coll = None
        try:
            # Write to file (which will already exist,
            # therefore the force, so no request for confimation:
            coll.to_json(dest=fname, force=True)
            
            # Get it back:
            new_coll = XenoCantoCollection.from_json(src=fname)
        finally:
            os.remove(fname)
        
        #**********
        #new_coll.__eq__(coll)
        #**********
        self.assertTrue(new_coll == coll) 

    #------------------------------------
    # test_download_non_existing_destdir
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_download_non_existing_destdir(self):
        
        rec = self.make_fake_rec_instance(load_dir=self.curr_dir)

        # Non-existing dest dir:
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir_name:
            os.rmdir(tmp_dir_name)
            go_ahead = rec.download(dest_dir=tmp_dir_name, testing=True)
            self.assertTrue(os.path.exists(tmp_dir_name) and \
                            os.path.isdir(tmp_dir_name))
            self.assertTrue(go_ahead)
            
    #------------------------------------
    # test_download_explicit_overwrite_ok 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_download_explicit_overwrite_ok(self):

        rec = self.make_fake_rec_instance(load_dir=self.curr_dir)

        # Provide explicit 'overwrite OK',
        # no global default set yet:
        
        XenoCantoRecording.always_overwrite = None
        self.download_call(rec, 
                           overwrite_existing=True,
                           expected_go_ahead=True,
                           expected_global_default=True
                           )

    #------------------------------------
    # test_download_global_overwrite_ok
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_download_global_overwrite_ok(self):
        
        rec = self.make_fake_rec_instance(load_dir=self.curr_dir)

        XenoCantoRecording.always_overwrite = True
        self.download_call(rec, 
                           overwrite_existing=True,
                           expected_go_ahead=True,
                           expected_global_default=True
                           )

    #------------------------------------
    # test_download_must_ask_for_input 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_download_must_ask_for_input(self):

        rec = self.make_fake_rec_instance(load_dir=self.curr_dir)

        # Must ask user for overwrite permission,
        # and user says Yes:
        try:
            saved_input_fn = __builtins__.input
            __builtins__.input = lambda _: 'Yes'
            
            XenoCantoRecording.always_overwrite = None
            self.download_call(rec, 
                               overwrite_existing=None,
                               expected_go_ahead=True,
                               expected_global_default=True
                               )
        finally:
            __builtins__.input = saved_input_fn

        # Must ask user for overwrite permission,
        # and user says No:
        try:
            saved_input_fn = __builtins__.input
            __builtins__.input = lambda _: 'No'
            
            XenoCantoRecording.always_overwrite = None
            self.download_call(rec, 
                               overwrite_existing=None,
                               expected_go_ahead=False,
                               expected_global_default=False
                               )
        finally:
            __builtins__.input = saved_input_fn

    #------------------------------------
    # test_download_no_permission 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_download_no_permission(self):

        # Forbidden dir creation:
        rec = self.make_fake_rec_instance(load_dir='/usr/bin/foo')
        XenoCantoRecording.always_overwrite = True                

        try:
            saved_input_fn = __builtins__.input
            
            # Replace input() with func that provides
            # a dir for which permissions are OK:
            with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir_name:
                __builtins__.input = lambda _: tmp_dir_name

                with tempfile.NamedTemporaryFile(suffix='.mp3', 
                                                 prefix='xeno_canto_tst', 
                                                 dir='/tmp',
                                                 delete=True) as fd:
                    fname = fd.name
                    rec.full_name = fname
                    rec._filename = os.path.basename(fname)
                    go_ahead = rec.download(dest_dir='/usr/bin/foo',
                                            testing=True)
    
                self.assertTrue(go_ahead)
            
        finally:
            __builtins__.input = saved_input_fn

# ----------------------- Utils ---------


    #------------------------------------
    # download_call 
    #-------------------
    
    def download_call(self,
                      rec,
                      overwrite_existing, # for download call
                      expected_go_ahead,
                      expected_global_default,
                      ):
        '''
        Creates a temporary file, and calls download 
        with the provided args. Then asserts 
        expected_go_ahead and expected_global_default
        values.

        @param rec: XenoCantoRecording on which to call download()
        @type rec: XenoCantoRecording
        @param overwrite_existing: whether or not to overwrite
        @type overwrite_existing: bool
        @param expected_go_ahead: whether resulting go_ahead should
            be True or False
        @type expected_go_ahead: bool
        @param expected_global_default: whether after the call the
            global default overwrite instruction should be True
            or False
        @type expected_global_default:bool
        '''
        
        vocalization_type = rec.type.upper()
        with tempfile.NamedTemporaryFile(suffix='.mp3', 
                                         prefix=f"{vocalization_type}_xeno_canto_tst", 
                                         dir='/tmp',
                                         delete=True) as fd:
            fname = fd.name
            rec.full_name = fname
            rec._filename = os.path.basename(fname)
            go_ahead = rec.download(dest_dir=os.path.dirname(fname),
                                    overwrite_existing=overwrite_existing, 
                                    testing=True)
            self.assertEqual(go_ahead, expected_go_ahead)
            self.assertEqual(XenoCantoRecording.always_overwrite,
                             expected_global_default
                             )

    #------------------------------------
    # make_fake_rec_instance 
    #-------------------
    
    def make_fake_rec_instance(self, load_dir=None): 
                               
        
        recording_metadata = {
            'id'        :  1234,
            'gen'       :  'my_genus',
            'sp'        :  'my_species',
            'cnt'       :  'chile',
            'loc'       :  'botanical gardens',
            'date'      :  '2021-02-03',
            'length'    :  10,
            'file-name' :  'bluebell.mp3',
            'type'      :  'call',
            'url'       :  'http://my_server/file_ingo'
            }
        rec = XenoCantoRecording(recording_metadata, load_dir)
        return rec

# -------------- Main ---------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()