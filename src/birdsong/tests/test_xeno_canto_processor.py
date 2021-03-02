'''
Created on Mar 1, 2021

@author: paepcke
'''
import json5
import os
import unittest
import tempfile

from logging_service.logging_service import LoggingService

from birdsong.xeno_canto_processor import XenoCantoCollection, XenoCantoRecording


TEST_ALL = True
#TEST_ALL = False

class XenoCantoProcessorTester(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.tst_dir = os.path.join(cls.curr_dir,
                                   'data/xeno_canto_tst_data'
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
    
    #****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
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

# -------------- Main ---------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()