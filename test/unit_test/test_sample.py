# -*- coding: utf-8 -*-


from nose.tools import raises
from scipy.sparse import coo_matrix

import h5py

from tg.config import config
from tg.sample import DataSetGenerator
from tg.ambig import AmbiguityMap

 

class TestDataSetGenerator:
    
    ambig_fname = config["test_data_dir"] +"/de-en_ambig.tab"         
    ambig_map = AmbiguityMap(ambig_fname)  
    samp_hdf_fname = config["test_data_dir"] +"/de-en_samples.hdf5_" 
    samp_hdfile = h5py.File(samp_hdf_fname, "r") 
    dsg = DataSetGenerator(ambig_map, samp_hdfile)
    
    def test_get_samples(self):
        samp_mat = self.dsg._get_sample_mat(u"dial/n")
        assert isinstance(samp_mat, coo_matrix)
        assert samp_mat.shape == (10000, 182479)
        
    @raises(KeyError)
    def test_get_samples_exception(self):
        self.dsg._get_sample_mat(u"x/y")
        
    def test_get_data_and_target(self):
        target_lempos = [u"delicious/jj", u"rich/jj"]
        data_set = self.dsg._get_labeled_data(None, target_lempos)
        assert data_set.samples.shape == (20000,182479)
        assert data_set.targets.shape[0] ==  data_set.samples.shape[0]
        assert data_set.target_lempos == target_lempos
        
    def test_get_data_and_target_2(self):
        # silently skip lempos for which there are no samples
        target_lempos = [u"x/y", u"delicious/jj", u"rich/jj"]
        data_set = self.dsg._get_labeled_data(None, target_lempos)
        assert data_set.samples.shape == (20000,182479)
        assert data_set.targets.shape[0] ==  data_set.samples.shape[0]
        assert data_set.target_lempos == target_lempos[1:]
        
    def test_iter(self):      
        data_sets = list(self.dsg)
        assert len(data_sets) == 3
        
        
        
        
        
        
    