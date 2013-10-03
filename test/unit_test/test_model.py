"""
test model building
"""

import tempfile

import h5py

from sklearn.naive_bayes import MultinomialNB

from tg.ambig import AmbiguityMap
from tg.sample import DataSetGenerator
from tg.model import ModelBuilder
from tg.config import config

import logging
logging.basicConfig(level=logging.INFO)


class TestModelBuilder:
    
    def test_model_builder(self):
        # get ambiguity map
        ambig_fname = config["test_data_dir"] +"/de-en_ambig.tab"
        ambig_map = AmbiguityMap(ambig_fname)
        
        # get samples 
        samp_hdf_fname = config["test_data_dir"] + "/de-en_samples.hdf5_"
        samp_hdfile = h5py.File(samp_hdf_fname, "r")
        
        # create data generator
        data_gen = DataSetGenerator(ambig_map, samp_hdfile)

        # build
        models_hdf_fname = tempfile.NamedTemporaryFile().name
        builder = ModelBuilder(
            data_gen,
            models_hdf_fname = models_hdf_fname,
            classifier = MultinomialNB() )
        builder.run()
        
        # TODO: how to verify the model?
        
