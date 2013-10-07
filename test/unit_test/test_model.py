"""
test model building
"""

import tempfile

import h5py

from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import SGDClassifier

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
        
    def test_model_builder_with_class_weights_nb(self):
        # get ambiguity map
        ambig_fname = config["test_data_dir"] +"/de-en_ambig.tab"
        ambig_map = AmbiguityMap(ambig_fname)
        
        # get samples 
        samp_hdf_fname = config["test_data_dir"] + "/de-en_samples.hdf5_"
        samp_hdfile = h5py.File(samp_hdf_fname, "r")
        
        # create data generator
        data_gen = DataSetGenerator(ambig_map, samp_hdfile)
        
        # get lemma counts
        # TODO: reading all counts is slow - file with subset of counts needed
        counts_fname = config["count"]["lemma"]["en"]["pkl_fname"]

        # build
        models_hdf_fname = tempfile.NamedTemporaryFile().name
        builder = ModelBuilder(
            data_gen,
            models_hdf_fname = models_hdf_fname,
            classifier = MultinomialNB(),
            counts_fname=counts_fname)
        builder.run()
        
    def test_model_builder_with_class_weights_sgd(self):
        # get ambiguity map
        ambig_fname = config["test_data_dir"] +"/de-en_ambig.tab"
        ambig_map = AmbiguityMap(ambig_fname)
        
        # get samples 
        samp_hdf_fname = config["test_data_dir"] + "/de-en_samples.hdf5_"
        samp_hdfile = h5py.File(samp_hdf_fname, "r")
        
        # create data generator
        data_gen = DataSetGenerator(ambig_map, samp_hdfile)
        
        # get lemma counts
        # TODO: reading all counts is slow - file with subset of counts needed
        counts_fname = config["count"]["lemma"]["en"]["pkl_fname"]

        from sklearn.linear_model import SGDClassifier
        # build
        models_hdf_fname = tempfile.NamedTemporaryFile().name
        builder = ModelBuilder(
            data_gen,
            models_hdf_fname = models_hdf_fname,
            classifier = SGDClassifier(),
            counts_fname=counts_fname)
        builder.run()
        
