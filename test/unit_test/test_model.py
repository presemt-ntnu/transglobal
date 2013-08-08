"""
test model building
"""

import tempfile

from sklearn.naive_bayes import MultinomialNB

from tg.ambig import AmbiguityMap
from tg.model import ModelBuilder
from tg.config import config

import logging
logging.basicConfig(level=logging.INFO)


class TestModelBuilder:
    
    def test_model_builder(self):
        ambig_fname = config["test_data_dir"] +"/de-en_ambig.tab"
        ambig_map = AmbiguityMap(ambig_fname)
        
        samp_hdf_fname = config["test_data_dir"] + "/de-en_samples.hdf5_"
        
        models_hdf_fname = tempfile.NamedTemporaryFile().name
        
        builder = ModelBuilder(
            ambig_map = ambig_map,
            samp_hdf_fname = samp_hdf_fname,
            models_hdf_fname = models_hdf_fname,
            classifier = MultinomialNB() )
                
        builder.run()
        
