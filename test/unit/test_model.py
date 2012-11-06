"""
test model building
"""

import tempfile

from sklearn.naive_bayes import MultinomialNB

from tg.model import ModelBuilder
from tg.config import config

import logging as log
log.basicConfig(level=log.INFO)


class TestModelBuilder:
    
    def test_model_builder(self):
        models_hdf_fname = tempfile.NamedTemporaryFile().name
        
        builder = ModelBuilder(
            tab_fname = config["test_data_dir"] +"/de-en_ambig.tab",
            samp_hdf_fname = config["test_data_dir"] + "/de-en_samples.hdf5_",
            models_hdf_fname = models_hdf_fname,
            classifier = MultinomialNB() )
                
        builder.run()
        
