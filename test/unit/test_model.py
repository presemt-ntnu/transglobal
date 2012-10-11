"""
test model building
"""

import tempfile

from sklearn.naive_bayes import MultinomialNB

from tg.model import ModelBuilder

from setup import test_data_dir



class TestModelBuilder:
    
    def test_model_builder(self):
        models_hdf_fname = tempfile.NamedTemporaryFile().name
        
        builder = ModelBuilder(
            tab_fname = test_data_dir +"/de-en_ambig.tab",
            samp_hdf_fname = test_data_dir + "/de-en_samples.hdf5_",
            models_hdf_fname = models_hdf_fname,
            classifier = MultinomialNB() )
                
        builder.run()
        
if __name__ == "__main__":
    import logging as log
    log.basicConfig(level=log.INFO)
    
    import nose, sys
    sys.argv.append("-v")
    nose.run(defaultTest=__name__)