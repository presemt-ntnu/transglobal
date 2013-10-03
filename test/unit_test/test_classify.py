# -*- coding: utf-8 -*-

"""
test classification of translation candidates
"""

import logging as log
import tempfile
import operator

import h5py

from sklearn.naive_bayes import MultinomialNB

from tg.config import config
from tg.ambig import AmbiguityMap
from tg.classify import TranslationClassifier
from tg.utils import coo_matrix_from_hdf5



class TestTranslationClassifier:
    
    def test_translation_classifier(self):
        models_hdf_fname = config["test_data_dir"] + "/de-en_models.hdf5_"
        
        # make a translation classifier that uses this model
        trans_clf = TranslationClassifier(models_hdf_fname)
        
        # load a couple of vectors from the samples (i.e. the training
        # material) to test the translation classifier
        f = h5py.File(config["test_data_dir"] + "/de-en_samples.hdf5_", "r")
        source_lempos = "Teller/n"
        targets = "basket/n dial/n disc/n dish/n disk/n plate/n".split()
        
        for target_lempos in targets: 
            log.info(u"True translation = " + target_lempos)
            
            m = coo_matrix_from_hdf5(f["/samples/" + target_lempos])
            m = m.tocsr()
            
            for vector in m[:10]:
                scores = trans_clf.score(source_lempos, vector)
                best = sorted(scores.items(), key=operator.itemgetter(1))[-1]
                log.info(u"Predicted translation = {} (P={})".format(*best))
                
        