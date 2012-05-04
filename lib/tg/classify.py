"""
classification of translation candidates
"""

# TODO: units tests
# TODO: generalize to other classification models

import logging
import cPickle

import h5py
import numpy as np

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

    
log = logging.getLogger(__name__)


class NaiveBayesClassifier(object):
    """
    Naive Bayes classification from models stored in HDF5 file
    
    Works with MultinomialNB or BernoulliNB models
    """
    
    models_group_path = "/models"
    
    def __init__(self, models_fname):
        self._load_models(models_fname)
        self._init_models_group()
        self._init_vocab()
        self._init_classifier()
        
    def _load_models(self, models_fname):
        log.info("loading models from " + models_fname)
        self.file = h5py.File(models_fname, "r")
        
    def _init_vocab(self):
        self.vocab = dict((lemma.decode("utf-8"),i) for i,lemma in enumerate(self.file["vocab"]))
        log.info("loaded vocabulary ({0} terms)".format(len(self.vocab)))
    
    def _init_models_group(self):
        self.models_group = self.file[self.models_group_path]
        log.info("loaded {} models".format(len(self.models_group)))
    
    def _init_classifier(self):
        # Loading this pickled object requires its class (e.g. MultinomialNB)
        # to be part of the current namespace. Alternative is to use the
        # _get_params() and set_params() methods from the BaseEstimator class
        pickled_str = self.models_group["classifier_pickle"].value
        self.classifier = cPickle.loads(pickled_str)
        log.info("loaded classifier {0}".format(self.classifier))
        
    def score(self, source_lempos, context_vec):
        """
        score translation candidates for source lempos combination,
        returning a dict mapping target lempos combinations to scores
        """
        try:
            model = self.models_group[source_lempos]
        except KeyError:
            # no model available for lempos
            return {}  

        self._update_classifier(model)
        return self._predict(context_vec)
        
    def _update_classifier(self, model):
        """
        update classifier object with model data
        """
        # target names are stored as utf-8 encoded bytestrin (HDF cannot handle unicode strings) 
        # so these need to be decode to unicode strings
        self.target_names = [name.decode("utf-8") for name in model["target_names"]]
        self.classifier.unique_y = np.arange(len(self.target_names))   
        self.classifier.class_log_prior_ = model["class_log_prior_"]
        self.classifier.feature_log_prob_ = model["feature_log_prob_"]
        
    def _predict(self, context_vec):
        """
        return a dict mapping target names to scores
        """
        preds = self.classifier.predict_proba(context_vec)
        return dict(zip(self.target_names, preds[0]))
        
        
    
    
    
    
    