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
        self.file = h5py.File(models_fname, "r")
        self._init_models_group()
        self._init_classifier()
        
    def _init_models_group(self):
        self.models_group = self.file[self.models_group_path]
    
    def _init_classifier(self):
        # Loading this pickled object requires its class (e.g. MultinomialNB)
        # to be part of the current namespace. Alternative is to use the
        # _get_params() and set_params() methods from the BaseEstimator class
        pickled_str = self.models_group["classifier_pickle"].value
        self.classifier = cPickle.loads(pickled_str)
        
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
        # this part is specifc to NB
        self.target_names = model["target_names"]
        self.classifier.unique_y = np.arange(len(self.target_names))   
        self.classifier.class_log_prior_ = model["class_log_prior_"]
        self.classifier.feature_log_prob_ = model["feature_log_prob_"]
        
    def _predict(self, context_vec):
        """
        return a dict mapping target names to scores
        """
        preds = self.classifier.predict_proba(context_vec)
        return dict(zip(self.target_names, preds[0]))
        
        
    
    
    
    
    