"""
classification of translation candidates
"""

# TODO: units tests

import logging
import cPickle

import h5py
import numpy as np

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.validation import atleast2d_or_csr

    
log = logging.getLogger(__name__)


class TranslationClassifier(object):
    """
    Abstract base class for translation classification with models stored in
    HDF5 file
    """
    
    models_group_path = "/models"
    classifier_classes = ()
    
    def __init__(self, models_fname):
        self._load_models(models_fname)
        self._init_models_group()
        self._init_vocab()
        self._init_classifier()
        
    def _load_models(self, models_fname):
        log.info("loading models from " + models_fname)
        self.file = h5py.File(models_fname, "r")
        
    def _init_vocab(self):
        self.vocab = dict((lemma.decode("utf-8"),i) 
                          for i,lemma in enumerate(self.file["vocab"]))
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
        assert isinstance(self.classifier, self.classifier_classes)
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
        self._update_target_names(model)
        context_vec = self._select_feats(model, context_vec)
        return self._predict(context_vec)
    
    def _update_target_names(self, model):
        # target names are stored as utf-8 encoded bytestrin (HDF cannot
        # handle unicode strings) so these need to be decoded to unicode
        # strings
        self.target_names = [name.decode("utf-8") 
                             for name in model["target_names"]]

        
    def _update_classifier(self, model):
        """
        update classifier object with model data
        """
        raise NotImplementedError
        
    def _select_feats(self, model, context_vec):
        try:
            mask = model["mask"]
        except KeyError:
            return context_vec
        else:
            return context_vec[0, mask]
        
    def _predict(self, context_vec):
        """
        return a dict mapping target names to scores
        """
        raise NotImplementedError
    
    
    
class NaiveBayesClassifier(TranslationClassifier):
    """
    Translation classification with Naive Bayes classifiers (MultinomialNB or
    BernoulliNB)
    """
    
    classifier_classes = (MultinomialNB, BernoulliNB)
        
    def _update_classifier(self, model):
        """
        update classifier object with model data
        """        
        self.classifier.unique_y = np.arange(len(self.target_names))   
        self.classifier.class_log_prior_ = model["class_log_prior_"]
        # TODO:
        # _joint_log_likelihood method of Naive Bayes transposes
        # feature_log_prob_ vector, which is not suported by h5py, so convert
        # to numpy array. This seems unncessary.
        self.classifier.feature_log_prob_ = model["feature_log_prob_"][:]
        
    def _predict(self, context_vec):
        """
        return a dict mapping target names to scores
        """
        preds = self.classifier.predict_proba(context_vec)
        return dict(zip(self.target_names, preds[0]))
    
    
    
from scipy.spatial.distance import cosine    
    
class NearestCentroidClassifier(TranslationClassifier):
    """
    Translation classification with Nearest Centroid (Rocchio) classifiers
    """
    
    classifier_classes = (NearestCentroid,)
    
    def _update_classifier(self, model):
        """
        update classifier object with model data
        """
        # self._classes_ is only used by the predict method,
        # which is currently not used (see _predict method below)
        #self.classifier.classes_ = np.arange(len(self.target_names))
        self.classifier.centroids_ = model["centroids_"]
        
    def _predict(self, context_vec):
        """
        return a dict mapping target names to scores
        """
        # TODO: this is a hack for now, because NearestCentroid offers no
        # method to retrieve distances to all classes (cf. predict method)
        context_vec = atleast2d_or_csr(context_vec)
        ##context_vec /= context_vec.sum().astype(float)
        context_vec = context_vec.astype("f8").todense()
        distances = pairwise_distances(context_vec, 
                                       self.classifier.centroids_, 
                                       metric=cosine)#self.classifier.metric)
        ##print distances
        # normalize so that sum of distances equals one
        distances = 1.0 - distances
        distances /= distances.sum()
        return dict(zip(self.target_names, distances[0]))