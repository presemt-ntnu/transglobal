"""
build disambiguation models
"""

import codecs
import cPickle
import datetime
import logging
import os
import time

import numpy as np
import scipy.sparse as sp
import h5py

try:
    from sklearn.feature_selection.univariate_selection import _BaseFilter
except ImportError:
    # sklearn<0.13
    from sklearn.feature_selection.univariate_selection import\
         _AbstractUnivariateFilter as _Basefilter

try:
    from sklearn.feature_selection.from_model import _LearntSelectorMixin
except ImportError:
    # sklearn<0.13
    from sklearn.feature_selection.selector_mixin import SelectorMixin as\
         _LearntSelectorMixin

from sklearn.feature_selection.rfe import RFE
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BaseDiscreteNB

from tg.store import DisambiguatorStore
from tg.utils import coo_matrix_from_hdf5
from tg.ambig import AmbiguityMap
from tg.sample import DataSetGenerator
from tg.exception import TGException


log = logging.getLogger(__name__)


class ModelBuilder(object):
    """
    Class for fitting disambiguators on samples and storing them in HDF5 file.
    
    Parameters
    ----------
    data_generator: DataSetGenerator instance
        Labeled samples generator
    samp_hdf_fname: str
        Name of HDF5 file containing context samples
    models_hdf_fname: str
        Name of HDF5 file for storing disambiguation models
    classifier: classifier instance
        Classifier instance from sklearn, to be fitted for each ambiguous 
        source lempos. Possibly a Pipeline instance.
    counts_fname: str
        Name of pickle file containing lemma counts. If used in combination 
        with  classifier that supports class weights (e.g. SGDClassifier)
        or class priors (e.g. Naive Bayes), classes will be weighted 
        according to their lemma frequency.
    with_vocab_mask: bool, optional
        If true, a vocabulary mask is stored for each model. This is required 
        when the classifier performs feature selection and the vocabulary has
        to be pruned accordingly.
    """
    
    # feature selectors that reduce the vocabulary
    FEATURE_SELECTORS = _BaseFilter, RFE, _LearntSelectorMixin
    
    def __init__(self, data_generator, models_hdf_fname, classifier,
                 counts_fname=None, with_vocab_mask=False, **kwargs):
        self.data_generator = data_generator
        self.models_hdf_fname = models_hdf_fname
        self.classifier = classifier
        self.counts_fname = counts_fname
        self.with_vocab_mask = with_vocab_mask
        
    def run(self):
        """
        build disambiguation models
        """
        self._prepare()
        self._build()
        self._finish()

    def _prepare(self):
        self.start_time = time.time() 
        self.disambiguator_count = 0
        
        log.info("creating models file " + self.models_hdf_fname)
        self.models_hdfile = DisambiguatorStore(self.models_hdf_fname, "w")
        
        self.models_hdfile.save_estimator(self.classifier)
        
        # FIXME: hmm, a bit sneaky...
        self.models_hdfile.copy_vocab(self.data_generator.samp_hdfile)
        
        if self.counts_fname:
            log.info("reading counts from " + self.counts_fname)
            self.counts_dict = cPickle.load(open(self.counts_fname))
        
        if self.with_vocab_mask:
            log.info("storage with vocabulary masks")
            self.vocab = self.models_hdfile.load_vocab()
            
    def _build(self):
        for data_set in self.data_generator:
            if data_set.target_lempos:
                self._build_disambiguator(data_set)
            else:
                log.error("no samples and thus no disambiguation models for " +
                          data_set.source_lempos)
            
    def _build_disambiguator(self, data_set):
        log.info(u"building disambiguator for {} with {} translations".format(
            data_set.source_lempos, len(data_set.target_lempos)))
        
        if self.counts_fname:
            self._set_class_weights(data_set.target_lempos)
        
        try:
            self.classifier.fit(data_set.samples, data_set.targets)  
        except ValueError, error:
            if ( error.args in [
                ("zero-size array to reduction operation maximum which has no "
                 "identity",),
                ("zero-size array to maximum.reduce without identity",),
                ('Invalid threshold: all features are discarded.',)] ):
                # this happens when there are no features selected 
                # e.g. when using SelectFpr
                log.error("No model created, because no features selected!")
                return
            else:
                raise error
            
        self.models_hdfile.store_fit(data_set.source_lempos, self.classifier)
        self.models_hdfile.save_target_names(data_set.source_lempos,
                                             data_set.target_lempos)
        if self.with_vocab_mask:        
            self._save_vocab_mask(data_set.source_lempos)
        self.disambiguator_count += 1  
        
    def _set_class_weights(self, target_lempos):
        """
        set class weights attribute of classifier
        """
        classifier = self._get_classifier()
        
        if hasattr(classifier, "class_prior"):
            # Naive Bayes with weights as list
            setattr(classifier, "class_prior", 
                    self._get_class_weights(target_lempos, as_list=True))
        elif hasattr(classifier, "class_weight"):
            # e.g. SGDClassifier with weights as dicts
            setattr(classifier, "class_weight",
                    self._get_class_weights(target_lempos))
        else:
            raise TGException("classifier {} does not support class "
                              "weights".format(classifier.__class__.__name__))
                
    def _get_class_weights(self, target_lempos, as_list=False):
        """
        compute normalized weight for each class according its lemma frequency
        """
        class_weights = np.zeros(len(target_lempos))
        
        for i, lempos in enumerate(target_lempos):
            lemma = lempos.rsplit("/", 1)[0].decode("utf-8")
            try:
                class_weights[i] = self.counts_dict[lemma]
            except KeyError:
                raise TGException(u"{} contains no count for lemma {}".format(
                self.counts_fname, lemma))
            
        class_weights /= class_weights.sum()
        #log.info(zip(target_lempos, class_weights))
        
        if as_list:
            return class_weights
        else:
            # target_lempos is ordered so target_lempos[0] corresponds to  
            # targets[0] and so on
            return dict(enumerate(class_weights))
    
    def _reset_class_weights(self):
        """
        reset class weights attribute of classifier to None
        """
        classifier = self._get_classifier()
            
        for attr in "class_prior", "class_weight":
            if hasattr(classifier, attr):
                setattr(classifier, attr, None)
                break   
            
    def _get_classifier(self):
        """
        get classifier, possibly embedded in sklearn Pipeline
        """
        if isinstance(self.classifier, Pipeline):
            return self.classifier.steps[-1][-1]
        else:
            return self.classifier            
        
    def _save_vocab_mask(self, lempos):
        # assume classifier is a fitted pipeline
        mask = np.arange(len(self.vocab))
        
        for _, transformer in self.classifier.steps[:-1]:
            # Is transformer a feature selector?
            if isinstance(transformer, self.FEATURE_SELECTORS):
                mask = transformer.transform(mask)
                
        # some transformers (e.g. SelectKBest) return a 2-dimensional array,
        # whereas others (e.g. classes using SelectorMixin) return a
        # 1-dimensional array
        mask = mask.flatten()
                
        self.models_hdfile.save_vocab_mask(lempos, mask)
        
    def _finish(self):
        log.info("closing models file " + self.models_hdf_fname)    
        self.models_hdfile.close()
        
        if self.counts_fname:
            # remove injected class priors, in case classifier is reused
            self._reset_class_weights()
        
        if self.disambiguator_count > 0:
            log.info("saved {} models".format(self.disambiguator_count))
            elapsed_time = time.time() - self.start_time
            size = os.path.getsize(self.models_hdf_fname) / float(1024.0 ** 2)
            log.info("elapsed time: {0}".format(
                datetime.timedelta(seconds=elapsed_time)))
            average_time = datetime.timedelta(seconds=elapsed_time/
                float(self.disambiguator_count))                              
            log.info("average time per model: {0}".format(average_time))
            log.info("models file size: {0:.2f} MB".format(size))
            average_size = size/ float(self.disambiguator_count)
            log.info("average model size: {:.2f} MB".format(average_size))
        else:
            log.error("No disambiguation models were created!")  
            
        
