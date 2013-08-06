"""
build classification models
"""

# TODO:
# - dco strings & comments


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
    from sklearn.feature_selection.univariate_selection import _AbstractUnivariateFilter as _Basefilter
    


from sklearn.feature_selection.rfe import RFE
from sklearn.feature_selection.selector_mixin import SelectorMixin

from tg.store import DisambiguatorStore
from tg.utils import coo_matrix_from_hdf5
from tg.ambig import AmbiguityMap
from tg.sample import DataSetGenerator


log = logging.getLogger(__name__)


class ModelBuilder(object):
    """
    Class for fitting disambiguators on samples and storing them in hdf5 file
    """
    
    # feature selectors that reduce the vocabulary
    FEATURE_SELECTORS = _BaseFilter, RFE, SelectorMixin
    
    def __init__(self, ambig_map, samp_hdf_fname, models_hdf_fname,
                 classifier, with_vocab_mask=False):
        self.ambig_map = ambig_map
        self.samp_hdf_fname = samp_hdf_fname
        self.models_hdf_fname = models_hdf_fname
        self.classifier = classifier
        self.disambiguator_count = 0
        self.with_vocab_mask = with_vocab_mask
        
    def run(self):
        self.prepare()
        self.build()
        self.finish()

    def prepare(self):
        self.start_time = time.time() 
        
        log.info("opening samples file " + self.samp_hdf_fname)
        self.sample_hdfile = h5py.File(self.samp_hdf_fname, "r")
        self.samples = self.sample_hdfile["samples"]
        
        log.info("creating models file " + self.models_hdf_fname)
        self.models_hdfile = DisambiguatorStore(self.models_hdf_fname, "w")
        
        self.models_hdfile.save_estimator(self.classifier)
        
        self.models_hdfile.copy_vocab(self.sample_hdfile)
        
        if self.with_vocab_mask:
            log.info("storage with vocabulary masks")
            self.vocab = self.models_hdfile.load_vocab()
            
    def build(self):
        for data_set in DataSetGenerator(self.ambig_map,
                                         self.sample_hdfile):
            if data_set.samples:
                self.build_disambiguator(data_set)
            else:
                log.error("No samples and thus no disambiguation models for " +
                          data_set.source_lempos)
            
    def build_disambiguator(self, data_set):
        log.info("building disambiguator for " + data_set.source_lempos)
        # it seems scilearn classes want sparse matrices in csr format
        try:
            self.classifier.fit(data_set.samples.tocsr(), data_set.targets)  
        except ValueError, error:
            if ( error.args in [
                ('zero-size array to reduction operation maximum which has no identity',),
                ("zero-size array to maximum.reduce without identity",),
                ('Invalid threshold: all features are discarded.',)] ):
                # this happens when there are no features selected 
                # e.g. when using SelectFpr
                log.error("No model created, because no features selected!")
                return
            else:
                raise error
            
        self.models_hdfile.store_fit(data_set.source_lempos, self.classifier)
        self.models_hdfile.save_target_names(data_set.source_lempos, data_set.target_lempos)
        if self.with_vocab_mask:        
            self.save_vocab_mask(data_set.source_lempos)
        self.disambiguator_count += 1  
        
    def save_vocab_mask(self, lempos):
        # assume classifier is a fitted pipeline
        mask = np.arange(len(self.vocab))
        
        for name, transformer in self.classifier.steps[:-1]:
            # Is transformer a feature selector?
            if isinstance(transformer, self.FEATURE_SELECTORS):
                mask = transformer.transform(mask)
                
        # some transformers (e.g. SelectKBest) return a 2-dimensional array,
        # whereas ohers (e.g. classes using SelectorMixin) return a
        # 1-dimensional array
        mask = mask.flatten()
                
        self.models_hdfile.save_vocab_mask(lempos, mask)
        
    def finish(self):
        log.info("closing samples file " + self.samp_hdf_fname)    
        self.sample_hdfile.close()
        log.info("closing models file " + self.models_hdf_fname)    
        self.models_hdfile.close()
        
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
        


