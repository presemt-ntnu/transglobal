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

from sklearn.feature_selection.univariate_selection import (
    _AbstractUnivariateFilter )
from sklearn.feature_selection.rfe import RFE

from tg.store import DisambiguatorStore
from tg.utils import coo_matrix_from_hdf5
from tg.sample import AmbiguityMap


log = logging.getLogger(__name__)



class ModelBuilder(object):
    """
    Class for fitting disambiguators on samples and storing them in hdf5 file
    """
    
    def __init__(self, tab_fname, samp_hdf_fname, models_hdf_fname,
                 classifier, graphs_pkl_fname=None, with_vocab_mask=False):
        self.tab_fname = tab_fname
        self.samp_hdf_fname = samp_hdf_fname
        self.models_hdf_fname = models_hdf_fname
        self.classifier = classifier
        self.graphs_pkl_fname = graphs_pkl_fname
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
        
        self.extract_source_lempos_subset()
        self.read_ambig_map()

    def read_ambig_map(self):
        self.ambig_map = AmbiguityMap(fn = self.tab_fname)


    def extract_source_lempos_subset(self):
        """
        extract all required source lempos from pickled graphs,
        where POS tag is the *lexicon* POS tag
        """
        if self.graphs_pkl_fname:
            log.info("extracting source lempos subset")
            self.source_lempos_subset = set()
            
            for graph in cPickle.load(open(self.graphs_pkl_fname)):
                for _,node_attr in graph.source_nodes_iter(data=True, 
                                                           ordered=True):
                    try:
                        self.source_lempos_subset.add(
                            " ".join(node_attr["lex_lempos"]))
                    except KeyError:
                        # not found in lexicon
                        pass
        else:
            self.source_lempos_subset = None
        
    def build(self):
        for source_lempos in self.ambig_map.source_iter():
            data = None
            
            for target_lempos in self.ambig_map[source_lempos]:
                try:
                    samp_group = self.samples[target_lempos]
                except KeyError:
                    # should not happen
                    log.warning("found no sample for " + target_lempos)
                    continue
                
                samp_mat = coo_matrix_from_hdf5(samp_group, dtype="f8")
                
                if not data:
                    # start new data set and targets
                    data = samp_mat
                    targets = np.zeros((samp_mat.shape[0],))
                    target_count = 0
                    target_names = []
                else:
                    # append to data and targets
                    data = sp.vstack([data, samp_mat])
                    # concat new targets corresponding to number of samples
                    new_targets = np.zeros((samp_mat.shape[0],)) + target_count
                    targets = np.hstack((targets, new_targets))
                
                target_count += 1
                # hdf5 cannot store array of unicode strings, so use byte
                # strings for target names
                target_names.append(target_lempos.encode("utf-8"))
                
            if data:
                self.build_disambiguator(source_lempos, data, targets,
                                         target_names)
            
    def _get_source_to_target_lempos_map(self):
        sl2tl = {}
        
        for line in codecs.open(self.tab_fname, encoding="utf8"):
            source_lempos, target_lempos = self._parse_line(line)
            
            if self._skip(source_lempos):
                log.debug(u"skipping model for {} -> {}".format(
                    source_lempos, target_lempos))
                continue
            
            sl2tl.setdefault(source_lempos, []).append(target_lempos)
            
        return sl2tl
            
    def _parse_line(self, line):
            source_label, target_label = line.rstrip().split("\t")[1:3]
            # strip corpus POS tag
            return ( source_label.rsplit("/", 1)[0], 
                     target_label.rsplit("/", 1)[0] )
        
    def _skip(self, source_lempos):
        return ( self.source_lempos_subset and 
                 source_lempos not in self.source_lempos_subset )
            
    def build_disambiguator(self, source_lempos, data, targets, target_names):
        log.info("building disambiguator for " + source_lempos)
        # it seems scilearn classes want sparse matrices in csr format
        ##try:
            ##self.classifier.fit(data.tocsr(), targets)  
        ##except ValueError:
            ### FIXME: this happens when there are no features selected by e.g.
            ### SelectFpr
            ##log.error("No model created!")
            ##return 
            
        self.classifier.fit(data.tocsr(), targets)  
        self.models_hdfile.store_fit(source_lempos, self.classifier)
        self.models_hdfile.save_target_names(source_lempos, target_names)
        if self.with_vocab_mask:        
            self.save_vocab_mask(source_lempos)
        self.disambiguator_count += 1  
        
    def save_vocab_mask(self, lempos):
        # assume classifier is a fitted pipeline
        mask = np.arange(len(self.vocab))
        
        for name, transformer in self.classifier.steps[:-1]:
            # Is transformer a feature selector?
            if isinstance(transformer, (_AbstractUnivariateFilter, RFE)):
                mask = transformer.transform(mask)
                
        self.models_hdfile.save_vocab_mask(lempos, mask[0])
        
    def finish(self):
        log.info("closing samples file " + self.samp_hdf_fname)    
        self.sample_hdfile.close()
        log.info("saved {} models".format(self.disambiguator_count))
        log.info("closing models file " + self.models_hdf_fname)    
        self.models_hdfile.close()
        elapsed_time = time.time() - self.start_time
        size = os.path.getsize(self.models_hdf_fname) / float(1024.0 ** 2)
        log.info("elapsed time: {0}".format(
            datetime.timedelta(seconds=elapsed_time)))
        log.info("average time per model: {0}".format(
            datetime.timedelta(seconds=elapsed_time/
                               float(self.disambiguator_count))))
        log.info("models file size: {0:.2f} MB".format(size))
        log.info("average model size: {:.2f} MB".format(
            size/ float(self.disambiguator_count)))
        


