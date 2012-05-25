"""
build classification models
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

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import NearestCentroid

from tg.config import config
from tg.utils import coo_matrix_from_hdf5

log = logging.getLogger(__name__)



class ModelBuilder(object):
    """
    Abstract base class for training translation classifiers on samples and
    storing the models in a hdf5 file
    """
    
    def __init__(self, tab_fname, samp_hdf_fname, models_hdf_fname,
                 classifier, graphs_pkl_fname=None, feat_selector=None):
        self.tab_fname = tab_fname
        self.samp_hdf_fname = samp_hdf_fname
        self.models_hdf_fname = models_hdf_fname
        self.classifier = classifier
        self.graphs_pkl_fname = graphs_pkl_fname
        self.feat_selector = feat_selector
        
    def run(self):
        self.prepare()
        self.build()
        self.finish()

    def prepare(self):
        self.start_time = time.time() 
        self.open_samples_file()
        self.open_models_file()
        self.save_classifier()
        self.save_vocab()
        self.extract_source_lempos_subset()
        
    def finish(self):
        log.info("closing samples file " + self.samp_hdf_fname)    
        self.sample_hdfile.close()
        n = len(self.models)
        log.info("saved {} models".format(n))
        log.info("closing models file " + self.models_hdf_fname)    
        self.models_hdfile.close()
        elapsed_time = time.time() - self.start_time
        size = os.path.getsize(self.models_hdf_fname) / float(1024.0 ** 2)
        log.info("elapsed time: {0}".format(datetime.timedelta(seconds=elapsed_time)))
        log.info("average time per model: {0}".format(
            datetime.timedelta(seconds=elapsed_time/float(n))))
        log.info("models file size: {0:.2f} MB".format(size))
        log.info("average model size: {:.2f} MB".format(size / float(n)))
        
    def open_samples_file(self):    
        log.info("opening samples file " + self.samp_hdf_fname)
        self.sample_hdfile = h5py.File(self.samp_hdf_fname, "r")
        self.samples = self.sample_hdfile["samples"]

    def open_models_file(self):
        log.info("creating models file " + self.models_hdf_fname)
        self.models_hdfile = h5py.File(self.models_hdf_fname, "w")
        self.models = self.models_hdfile.create_group("models")

    def save_classifier(self):
        # Pickle classifier and include in hdf5 file.
        # This saves the parameters from __init__.
        # This is before a call to fit(), so class_log_prior_ and 
        # feature_log_prob_ are excluded.
        # Loading this pickled object requires its class (e.g. MultinomialNB)
        # to be part of the current namespace.
        # Alternative is to use the _get_params() and set_params() methods
        # from the BaseEstimator class
        log.info("saving classifier {0}".format(self.classifier))
        self.models["classifier_pickle"] = cPickle.dumps(self.classifier) 
        

    def save_vocab(self):
        log.info("saving vocabulary ({0} terms)".format(
            len(self.sample_hdfile["vocab"])))
        # create new type for variable-length strings
        # see http://code.google.com/p/h5py/wiki/HowTo#Variable-length_strings
        str_type = h5py.new_vlen(str)
        self.models_hdfile.create_dataset("vocab",
                                          data=self.sample_hdfile["vocab"], 
                                          dtype=str_type)
        
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
                        self.source_lempos_subset.add(" ".join(node_attr["lex_lempos"]))
                    except KeyError:
                        # not found in lexicon
                        pass
        else:
            self.source_lempos_subset = None
        
    def build(self):
        prev_source_lempos = None
        models_count = 0
        
        for line in codecs.open(self.tab_fname, encoding="utf8"):
            source_label, target_label = line.rstrip().split("\t")[1:3]
            # strip corpus POS tag
            source_lempos = source_label.rsplit("/", 1)[0]
            target_lempos = target_label.rsplit("/", 1)[0]
            
            if self.source_lempos_subset and source_lempos not in self.source_lempos_subset:
                #log.debug(u"skipping model for {} -> {}".format(source_lempos,
                #                                               target_lempos))
                continue
    
            try:
                samp_group = self.samples[target_lempos]
            except KeyError:
                # should not happen
                log.warning("found no sample for " + target_lempos)
                continue
                    
            sm = coo_matrix_from_hdf5(samp_group)        
            
            # hdf5 cannot store array of unicode strings, so use byte strings for
            # label names
            target_lempos = target_lempos.encode("utf-8")
            
            if source_lempos == prev_source_lempos:
                if target_lempos in target_names:
                    # this is due to an old bug in the code that finds
                    # translation ambiguities in the lexicon - test becomes
                    # redundant in the future
                    log.warn(u"skipping duplicate target lempos " + target_lempos.decode("utf-8"))
                else:
                    data = sp.vstack([data, sm])
                    target_count += 1
                    target_names.append(target_lempos)
                    # concat new targets depending on number of instances
                    new_targets = np.zeros((sm.shape[0],)) + target_count
                    targets = np.hstack((targets, new_targets))
            else:
                if prev_source_lempos and target_count:
                    # it seems scilearn classes want sparse matrices in csr format  
                    data = data.tocsr()
                    self.build_single_model(prev_source_lempos, data, targets, target_names)
                    models_count += 1
                    
                # init data for a new model
                data = sm
                targets = np.zeros((sm.shape[0],))
                target_count = 0
                target_names = [target_lempos]
                
            prev_source_lempos = source_lempos
            
    def build_single_model(self, source_lempos, data, targets, target_names):
        log.info("building classifier model for " + source_lempos)
        group = self.models.create_group(source_lempos)
        self.store_target_names(group, target_names)
        if self.feat_selector:        
            data = self.select_features(group, data, targets)
        self.fit_classifier(group, data, targets, target_names)
        
    def select_features(self, group, data, targets):  
        data = self.feat_selector.fit_transform(data, targets)
        log.debug("selected {0} features with {1}".format(
            data.shape[1], self.feat_selector))
        group.create_dataset("mask",
                             data=self.feat_selector.get_support(indices=True))
        return data
            
    def fit_classifier(self, group, data, targets):
        raise NotImplementedError
    
    def store_target_names(self, group, target_names):
        group.create_dataset("target_names", data=target_names)
        



class NBModelBuilder(ModelBuilder):
    
    def __init__(self, tab_fname, samp_hdf_fname, models_hdf_fname,
                 classifier, graphs_pkl_fname=None, counts_pkl_fname=None,
                 feat_selector=None):
        ModelBuilder.__init__(self, tab_fname, samp_hdf_fname,
                              models_hdf_fname, classifier, graphs_pkl_fname=graphs_pkl_fname,
                              feat_selector=feat_selector)
        self.counts_pkl_fname = counts_pkl_fname
        
    def prepare(self):
        ModelBuilder.prepare(self)
        self.read_counts()
        
    def read_counts(self):
        if self.counts_pkl_fname:
            log.info("reading counts from " + self.counts_pkl_fname)
            self.counts_dict = cPickle.load(open(self.counts_pkl_fname))
        else:
            self.counts_dict = None
        
    def get_class_priors(self, target_names):
        if self.counts_dict:
            priors = np.zeros(len(target_names))
            
            for i, lempos in enumerate(target_names):
                lemma = lempos.rsplit("/", 1)[0].decode("utf-8")
                # missing counts should never happen
                priors[i] = self.counts_dict.get(lemma, 1) 
        
            # convert to list to prevent an error message from scilearn
            return list(priors / priors.sum())
        
        
    def fit_classifier(self, group, data, targets, target_names):
        log.debug(u"fitting classifer on {} instances with {} features".format(
            data.shape[0], 
            data.shape[1]))
        self.classifier.fit(data, targets, 
                            class_prior=self.get_class_priors(target_names))

        # using compression makes a huge difference in size and speed
        # e.g. with lzf, size is about 5% of orginal size, with gzip about 2.5% 
        group.create_dataset("class_log_prior_",
                             data=self.classifier.class_log_prior_, 
                             compression='lzf')
        group.create_dataset("feature_log_prob_",
                             data=self.classifier.feature_log_prob_, 
                             compression='lzf')





class NearestCentroidModelBuilder(ModelBuilder):
    
    def fit_classifier(self, group, data, targets, target_names):
        self.classifier.fit(data.astype(float), targets)
        group.create_dataset("centroids_",
                             data=self.classifier.centroids_, 
                             compression='lzf')
        
    
    
    
