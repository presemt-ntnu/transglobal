#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create naive bayes models and store parameters in hdf5 
"""

# TODO: experiment with compression and chunk size
# TODO: feature selection
# TDOD: include vocab in hdf5 file?

import codecs
import cPickle
import datetime
import logging
import os
import time

import numpy as np
import scipy.sparse as sp
import h5py

from tg.utils import coo_matrix_from_hdf5

log = logging.getLogger(__name__)


def make_models(tab_fname, samp_hdf_fname, models_hdf_fname, classifier,
                save_classifier_func, max_models=None):
    start_time = time.time() 
    log.info("opening samples file " + samp_hdf_fname)
    sample_hdfile = h5py.File(samp_hdf_fname, "r")
    samples = sample_hdfile["samples"]

    log.info("creating models file " + models_hdf_fname)
    models_hdfile = h5py.File(models_hdf_fname, "w")
    models = models_hdfile.create_group("models")
    # Pickle classifier and include in hdf5 file.
    # This saves the parameters from __init__.
    # This is before a call to fit(), so class_log_prior_ and 
    # feature_log_prob_ are excluded.
    # Loading this pickled object requires its class (e.g. MultinomialNB)
    # to be part of the current namespace.
    # Alternative is to use the _get_params() and set_params() methods
    # from the BaseEstimator class
    log.info("saving classifier {0}".format(classifier))
    models["classifier_pickle"] = cPickle.dumps(classifier) 
    
    prev_source_lempos = None
    models_count = 0
    
    for line in codecs.open(tab_fname, encoding="utf8"):
        if models_count == max_models:
            log.info("reached max number of models")
            break
        
        source_label, target_label = line.rstrip().split("\t")[1:3]
        # strip corpus POS tag
        source_lempos = source_label.rsplit("/", 1)[0]
        target_lempos = target_label.rsplit("/", 1)[0]

        try:
            samp_group = samples[target_lempos]
        except KeyError:
            log.warning("found no sample for " + target_lempos)
            continue
                
        sm = coo_matrix_from_hdf5(samp_group)        
        
        # hdf5 cannot store array of unicode strings, so use byte strings for
        # label names
        target_lempos = target_lempos.encode("utf-8")
        
        if prev_source_lempos == source_lempos:
            data = sp.vstack([data, sm])
            target_count += 1
            target_names.append(target_lempos)
            # concat new targets depending on number of instances
            new_targets = np.zeros((sm.shape[0],)) + target_count
            targets = np.hstack((targets, new_targets))
        else:
            if prev_source_lempos and target_count:
                log.debug("fitting classifer for {} with {} targets on {} instances with {} features".format(
                    prev_source_lempos, len(target_names), data.shape[0], data.shape[1]))
                classifier.fit(data, targets)
                class_group = models.create_group(prev_source_lempos)
                log.info("saving classifier model for " + prev_source_lempos)
                save_classifier_func(class_group, classifier)
                class_group.create_dataset("target_names", data=target_names)
                models_count += 1
                
            data = sm
            targets = np.zeros((sm.shape[0],))
            target_count = 0
            target_names = [target_lempos]
            
        prev_source_lempos = source_lempos
    
    log.info("saved {} models".format(models_count))
    log.info("closing models file " + models_hdf_fname)    
    models_hdfile.close()
    size = os.path.getsize(models_hdf_fname) / float(1024.0 ** 2)
    elapsed_time = time.time() - start_time
    log.info("elapsed time: {0}".format(datetime.timedelta(seconds=elapsed_time)))
    log.info("average time per model: {0}".format(
        datetime.timedelta(seconds=elapsed_time/float(models_count))))
    log.info("models file size: {0:.2f} MB".format(size))
    log.info("average model size: {:.2f} MB".format(size / float(models_count)))
    sample_hdfile.close()


def save_nb_classifier_to_hdf5(group, classifier):
    # using compression makes a huge difference in size and speed
    # e.g. with lzf, size is about 5% of orginal size, with gzip about 2.5% 
    group.create_dataset("class_log_prior_", data=classifier.class_log_prior_, compression='lzf')
    group.create_dataset("feature_log_prob_", data=classifier.feature_log_prob_, compression='lzf')
    

if __name__ == "__main__":
    from tg.utils import set_default_log
    set_default_log(level=logging.INFO)
    
    from sklearn.naive_bayes import MultinomialNB
        
    # en-de
    tab_fname = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/en/de/en-de_ambig.tab"
    samp_hdf_fname = "de_samples.hdf5"
    models_hdf_fname = "en-de_nb-models.hdf5"
    
    # these are the default settings
    classifier = MultinomialNB(alpha=1.0, fit_prior=True)
        
    make_models(tab_fname, samp_hdf_fname, models_hdf_fname, classifier,
                save_nb_classifier_to_hdf5, 
                #max_models=100
                )
    
    
