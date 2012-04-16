#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create naive bayes models and store parameters in hdf5 
"""

# TODO: experiment with compression and chunk size
# TODO: store model params 
# TODO: feature selection
# TDOD: include vocab in hdf5 file?

import codecs
import logging

import numpy as np
import scipy.sparse as sp
import h5py

import sklearn.naive_bayes as nb


log = logging.getLogger(__name__)

def make_models(tab_fname, samp_hdf_fname, model_hdf_fname):
    sample_hdfile = h5py.File(samp_hdf_fname, "r")
    samples = sample_hdfile["samples"]

    model_hdfile = h5py.File(model_hdf_fname, "w")
    models = model_hdfile.create_group("models")
    
    prev_source_lempos = None
    classifier = nb.BernoulliNB()
    
    for line in codecs.open(tab_fname, encoding="utf8"):
        source_label, target_label = line.rstrip().split("\t")[1:3]
        # strip corpus POS tag
        source_lempos = source_label.rsplit("/", 1)[0]
        target_lempos = target_label.rsplit("/", 1)[0]
        target_lempos = target_lempos.replace("/", "_")

        try:
            samp = samples[target_lempos]
        except KeyError:
            log.warning("found no sample for " + target_lempos)
            continue
        
        # create sparse matrix from COO data
        sm = sp.coo_matrix((samp[2], samp[:2]), shape=samp.attrs["shape"])
        
        if prev_source_lempos == source_lempos:
            data = sp.vstack([data, sm])
            target_count += 1
            target_names.append(target_lempos.encode("utf-8"))
            new_targets = np.zeros((sm.shape[0],)) + target_count
            targets = np.hstack((targets, new_targets))
        else:
            if prev_source_lempos and target_count:
                # FIX: using prev_source_lemma below is ugly
                # build model
                print prev_source_lempos, target_names
                print data.shape, targets.shape
                classifier.fit(data, targets)
                #y_pred = classifier.predict(data)

                #print "percent of mislabeled points : %.2f" % (((targets != y_pred).sum() / float(y_pred.shape[0])) * 100)
                #print "percent of correctly labeled points : %.2f" % (((targets == y_pred).sum() / float(y_pred.shape[0])) * 100)
                model = models.create_group(prev_source_lempos)
                model.create_dataset("class_log_prior", data=classifier.class_log_prior_)
                model.create_dataset("feature_log_prob", data=classifier.feature_log_prob_, compression='lzf')
                print model.create_dataset("target_names", data=target_names)
                
            data = sm
            targets = np.zeros((sm.shape[0],))
            target_count = 0
            # hdf5 cannot store array of unicode strings, so use byte strings
            target_names = [target_lempos.encode("utf-8")]
            
        prev_source_lempos = source_lempos
        
    sample_hdfile.close()
    model_hdfile.close()

            
            
        
        
    

if __name__ == "__main__":
    from tg.utils import set_default_log
    set_default_log(level=logging.INFO)
        
    # en-de
    tab_fname = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/en/de/en-de_ambig.tab"
    samp_hdf_fname = "de_samples.hdf5"
    model_hdf_fname = "en-de_nb-model.hdf5"
    
    make_models(tab_fname, samp_hdf_fname, model_hdf_fname)