#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filter vocabulary words which do not occur in the translation lexicon. This
reduces the size of the vocabulary and adjusts the context samples
accordingly.

Assumes that vocab dos NOT contain:
- POS tags (i.e. lempos combination)
- multi-word units (MWUs)
"""

import logging
import cPickle

log = logging.getLogger(__name__)

import numpy as np

import h5py

from tg.transdict import TransDict
from tg.utils import coo_matrix_from_hdf5, coo_matrix_to_hdf5


def filter_vocab(sample_hdf_fname, tdict_pkl_fname, filtered_hdf_fname):
    log.info("opening original samples file " + sample_hdf_fname)
    sample_hdfile = h5py.File(sample_hdf_fname, "r")    
        
    columns_selector, filtered_vocab = make_new_vocab(sample_hdfile, tdict_pkl_fname)
    
    log.info("creating filtered samples file " + filtered_hdf_fname)
    filtered_hdfile = h5py.File(filtered_hdf_fname, "w")    
    
    log.info("storing filtered vocabulary ({0} terms)".format(len(filtered_vocab)))
    # create new type for variable-length strings
    # see http://code.google.com/p/h5py/wiki/HowTo#Variable-length_strings
    str_type = h5py.new_vlen(str)
    # hdf5 can't handle unicode strings, so encode terms as utf-8 byte strings
    filtered_hdfile.create_dataset("vocab", 
                                   data=[t.encode("utf-8") for t in filtered_vocab],
                                   dtype=str_type)
    
    make_new_samples(sample_hdfile, filtered_hdfile, columns_selector)
            
    log.info("closing " + sample_hdf_fname)
    sample_hdfile.close()          

    log.info("closing " + filtered_hdf_fname)
    filtered_hdfile.close()                  
    
    

def make_new_vocab(sample_hdfile, tdict_pkl_fname):
    tdict = TransDict.load(tdict_pkl_fname)
    # disable POS mapping
    tdict.pos_map = None
    
    log.info("extracting target lemmas from translation dictionary")
    dict_target_lemmas = set()
    
    for target_lempos_list in tdict._lempos_dict.itervalues():
        for target_lempos in target_lempos_list:
            # skip MWU
            if not " " in target_lempos:
                target_lemma = target_lempos.rsplit("/",1)[0]
                dict_target_lemmas.add(target_lemma)
        
    del tdict
    
    vocab = [t.decode("utf-8") for t in sample_hdfile["vocab"]]
    
    # select columns numbers and corresponding target lemmas
    # sorting is required because order of column number is relevant    
    selection = [ (i, lemma) 
                  for i, lemma in enumerate(vocab)
                  if lemma in dict_target_lemmas ]
    
    columns_selector, filtered_vocab = zip(*selection)
    return columns_selector, filtered_vocab
    
    
        
def make_new_samples(sample_hdfile, filtered_hdfile, columns_selector):
    org_samples = sample_hdfile["samples"]
    filtered_samples = filtered_hdfile.create_group("samples")
    
    for lemma, lemma_group in org_samples.iteritems():
        for pos, pos_group in lemma_group.iteritems():
            lempos = lemma + u"/" + pos
            log.info("adding filtered samples for " + lempos)
            sample_mat = coo_matrix_from_hdf5(pos_group)
            sample_mat = sample_mat.tocsc()
            # select only columns corresponding to filtered vocabulary,
            # removing other columns
            sample_mat = sample_mat[:,columns_selector]
            # get indices of non-empty rows
            sample_mat = sample_mat.tolil()
            rows_selector = sample_mat.rows.nonzero()[0]
            ## FIXME: this is way slow...
            ## rows_selector = [i for i,row in enumerate(sample_mat) if row.nnz ]
            # select only non-empty rows, removing empty rows
            sample_mat = sample_mat.tocsr()
            sample_mat = sample_mat[rows_selector]
            sample_mat = sample_mat.tocoo()
            filtered_group = filtered_samples.create_group(lempos)
            coo_matrix_to_hdf5(sample_mat, filtered_group, data_dtype="=i1", compression='gzip')


    
if __name__ == "__main__":
    from tg.utils import set_default_log
    set_default_log(level=logging.INFO)
    
    from tg.config import config
    
    # de-en
    filter_vocab(
        #sample_hdf_fname = "en_samples.hdf5",
        sample_hdf_fname = "en_samples_subset.hdf5",
        tdict_pkl_fname = config["dict"]["de-en"]["pkl_fname"], 
        #filtered_hdf_fname = "en_samples_filtered.hdf5")
        filtered_hdf_fname = "en_samples_subset_filtered.hdf5")
    
    # en-de
    filter_vocab(
        #sample_hdf_fname = "de_samples.hdf5",
        sample_hdf_fname = "de_samples_subset.hdf5",
        tdict_pkl_fname = config["dict"]["en-de"]["pkl_fname"] , 
        #filtered_hdf_fname = "de_samples_filtered.hdf5")
        filtered_hdf_fname = "de_samples_subset_filtered.hdf5")
    
    
    
    
