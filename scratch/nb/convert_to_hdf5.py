#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
convert collection of samples in gzipped matrix market format to single hdf5 file,
including vocabulary
"""

# TODO: optimal compression and chunks

import cPickle
import glob
import codecs
import logging
import os

import numpy as np
import scipy.io
import h5py

from tg.utils import coo_matrix_to_hdf5

log = logging.getLogger(__name__)


def convert(tab_fname, samp_dir, vocab_fname, hdf_fname,
            samp_fpat="de-sample-{0}.mtx.gz", max_samples=None, source_lempos_subset=None):
    log.info("opening " + hdf_fname)
    hdfile = h5py.File(hdf_fname, "w")
    
    log.info("reading pickled vocabulary from file " + vocab_fname)
    vocab = cPickle.load(open(vocab_fname))   
    
    log.info("storing vocabulary ({0} terms)".format(len(vocab)))    
    # reverse lemma-to-index mapping
    reversed_vocab = dict((i, lemma) for lemma, i in vocab.iteritems())    
    assert len(reversed_vocab) == len(vocab)
    # convert reversed mapping to a sorted list
    # hdf5 can't handle unicode strings, so encode terms as utf-8 byte strings    
    vocab = [reversed_vocab[i].encode("utf-8") 
             for i in xrange(len(reversed_vocab))]
    # create new type for variable-length strings
    # see http://code.google.com/p/h5py/wiki/HowTo#Variable-length_strings
    str_type = h5py.new_vlen(str)
    hdfile.create_dataset("vocab", 
                          data=vocab,
                          dtype=str_type)
    del reversed_vocab

    samples = hdfile.create_group("samples")
    samp_count = 0
         
    for line in codecs.open(tab_fname, encoding="utf8"):
        if max_samples and samp_count > max_samples:
            break
        
        source_label, target_label, samp_fid, new = line.rstrip().split("\t")[1:]
        
        # Remove deWac POS tag.
        # Samples are stored under target lemma plus *lexicon* POS tag combination!
        source_lempos = source_label.rsplit("/", 1)[0]        
        target_lempos = target_label.rsplit("/", 1)[0]  
        
        if source_lempos_subset and source_lempos not in source_lempos_subset:
            log.info(u"skipping samples for {} -> {}".format(source_lempos, target_lempos))
            continue
        
        log.info(u"adding samples for {} -> {}".format(source_lempos, target_lempos))    
        
        # only if this is the first occurrence of this lempos
        if target_lempos not in samples:
            samp_fname = samp_dir + samp_fpat.format(samp_fid)
            
            try:
                m = scipy.io.mmread(samp_fname.encode("utf-8"))
            except IOError:
                log.error("oops, found no sample file! " + samp_fname)
                continue
            
            # lempos contains "/" as separator, which h5py interpretes as a
            # subgroup
            group = samples.create_group(target_lempos)
            log.info("from " + samp_fname)
            # using 8-bit int to save space, assuming that no lemma will
            # occur over 256 times in a single context
            coo_matrix_to_hdf5(m, group, data_dtype="=i1", compression='gzip')
            samp_count += 1
            
    log.info("closing " + hdf_fname)
    hdfile.close()          
    
    


def extract_source_lempos_subset(graphs_pkl_fname):
    """
    extract all required source lempos from pickled graphs,
    where POS tag is the *lexicon* POS tag
    """
    lempos_subset = set()
    
    for graph in cPickle.load(open(graphs_pkl_fname)):
        for _,d in graph.source_nodes_iter(data=True, ordered=True):
            try:
                lempos_subset.add(" ".join(d["lex_lempos"]))
            except KeyError:
                # not found in lexicon
                pass
        
    return lempos_subset

    
    
if __name__ == "__main__":
    from tg.utils import set_default_log
    set_default_log(level=logging.INFO)
        
    ## en-de
    #tab_fname = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/en/de/en-de_ambig.tab"
    #samp_dir = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/de/samples/"
    #vocab_fname = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/de/de_vocab.pkl"
    ##hdf_fname = "de_samples.hdf5"
    #hdf_fname = "de_samples_subset.hdf5"
    #graphs_pkl_fname = "en-de_graphs.pkl"
    #convert(tab_fname, samp_dir, vocab_fname, hdf_fname, 
            ## max_samples=10,
            #source_lempos_subset = extract_source_lempos_subset(graphs_pkl_fname))
        
    # de-en
    tab_fname = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/de/en/de-en_ambig.tab"
    samp_dir = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/en/samples/"
    vocab_fname = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/en/en_vocab.pkl"
    #hdf_fname = "en_samples.hdf5"
    hdf_fname = "en_samples_subset.hdf5"
    graphs_pkl_fname = "de-en_graphs.pkl"
    convert(tab_fname, samp_dir, vocab_fname, hdf_fname, 
            samp_fpat="en-sample-{0}.mtx.gz",
            # max_samples=10,
            source_lempos_subset = extract_source_lempos_subset(graphs_pkl_fname))

    