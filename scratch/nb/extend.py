#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This extends the sample vectors in the following way:

1. For each target word in the context vector, we lookup all the source words
which have this target word among their possible tranlations (using a
reversed dictionary).

2. We then lookup all possible target translations of these source words
and add those to the vector.

This makes the training vectors more similar to the test vectors, where
translations of the context are not disambiguated. However, the above method
is worse, because the reverse translation from target to source introduces
even more ambiguity.
"""

import logging
import cPickle

from scipy import sparse as sp

import h5py

from tg.transdict import TransDict
from tg.utils import coo_matrix_from_hdf5, coo_matrix_to_hdf5


log = logging.getLogger(__name__)


def extend_samples(samp_hdf_fname, tdict_pkl_fname, reverse_tdict_pkl_fname,
                   ext_hdf_fname, max_samp=None):
    log.info("opening original samples file " + samp_hdf_fname)
    samp_hdfile = h5py.File(samp_hdf_fname, "r") 
    
    ext_mat = make_extension_matrix(samp_hdfile, tdict_pkl_fname, reverse_tdict_pkl_fname)
    
    log.info("creating extended samples file " + ext_hdf_fname)
    ext_hdfile = h5py.File(ext_hdf_fname, "w") 
    ext_samples = ext_hdfile.create_group("samples") 
    
    log.info("copying vocabulary ({0} terms)".format(len(samp_hdfile["vocab"])))
    ext_hdfile.create_dataset("vocab", data=samp_hdfile["vocab"])
    i = 0
    
    for lemma, lemma_group in samp_hdfile["samples"].iteritems():
        for pos, pos_group in lemma_group.iteritems():
            log.info(u"{0}: creating extended samples for {1}/{2}".format(i, lemma,pos))
            samp_mat = coo_matrix_from_hdf5(pos_group).tocsr()
            mat = (samp_mat * ext_mat).tocoo()
            group = ext_hdfile.create_group(u"samples/{0}/{1}".format(lemma,pos))
            coo_matrix_to_hdf5(mat, group, data_dtype="i1", compression="gzip")
            
            i += 1
            if i == max_samp:
                log.info("reached maximum number of samples")
                break
        if i == max_samp:
            break
    
    log.info("closing " + samp_hdf_fname)
    samp_hdfile.close()          

    log.info("closing " + ext_hdf_fname)
    ext_hdfile.close()          
    
    
    


def make_extension_matrix(samp_hdfile, tdict_pkl_fname, reverse_tdict_pkl_fname):
    reverse_vocab = [lemma.decode("utf-8") for lemma in samp_hdfile["vocab"]]
    vocab = dict((lemma, i) 
                 for i, lemma in enumerate(reverse_vocab))
    assert len(reverse_vocab) == len(vocab)
    
    tdict = TransDict.load(tdict_pkl_fname)
    # disable POS mapping
    tdict.pos_map = None
    
    reverse_tdict = TransDict.load(reverse_tdict_pkl_fname)
    reverse_tdict.pos_map = None
    
    shape = len(vocab), len(vocab)
    log.info("making extension matrix as sparse lil_matrix {0}".format(shape))
    em = sp.lil_matrix(shape, dtype="int8")
    
    for i, target_lemma in enumerate(reverse_vocab):
        try:
            reverse_lookup = reverse_tdict.lookup_lemma(target_lemma)
        except KeyError:
            # vocab term not in reverse dict
            # FIXME: these terms should be removed from vocab
            continue
        
        log.debug(40 * "=")
        
        for _, source_lempos_list in reverse_lookup: 
            for source_lempos in source_lempos_list:
                target_lempos_list = tdict.lookup_lempos(source_lempos)[1]
                for target_lempos in target_lempos_list:
                    # does not handle MWU, but vocab contains only atomic
                    # lemmas so far
                    ext_target_lemma = target_lempos.rsplit("/",1)[0]
                    try:
                        j = vocab[ext_target_lemma]
                    except:
                        # oov
                        continue
                    
                    log.debug(u"{0} --> {1} --> {2}".format(
                        target_lemma,
                        source_lempos, 
                        ext_target_lemma))
                    # counting occurrences does not make a a lot of sense,
                    # so assume boolean
                    em[i,j] = 1
                    
        if log.isEnabledFor(logging.DEBUG):
            log.debug(u"{0} ==> {1}".format(
                target_lemma,
                ", ".join([str((reverse_vocab[j], count)) for j, count in zip(em.rows[i], em.data[i])])))
                   
    log.info("converting to csr_matrix") 
    return em.tocsr()



if __name__ == "__main__":
    from tg.utils import set_default_log
    set_default_log(level=logging.DEBUG)
    
    from tg.config import config
    
    extend_samples(#samp_hdf_fname = "en_samples_filtered.hdf5", 
                   samp_hdf_fname = "en_samples_subset_filtered.hdf5", 
                   tdict_pkl_fname = config["dict"]["de-en"]["pkl_fname"],
                   reverse_tdict_pkl_fname = config["dict"]["en-de"]["pkl_fname"],
                   #ext_hdf_fname = "en_samples_filtered_extended.hdf5",
                   ##ext_hdf_fname = "en_samples_subset_filtered_extended.hdf5",
                   ext_hdf_fname = "ff.hdf5",
                   max_samp = 1,
                   )
    
    #extend_samples(#samp_hdf_fname = "de_samples_filtered.hdf5", 
                   #samp_hdf_fname = "de_samples_subset_filtered.hdf5", 
                   #tdict_pkl_fname = config["dict"]["en-de"]["pkl_fname"],
                   #reverse_tdict_pkl_fname = config["dict"]["de-en"]["pkl_fname"],
                   ##ext_hdf_fname = "de_samples_filtered_extended.hdf5",
                   #ext_hdf_fname = "de_samples_subset_filtered_extended.hdf5",
                   ##max_samp = 1,
                   #)
    
  
    
    


    
    