"""
operations on samples
"""

# this is temporary code that should be intergrated with the sampling and
# model building code from Presemt


import logging
import cPickle
import codecs

log = logging.getLogger(__name__)

import h5py

from tg.config import config
from tg.transdict import TransDict
from tg.utils import coo_matrix_from_hdf5, coo_matrix_to_hdf5




def filter_sample_vocab(lang_pair):
    """
    Filter vocabulary words which do not occur in the translation lexicon.
    This reduces the size of the vocabulary and adjusts the context samples
    accordingly.
    
    Assumes that vocab dos NOT contain:
    - POS tags (i.e. lempos combination)
    - multi-word units (MWUs)
    """
    sample_hdf_fname = config["sample"][lang_pair]["samples_fname"] 
    log.info("opening original samples file " + sample_hdf_fname)
    sample_hdfile = h5py.File(sample_hdf_fname, "r")      
    
    filtered_hdf_fname = config["sample"][lang_pair]["samples_filt_fname"] 
    log.info("creating filtered samples file " + filtered_hdf_fname)
    filtered_hdfile = h5py.File(filtered_hdf_fname, "w")    
    
    tdict_pkl_fname =  config["dict"][lang_pair]["pkl_fname"]  
    columns_selector, filtered_vocab = make_new_vocab(sample_hdfile, tdict_pkl_fname)
    
    log.info("storing filtered vocabulary")
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
    
    vocab = [t.decode("utf-8") for t in sample_hdfile["vocab"][()]]
    org_size = len(vocab)
    log.info("orginal vocab size: {} lemmas".format(org_size))
    
    # select columns numbers and corresponding target lemmas
    # sorting is required because order of column number is relevant    
    selection = [ (i, lemma) 
                  for i, lemma in enumerate(vocab)
                  if lemma in dict_target_lemmas ]
    
    columns_selector, filtered_vocab = zip(*selection)
    
    new_size = len(filtered_vocab)
    log.info("filtered vocab size: {} lemmas".format(new_size))
    reduction = (new_size / float(org_size)) * 100
    log.info("vocab reduced to {:.2f}% of orginal size".format(reduction)) 
    
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
            # select only non-empty rows, removing empty rows
            sample_mat = sample_mat.tocsr()
            sample_mat = sample_mat[rows_selector]
            sample_mat = sample_mat.tocoo()
            filtered_group = filtered_samples.create_group(lempos)
            coo_matrix_to_hdf5(sample_mat, filtered_group, data_dtype="=i1",
                               compression='gzip')
