#!/usr/bin/env python

"""
create pickled dictionaries
"""

import logging as log

from tg.config import config
from tg.transdict import TransDict
from tg.counts import mk_counts_pkl
from tg.utils import create_dirs
from tg.eval import lemmatize

log.basicConfig(level = log.INFO)

# create pickled translation dictionaries
dict_fname = config["de-en_dict_fname"]
dict_en_de = TransDict.from_xml(dict_fname, reverse=True)
pkl_fname = config["en-de_dict_pkl"]
create_dirs(pkl_fname)
dict_en_de.dump(pkl_fname)

# create pickled counts
pkl_fname = config["de_lemma_counts_pkl"]
create_dirs(pkl_fname)
mk_counts_pkl(config["de_lemma_counts_fname"], pkl_fname,
              int(config["min_count"]))

# create lemmatized evaluation data
lemma_ref_fname = config["de_lemma_ref_fname"]
create_dirs(lemma_ref_fname)
lemmatize(
    config["de_word_ref_fname"],
    "tree-tagger-german",
    "latin1",
    lemma_ref_fname)
    
    



