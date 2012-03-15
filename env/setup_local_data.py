#!/usr/bin/env python

"""
setup local data
"""

import logging 
from tg.utils import set_default_log
set_default_log(level=logging.INFO)

from tg.config import config
from tg.transdict import TransDict
from tg.counts import mk_counts_pkl
from tg.eval import lemmatize
from tg.utils import create_dirs


#-----------------------------------------------------------------------------
# create pickled translation dictionaries
#-----------------------------------------------------------------------------

dict_fname = config["dict"]["de-en"]["xml_fname"]
dict_en_de = TransDict.from_xml(dict_fname)
pkl_fname = config["dict"]["de-en"]["pkl_fname"]
create_dirs(pkl_fname)
dict_en_de.dump(pkl_fname)

dict_fname = config["dict"]["de-en"]["xml_fname"]
# use same dict but reversed
dict_en_de = TransDict.from_xml(dict_fname, reverse=True)
pkl_fname = config["dict"]["en-de"]["pkl_fname"]
create_dirs(pkl_fname)
dict_en_de.dump(pkl_fname)


#-----------------------------------------------------------------------------
# create pickled counts
#-----------------------------------------------------------------------------

for lang in ("de", "en"):
    pkl_fname = config["count"]["lemma"][lang]["pkl_fname"]
    create_dirs(pkl_fname)
    mk_counts_pkl(
        config["count"]["lemma"][lang]["counts_fname"], 
        pkl_fname,
        int(config["count"]["lemma"][lang]["min_count"]))


#-----------------------------------------------------------------------------
# create lemmatized evaluation data
#-----------------------------------------------------------------------------

lemma_ref_fname = config["eval"]["presemt"]["de-en"]["lemma_ref_fname"]
create_dirs(lemma_ref_fname)
lemmatize(
    config["eval"]["presemt"]["de-en"]["word_ref_fname"],
    config["tagger"]["en"]["command"],
    config["tagger"]["en"]["encoding"],
    lemma_ref_fname)

lemma_ref_fname = config["eval"]["presemt"]["en-de"]["lemma_ref_fname"]
create_dirs(lemma_ref_fname)
lemmatize(
    config["eval"]["presemt"]["en-de"]["word_ref_fname"],
    config["tagger"]["de"]["command"],
    config["tagger"]["de"]["encoding"],
    lemma_ref_fname)

    



