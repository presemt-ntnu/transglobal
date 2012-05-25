#!/usr/bin/env python

"""
setup local data
"""

import logging 
from tg.utils import set_default_log
set_default_log(level=logging.INFO)

from tg.config import config
from tg.transdict import TransDict, TransDictGreek
from tg.counts import mk_counts_pkl
from tg.eval import lemmatize
from tg.utils import create_dirs



def create_all():
    create_dict_pkl()
    create_counts_pkl()
    create_lemma_data()

#-----------------------------------------------------------------------------
# create pickled translation dictionaries
#-----------------------------------------------------------------------------

def create_dict_pkl(lang_pairs=("de-en", "en-de", "gr-de", "gr-en")):
    for lang_pair in lang_pairs:
        if lang_pair == "de-en":
            reverse = True
        else:
            reverse = False
            
        if lang_pair.startswith("gr-"):
            trans_dict_class = TransDictGreek
        else:
            trans_dict_class = TransDict
            
        dict_fname = config["dict"][lang_pair]["xml_fname"]
        posmap_fname = config["dict"][lang_pair]["posmap_fname"]
        trans_dict = trans_dict_class.from_xml(dict_fname, 
                                               reverse=reverse, 
                                               pos_map=posmap_fname)
        pkl_fname = config["dict"][lang_pair]["pkl_fname"]
        create_dirs(pkl_fname)
        trans_dict.dump(pkl_fname)


#-----------------------------------------------------------------------------
# create pickled counts
#-----------------------------------------------------------------------------

def create_counts_pkl(languages=(("de", "en", "gr"),)):
    # so far, counts are only used for target languages
    for lang in languages:
        pkl_fname = config["count"]["lemma"][lang]["pkl_fname"]
        create_dirs(pkl_fname)
        mk_counts_pkl(
            config["count"]["lemma"][lang]["counts_fname"], 
            pkl_fname,
            int(config["count"]["lemma"][lang]["min_count"]))


#-----------------------------------------------------------------------------
# create lemmatized evaluation data
#-----------------------------------------------------------------------------


def create_lemma_data():
    create_lemma_data_METIS()
    create_lemma_data_PRESEMT_dev()

def create_lemma_data_METIS():
    # METIS data 
    for lang_pair in "de-en", "en-de":
        target_lang = lang_pair.split("-")[1]
        lemma_ref_fname = config["eval"]["metis"][lang_pair]["lemma_ref_fname"]
        create_dirs(lemma_ref_fname)
        lemmatize(
            config["eval"]["metis"][lang_pair]["word_ref_fname"],
            config["tagger"][target_lang]["command"],
            config["tagger"][target_lang]["encoding"],
            lemma_ref_fname)


def create_lemma_data_PRESEMT_dev():
    # PRESEMT development data
    #TODO: Greek data is automatically produces yet
    for lang_pair in "de-en", "en-de":
        target_lang = lang_pair.split("-")[1]
        lemma_ref_fname = config["eval"]["presemt-dev"][lang_pair]["lemma_ref_fname"]
        create_dirs(lemma_ref_fname)
        lemmatize(
            config["eval"]["presemt-dev"][lang_pair]["word_ref_fname"],
            config["tagger"][target_lang]["command"],
            config["tagger"][target_lang]["encoding"],
            lemma_ref_fname)


if __name__ == "__main__":
    create_all()


