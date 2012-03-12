#!/usr/bin/env python

"""
create pickled dictionaries
"""

import logging as log

from tg.config import config
from tg.transdict import TransDict
from tg.counts import mk_counts_pkl
from tg.utils import create_dirs

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

