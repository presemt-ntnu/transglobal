#!/usr/bin/env python

"""
create pickled dictionaries
"""

from os import getenv, mkdir
from os.path import exists

import logging as log

from tg.transdict import TransDict

log.basicConfig(level = log.INFO)

data_dir = getenv("TG_BASE_DIR") + "/_data" 

local_dir = getenv("TG_BASE_DIR") + "/_local" 

local_dict_dir = local_dir + "dicts/"
if not exists(local_dict_dir): mkdir(local_dict_dir)


# create pickled dictionary
dict_fname = data_dir + "/dicts/lex_DE-EN.xml"
pkl_fname = local_dict_dir + "/dict_en-de.pkl"

dict_en_de = TransDict.from_xml(dict_fname, reverse=True)
dict_en_de.dump(pkl_fname)