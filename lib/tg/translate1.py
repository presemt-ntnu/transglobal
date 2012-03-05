#!/usr/bin/env python

"""
word-for-word translation 
- with lemmatzation and POS tagging for lookup
- with lookup of MWUs
- translation disambiguation
- incremental disambiguation of context vectors
"""

import logging as log
import cPickle

from annot import TreeTaggerEnglish
from transdict import TransDict, DictAdaptor
from lookup import Lookup
from draw import Draw
import transdict


log.basicConfig(level=log.DEBUG)

text = """It may seem obvious to just create one language for everybody to 
use. Luckily, several linguists felt the same way. They made up what we call \
constructed languages. But, languages are a big part of a people's culture \
and identity and most of them have long interesting histories. People aren't \
willing to give them up. It is also very hard to become fluent in a language. \
It may seem natural to you to speak English, but it is actually very hard for \
many adults to learn."""

# annotate
annotator = TreeTaggerEnglish()
graph_list = annotator(text)

# lookup
en_de_dict = DictAdaptor("dict-en-de.pkl", "en-de_posmap")
lookup = Lookup(en_de_dict)
lookup(graph_list)

# draw
for i, graph in enumerate(graph_list):
    draw = Draw(graph)
    draw.write("g{0}.pdf".format(i), format="pdf")

# save
cPickle.dump(graph_list, open("graphs.pkl", "wb"))
    
    
    
    
    




