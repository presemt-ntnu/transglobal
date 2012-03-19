#!/usr/bin/env python

import logging
import cPickle

from tg.config import config
from tg.annot import TreeTaggerEnglish
from tg.transdict import TransDict, DictAdaptor
from tg.lookup import Lookup
from tg.freqscore import FreqScore
from tg.draw import Draw
from tg.arrange import Arrange
from tg.utils import set_default_log


# for logging to stderr in utf-8 use:
set_default_log(level=logging.DEBUG)

# for logging to stderr in ascii use:
# set_default_log(level=logging.DEBUG, encoding="ascii", errors="replace")


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
en_de_dict = DictAdaptor(config["dict"]["en-de"]["pkl_fname"],
                         config["dict"]["en-de"]["posmap_fname"])
lookup = Lookup(en_de_dict)
lookup(graph_list)

# frequency scoring
freq_score = FreqScore(config["count"]["lemma"]["de"]["pkl_fname"])
freq_score(graph_list)

# draw
draw = Draw()
draw(graph_list, out_format="pdf")

# save
cPickle.dump(graph_list, open("graphs.pkl", "wb"))
    
# arrange 
arrange = Arrange()
arrange(graph_list)

# write
for graph in graph_list:
    print "SOURCE:", graph.source_string()
    print "TARGET:", graph.graph["target_string"]
    print


    
    
    
    



