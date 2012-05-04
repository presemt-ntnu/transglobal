"""
evaluate en-de translation on Presemt evaluation data
"""

import logging
import cPickle
import xml.etree.cElementTree as et

from tg.config import config
from tg.annot import TreeTaggerEnglish
from tg.transdict import TransDict
from tg.lookup import Lookup
from tg.freqscore import FreqScore
from tg.draw import Draw
from tg.arrange import Arrange
from tg.format import TextFormat, MtevalFormat
from tg.eval import mteval
from tg.utils import set_default_log


# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)

# get text from input source
xml_tree = et.ElementTree(file=config["eval"]["presemt"]["en-de"]["src_fname"])
text = " ".join(seg.text.strip() for seg in  xml_tree.iter("seg"))

# annotate
annotator = TreeTaggerEnglish()
graph_list = annotator(text)

# lookup
en_de_dict = TransDict.load(config["dict"]["en-de"]["pkl_fname"])
lookup = Lookup(en_de_dict)
lookup(graph_list)

# frequency scoring
freq_score = FreqScore(config["count"]["lemma"]["de"]["pkl_fname"])
freq_score(graph_list)
    
# arrange 
arrange = Arrange(score_attrs=["freq_score"])
arrange(graph_list)

# draw
draw = Draw()
draw(graph_list, out_format="pdf", score_attrs=["freq_score"])

# save
cPickle.dump(graph_list, open("graphs_en-de.pkl", "wb"))

# graph_list = cPickle.load(open("graphs_en-de.pkl"))

format = TextFormat()
format(graph_list)
format.write()

# write translation output in Mteval format
format = MtevalFormat(srclang="English", trglang="German",
                      sysid="transglobal:most frequent lemma")
format(graph_list)
tst_fname = "out_en_de.tst"
format.write(tst_fname)

# calculate BLEU and NIST scores using mteval script
mteval(config["eval"]["presemt"]["en-de"]["lemma_ref_fname"],
       config["eval"]["presemt"]["en-de"]["src_fname"],
       tst_fname,
       "out_en_de.scores")








