"""
evaluate de-en translation on Presemt evaluation data
"""

import logging
import cPickle
import xml.etree.cElementTree as et

from tg.config import config
from tg.annot import TreeTaggerGerman
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

#log = logging.getLogger("tg.lookup")
#log.setLevel(logging.DEBUG)



# get text from input source
xml_tree = et.ElementTree(file=config["eval"]["presemt"]["de-en"]["src_fname"])
text = " ".join(seg.text.strip() for seg in  xml_tree.iter("seg"))

# annotate
annotator = TreeTaggerGerman()
graph_list = annotator(text)

# lookup
de_en_dict = TransDict.load(config["dict"]["de-en"]["pkl_fname"])
lookup = Lookup(de_en_dict)
lookup(graph_list)

# frequency scoring
freq_score = FreqScore(config["count"]["lemma"]["en"]["pkl_fname"])
freq_score(graph_list)

# draw
draw = Draw()
draw(graph_list, out_format="pdf")
    
# arrange 
arrange = Arrange()
arrange(graph_list)

# save
cPickle.dump(graph_list, open("graphs.pkl", "wb"))

# graph_list = cPickle.load(open("graphs.pkl"))

format = TextFormat()
format(graph_list)
format.write()

# write translation output in Mteval format
format = MtevalFormat(srclang="German", trglang="English",
                      sysid="transglobal:most frequent lemma")
format(graph_list)
tst_fname = "out_de_en.tst"
format.write(tst_fname)

# calculate BLEU and NIST scores using mteval script
mteval(config["eval"]["presemt"]["de-en"]["lemma_ref_fname"],
       config["eval"]["presemt"]["de-en"]["src_fname"],
       tst_fname,
       "out_de_en.scores")








