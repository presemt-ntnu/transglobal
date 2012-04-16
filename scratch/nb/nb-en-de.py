"""
evaluate naive bayes scoring on en-de translation on Presemt evaluation data
"""

import logging
import cPickle
import xml.etree.cElementTree as et

import pydot

from tg.config import config
from tg.annot import TreeTaggerEnglish
from tg.transdict import TransDict
from tg.lookup import Lookup
from tg.freqscore import FreqScore
from tg.draw import Draw, DrawGV
from tg.arrange import Arrange
from tg.format import TextFormat, MtevalFormat
from tg.eval import mteval
from tg.utils import set_default_log



class _DrawGV(DrawGV):
    
    def trans_edge(self, u, v, data):
        scores = []
        color = "gray"
        penwidth = 1        

        if "freq_score" in data:
            scores.append("{0:.2f}".format(data["freq_score"]))
            penwidth = max(10 * data["freq_score"], 1)
            color = "gray"        
        
        if "nb_score" in data:
            scores.append("{0:.2f}".format(data["nb_score"]))
            penwidth = max(10 * data["nb_score"], 1)
            color = "#fdc086"
            
        label = " & ".join(scores)

        return pydot.Edge(str(u), str(v), color=color, label=label,
                          penwidth=penwidth, **self.EDGE_DEFAULTS)
    
    
class _Arrange(Arrange):
    
    def score(self, data):
        try:
            return data["nb_score"]
        except:
            pass
        
        try:
            return data["freq_score"]
        except:
            return -1
        
        

# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)

#import logging
#logging.getLogger("tg.freqscore").setLevel(logging.DEBUG)

### get text from input source
##xml_tree = et.ElementTree(file=config["eval"]["presemt"]["en-de"]["src_fname"])
##text = " ".join(seg.text.strip() for seg in  xml_tree.iter("seg"))

### annotate
##annotator = TreeTaggerEnglish()
##graph_list = annotator(text)

### lookup
##en_de_dict = TransDict.load(config["dict"]["en-de"]["pkl_fname"])
##lookup = Lookup(en_de_dict)
##lookup(graph_list)

### save
##cPickle.dump(graph_list, open("graphs.pkl", "wb"))

graph_list = cPickle.load(open("graphs.pkl"))#[:5]

# frequency scoring
freq_score = FreqScore(config["count"]["lemma"]["de"]["pkl_fname"])
freq_score(graph_list)


from tg.nbscore import NBScore

vocab_fname = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/de/de_vocab.pkl"
vocab = cPickle.load(open(vocab_fname))

import h5py

model_hdf_fname = "en-de_nb-model.hdf5"
model = h5py.File(model_hdf_fname, "r")

nbscorer = NBScore(vocab, model)
nbscorer(graph_list)


# draw
draw = Draw(drawer=_DrawGV)
draw(graph_list, out_format="pdf")
    
# arrange 
arrange = _Arrange()
arrange(graph_list)


format = TextFormat()
format(graph_list)
format.write()

# write translation output in Mteval format
format = MtevalFormat(srclang="English", trglang="German",
                      sysid="transglobal:naive bayes, most frequent lemma")
format(graph_list)
tst_fname = "out_nb_en_de.tst"
format.write(tst_fname)

# calculate BLEU and NIST scores using mteval script
mteval(config["eval"]["presemt"]["en-de"]["lemma_ref_fname"],
       config["eval"]["presemt"]["en-de"]["src_fname"],
       tst_fname,
       "out_nb_en_de.scores")








