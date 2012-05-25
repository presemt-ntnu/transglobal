"""
Evaluate "most frequent translation" (MFT) baseline
on various Presemt data sets and language pairs
"""

import codecs
import logging
import cPickle
import os
import xml.etree.cElementTree as et

from tg.config import config
from tg.annot import get_annotator
from tg.transdict import TransDict, TransDictGreek
from tg.lookup import Lookup
from tg.freqscore import FreqScore
from tg.draw import Draw
from tg.format import TextFormat, MtevalFormat
from tg.eval import mteval, get_scores, mteval_lang
from tg.utils import set_default_log


def most_frequent_translation(lang_pair, data_set, draw=False):
    source_lang, target_lang = lang_pair.split("-")
    exp_name = "mft_{}_{}".format(data_set, lang_pair)
    out_dir = "_" + exp_name
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # get marked-up text from input source
    # TODO: remove this temporary hack for Greek
    if source_lang == "gr":
        xml_tree = et.ElementTree(file=config["eval"][data_set][lang_pair]["src_fname"])
        text = "\n".join(seg.text.strip() for seg in  xml_tree.iter("seg"))        
    else:
        text = codecs.open(config["eval"][data_set][lang_pair]["src_fname"], encoding="utf-8").read()    
    
    # annotate
    annotator = get_annotator(source_lang, xml_sent_tag="seg")
    graph_list = annotator(text)
    
    # lookup translations
    dict_fname = TransDict.load(config["dict"][lang_pair]["pkl_fname"])
    lookup = Lookup(dict_fname)
    lookup(graph_list)
    
    # score most frequent translation
    freq_score = FreqScore(config["count"]["lemma"][target_lang]["pkl_fname"])
    freq_score(graph_list)
    
    # save graphs
    pkl_fname = os.path.join(out_dir, exp_name + ".pkl")
    cPickle.dump(graph_list, open(pkl_fname, "wb"))
    
    # to load graphs, use:
    # graph_list = cPickle.load(open("graphs_de-en.pkl"))
        
    # draw graphs
    if draw:
        draw = Draw()
        draw(graph_list, out_format="pdf", best_score_attr="freq_score", 
             out_dir=out_dir)
    
    # write translation output in plain text format
    format = TextFormat(score_attr="freq_score")
    format(graph_list)
    format.write(os.path.join(out_dir, exp_name + ".txt"))
    
    # write translation output in Mteval format
    srclang, trglang = mteval_lang(lang_pair)
    format = MtevalFormat(score_attr="freq_score", 
                          srclang=srclang, trglang=trglang, 
                          sysid="transglobal:most frequent translation",
                          setid=config["eval"][data_set][lang_pair]["setid"],
                          docid=config["eval"][data_set][lang_pair]["docid"])
    format(graph_list)
    tst_fname = os.path.join(out_dir, exp_name + ".tst")
    format.write(tst_fname)
    
    # calculate BLEU and NIST scores using mteval script
    scores_fname = os.path.join(out_dir, exp_name + ".scores")
    mteval(config["eval"][data_set][lang_pair]["lemma_ref_fname"],
           config["eval"][data_set][lang_pair]["src_fname"],
           tst_fname,
           scores_fname)
    scores = get_scores(scores_fname)
    return scores



def mft_metis(draw=False):
    """
    Method: "most frequent translation" (MFT) baseline
    Data: METIS evaluation data, lemmatized
    Language pairs: EN-DE, DE-EN
    Measures: NIST and BLEU, lemmatized
    """
    lang_pairs = "en-de", "de-en"
    scores = []
    
    for lang_pair in lang_pairs:
        scores.append(most_frequent_translation(lang_pair, data_set="metis", 
                                                draw=draw))
        
    print "Scores on METIS\n"
    print "LANG:\tNIST:\tBLEU"
    
    for lang_pair, score in zip(lang_pairs, scores):
        print "{0}\t{1[0]}\t{1[1]}".format(lang_pair, score)
        
    print


def mft_presemt_dev(draw=False):
    """
    Method: "most frequent translation" (MFT) baseline
    Data: PRESEMT development data, lemmatized
    Language pairs: EN-DE, DE-EN
    Measures: NIST and BLEU, lemmatized
    """
    lang_pairs = "en-de", "de-en", "gr-de", "gr-en"
    scores = []
    
    for lang_pair in lang_pairs:
        scores.append(most_frequent_translation(lang_pair, data_set="presemt-dev", 
                                                draw=draw))
        
    print "Scores on PRESEMT DEV\n"
    print "LANG:\tNIST:\tBLEU"
    
    for lang_pair, score in zip(lang_pairs, scores):
        print "{0}\t{1[0]}\t{1[1]}".format(lang_pair, score)
        
    print


        
# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)

mft_metis()
mft_presemt_dev(draw=False)