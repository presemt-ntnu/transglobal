"""
processing of evaluation data
"""

import codecs
import logging
import subprocess
import sys
import xml.etree.ElementTree as et

from tg.config import config
from tg.annot import get_annotator


log = logging.getLogger(__name__)

#log.setLevel(logging.DEBUG)



def lemmatize(infname, lang, outf=sys.stdout, sent_tag="seg",
              encoding="utf-8", replace_unknown_lemma=True):
    """
    Lemmatize reference translations in mteval format (may work for other
    formats too)
    
    Parameters
    ----------
    infname: str
        name of inputfile in mteval xml format
    lang: str
        two-letter language identifier
    outf: file or str, optional
        file or filename for output
    encoding: str, optional
        char encoding for output (should be the same as that of input)
    replace_unknown_lemma: bool, optional
        replace unknown lemma by word
    """
    annotator = get_annotator(lang, replace_unknown_lemma=replace_unknown_lemma)
    log.info("using annotator " + annotator.__class__.__name__)
    log.info("reading evaluation data from file " + infname)
    etree = et.ElementTree(file=infname)
    sentences = [ sent_elem.text for sent_elem in etree.iter(sent_tag) ]
    log.debug(u"input:\n" + u"\n".join(sentences))
    graph_list = annotator.annot_sentences(sentences)
        
    for sent_elem, graph in zip(etree.iter(sent_tag), graph_list):
        lemma_text = " ".join(graph.source_lemmas())
        sent_elem.text = lemma_text
        
    log.info("writing lemmatized evaluation data to {0}".format(
        getattr(outf, "name", outf)))
        
    etree.write(outf, encoding=encoding)
    
    
    
def mteval(ref_fname, src_fname, tst_fname, outf=sys.stdout,
           options=config["eval"]["mteval_opts"]):
    """
    simple wrapper of NIST mteval-v13a.pl script
    """
    command = '{0} {1} -r "{2}" -s "{3}" -t "{4}" {5}'.format(
        config["eval"]["perl_fname"],
        config["eval"]["mteval_fname"],
        ref_fname,
        src_fname,
        tst_fname,
        options or "")
        
    # create pipe to eval script
    log.debug("Calling evaluation script as as: " + command)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
      
    # send text and retrieve tagger output 
    out, err = proc.communicate()    
    
    log.debug(u"mteval standard output:\n" + out.decode("utf-8"))
    
    if err:
        log.error(u"mteval standard error:\n" + err.decode("utf-8"))
    
    if isinstance(outf, basestring):
        outf = open(outf, "w")
        close = True
    else:
        close = False
        
    log.info("writing mteval output to " + outf.name)
    outf.write(out)
    
    if close:
        outf.close()
    
    return out, err


def get_scores(score_fname):
    """
    get overall NIST and BLEU scores from scores file
    """
    # TODO: make parsing of scores more robust
    tokens = codecs.open(score_fname, encoding="utf-8").readlines()[-19].split()
    scores = float(tokens[3]), float(tokens[7])
    log.info("scores: NIST = {0}; BLEU = {1}".format(*scores))
    return scores


