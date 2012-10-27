"""
processing of evaluation data
"""

import logging
import sys
import xml.etree.ElementTree as et

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
    
    

