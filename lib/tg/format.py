"""
format output
"""

import codecs
import logging
import sys
import xml.etree.cElementTree as et

from tg.graphproc import GraphProces
from tg.utils import indent


log = logging.getLogger(__name__)


class Format(GraphProces):
    """
    Abstract base class for formatting translations
    """

    def __init__(self, score_attr="best_score", unknown=None):
        GraphProces.__init__(self)
        self.score_attr = score_attr
        self.unknown = unknown

    def _target_lemma_list(self, graph): 
        target_lemmas = []
        
        # TODO: handle hypernodes
        for u in graph.source_nodes_iter(ordered=True):
            v = graph.max_score(u, self.score_attr)[1]
            if v:
                lemma = graph.lemma(v)
            else:
                # no translation edges
                source_lempos = graph.lempos(u)
                lemma = self.unknown or graph.lemma(u)
                log.warning(u"no translation for lempos {}; using {} "
                            "instead".format(source_lempos, lemma))
            target_lemmas.append(lemma)
            
        return target_lemmas
    
    def _target_lemma_string(self, graph):
        return u" ".join(self._target_lemma_list(graph))
            


class TextFormat(Format):
    """
    Format translations in plain text format
    
    Parameters
    ----------
    score_attr: string, optional
        attribute on translation edges which contains the scores
    unknown: str or None
        Target lemma to use when no translations were found. 
        The default value of None means that the source lemma will be used. 
        
        
    Attributes
    ----------
    out_str: str
        Contains result of formatting.
    """
    
    def __init__(self, score_attr="best_score", unknown=None):
        Format.__init__(self, score_attr=score_attr, unknown=unknown)
        self.out_str = ""
    
    def _single_run(self, graph):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        self.out_str += u"SOURCE {0}: {1}\nTARGET {2}: {3}\n\n".format(
            graph.graph.get("id", ""),
            graph.source_string(),
            graph.graph.get("id", ""),
            self._target_lemma_string(graph))
        
    def write(self, outf=sys.stdout):
        """
        Write formatted translation output to stream
        
        Parameters
        ----------
        outf: str of file instance, optional
        """
        if isinstance(outf, basestring):
            outf = codecs.open(outf, "w", encoding="utf-8")
            close = True
        else:
            close = False
            
        log.info("writing output in plain text format to " + outf.name)
        outf.write(self.out_str)
        
        if close:
            outf.close()
    

class MtevalFormat(Format):
    """
    Format translation output in Mteval XML format
    
    Parameters
    ----------
    src_fname: str
        Name of file containing translation source text in Mteval XML format 
    trglang: str
        Value of trglang attribute on root element
    sysid: str
        Value of sysid attribute on doc elements
    score_attr: string, optional
        attribute on translation edges which contains the scores
    unknown: None or str, optional
        Target lemma to use when no translations were found. 
        The default value of None means that the source lemma will be used. 
        
    Attributes
    ----------
    elem_tree: ElementTree instance
        Contains result of formatting as an XML tree.
    """
    
    def __init__(self, src_fname, trglang, sysid,
                 score_attr="best_score", unknown=None):
        Format.__init__(self, score_attr=score_attr, unknown=unknown)
        self.src_fname = src_fname
        self.trglang = trglang
        self.sysid = sysid
        self.elem_tree = None
        
    def _batch_run(self, graph_list):
        # _single_run() does not make sense in this format
        graph_list = iter(graph_list)
            
        for event, elem in et.iterparse(self.src_fname):  
            if elem.tag == "srcset":
                elem.tag = "tstset"
                elem.set("trglang", self.trglang)
            elif elem.tag == "doc":
                elem.set("sysid", self.sysid)
            elif elem.tag == "seg":
                graph = graph_list.next()
                log.info("applying {0} to graph {1}".format(
                    self.__class__.__name__,
                    graph.graph["id"]))                
                elem.text = self._target_lemma_string(graph)

                
        self.elem_tree = et.ElementTree(elem) 

    def write(self, outf=sys.stdout, pprint=True):
        """
        Write formatted translation output to stream
        
        Parameters
        ----------
        outf: str of file instance, optional
            Output file
        pprint: bool, optional
            Pretty print XML output using indentation
        """
        if pprint: 
            indent(self.elem_tree.getroot())

        if isinstance(outf, basestring):
            outf = codecs.open(outf, "w")
            close = True
        else:
            close = False
            
        log.info("writing output in Mteval format to " + outf.name)            
        self.elem_tree.write(outf, encoding="utf-8")
        
        if close:
            outf.close()                 

