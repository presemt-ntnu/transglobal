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

    def __init__(self):
        GraphProces.__init__(self)



class TextFormat(Format):
    """
    write translations in plain text format
    """
    
    def __init__(self):
        Format.__init__(self)
        self.out_str = ""
    
    
    def _single_run(self, graph):
        self.out_str += u"SOURCE {0}: {1}\nTARGET {2}: {3}\n\n".format(
            graph.graph.get("id", ""),
            graph.source_string(),
            graph.graph.get("id", ""),
            graph.graph["target_string"])
        
    def write(self, outf=sys.stdout):
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
    write translation output in Mteval XML format
    """
    
    xml_declaration = ( 
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE mteval SYSTEM '
        '"ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-xml-v1.3.dtd">\n' )
    
    def __init__(self, setid="presemt", srclang="src", trglang="tgt",
                 sysid="transglobal", docid="test", genre="xx"):
        # see ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-xml-v1.6.dtd
        Format.__init__(self)
        self.root_el = et.Element("mteval")
        tstset_el = et.SubElement(self.root_el, "tstset", setid=setid, srclang=srclang,
                                  trglang=trglang)
        self.doc_el = et.SubElement(tstset_el, "doc", docid=docid, genre=genre, sysid=sysid)
        self.seg_count = 0
        
    def _single_run(self, graph):
        self.seg_count += 1
        seg_el = et.SubElement(self.doc_el, "seg", id=str(self.seg_count))
        seg_el.text = graph.graph["target_string"]
        
    def write(self, outf=sys.stdout, pprint=True):
        tree = et.ElementTree(self.root_el)
        if pprint: 
            indent(tree.getroot())

        if isinstance(outf, basestring):
            outf = codecs.open(outf, "w")
            close = True
        else:
            close = False
            
        log.info("writing output in Mteval format to " + outf.name)            
        outf.write(self.xml_declaration)
        tree.write(outf, encoding="utf-8")
        
        if close:
            outf.close()
        
        
        
        
        
        
    
    