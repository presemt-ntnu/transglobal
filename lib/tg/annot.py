"""
constructing linguistically annotated translation graphs
"""

# TODO:
# - more documentation

import codecs
import cStringIO
import xml.etree.ElementTree as et
import logging
import cStringIO
import subprocess
import urllib

import suds

from tg.config import config
from tg.transgraph import TransGraph
from tg.exception import TGException



log = logging.getLogger(__name__)



class Annotator(object):
    
    # string value to use for unknown values
    unknown = "__UNKNOWN__"    
    
    def annot_text(self, text, encoding=None, errors='strict'):
        pass
    
    def annot_text_file(self, inf, encoding="utf-8", errors='strict'):
        if not hasattr(inf, "read"):
            inf = codecs.open(inf, encoding=encoding, errors=errors)
        text = inf.read()    
        return self.annot_text(text)
    
    def annot_sentences(self, sentences, encoding=None):
        pass
    
    def annot_xml(self, source, xml_sent_tag="seg"):
        # Source MUST be a byte string (i.e. encoded).
        # If encoding is different from utf-8, 
        # then it must be specifid by the XMl header
        inf = cStringIO.StringIO(source)
        return self.annot_xml_file(inf, xml_sent_tag)
        
    def annot_xml_file(self, inf, xml_sent_tag="seg"):
        sentences = self._extract_sentences_from_xml(inf, xml_sent_tag)
        return self.annot_sentences(sentences)
        
    def _extract_sentences_from_xml(self, inf, xml_sent_tag):
        # this assumes that a sentence contains no XML markup 
        # (e.g. <b>...</b>)
        return ( elem.text
                 for _, elem in et.iterparse(inf)
                 if elem.tag == xml_sent_tag )
        
        
            
        
class TreeTagger(Annotator):
    """
    annotation of input text with TreeTagger 
    """
    
    unknown_lemma = "<unknown>"
    
    def __init__(self, command, tagger_encoding, eos_pos_tag=None,
                 replace_unknown_lemma=True):
        Annotator.__init__(self)
        self.command = command
        self.tagger_encoding = tagger_encoding
        self.eos_pos_tag = eos_pos_tag
        self.replace_unknown_lemma = replace_unknown_lemma
        
    def annot_text(self, text, encoding=None, errors='strict'):
        if encoding:
            text = text.decode(encoding, errors)
        else:
            assert isinstance(text, unicode)
            
        tagger_out = self._tree_tagger(text)
        return self._extract_from_text(tagger_out)
    
    def annot_sentences(self, sentences, encoding=None):
        xml_sent_tag = "seg"
        text = self._embed_in_xml(sentences, encoding, xml_sent_tag)
        tagger_out = self._tree_tagger(text)
        return self._extract_from_xml(tagger_out, xml_sent_tag)

    def _embed_in_xml(self, sentences, encoding, xml_sent_tag):
        # Embed sentences in a simple xml strcture.
        # Default Treetagger skips xml tags but keep them in its output,
        # so sentence boundaries are retained.
        text = u"<doc>"
        
        for sent in sentences:
            if encoding:
                sent = sent.decode(encoding)
            text += u"<{0}>{1}</{2}>".format(xml_sent_tag, sent, xml_sent_tag)
            
        text += u"</doc>"
        return text
        
    def _tree_tagger(self, text):
        log.debug("TreeTagger input:\n" + text)
        
        # convert from unicode to given encoding
        # we may loose some data here!
        text = text.encode(self.tagger_encoding, "backslashreplace")
        
        # create pipe to tagger
        log.debug("Calling TreeTagger as " + self.command)
        tagger_proc = subprocess.Popen(self.command, 
                                       shell=True,
                                       stdin=subprocess.PIPE, 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

        # send text and retrieve tagger output 
        tagger_out, tagger_err = tagger_proc.communicate(text)
        
        # and convert back from given encoding to unicode
        tagger_out = tagger_out.decode(self.tagger_encoding, 
                                       "replace")        
        log.debug("TreeTagger standard output:\n" + tagger_out)
        log.debug("TreeTagger standard error:\n" + tagger_err)
        
        return tagger_out 
    
    def _extract_from_text(self, tagger_out):
        graph_list = []
        start_new_graph = True
        
        for line in tagger_out.strip().split("\n"):
            if start_new_graph:
                graph = self._add_new_graph(n=len(graph_list) + 1)
                graph_list.append(graph)
                start_new_graph = False
                prev_node = None
                
            new_node, pos_tag = self._add_new_node(line, graph, prev_node)
            
            if pos_tag == self.eos_pos_tag:
                start_new_graph = True
                prev_node = None
            else:
                prev_node = new_node
                
        return graph_list
    
    def _extract_from_xml(self, tagger_out, xml_sent_tag):
        graph_list = []
        sent_count = 0
        # XML parser interprets "<unknown>" as an XML tag and crashes when it
        # finds no matching "</unknown>" tag, so replace by __UNKNOWN__
        tagger_out = tagger_out.replace(self.unknown_lemma, self.unknown)
        # Replace illegal ampercents too
        tagger_out = tagger_out.replace("&", "&amp;")
        # tagger_out is unicode but XML parser expects byte string
        # (cStringIO cannot cope with unicode either)
        tagger_out = tagger_out.encode("utf-8")
        
        for _, elem in et.iterparse(cStringIO.StringIO(tagger_out)):
            if elem.tag == xml_sent_tag:
                sent_count += 1
                graph = self._add_new_graph(sent_count)
                graph_list.append(graph)
                prev_node = None
                
                for line in elem.text.strip().split("\n"):
                    prev_node, _ = self._add_new_node(line, graph, prev_node)
                    
        return graph_list

    def _add_new_graph(self, graph_id=None, n=None):
        if n:
            graph_id = "{0:03}".format(n)
            
        log.info("creating graph {}".format(graph_id))
        return TransGraph(id=graph_id)

    def _add_new_node(self, line, graph, prev_node):
        # fix: TreeTagger for English sometimes produces output like
        # 'that\t\tIN\tthat',
        line = line.replace("\t\t", "\t")                
        word, pos, lemma = line.split("\t")
        # TODO how to handle ambiguity in lemmatization as in er|sie|Sie   
   
        if lemma == self.unknown:
            log.warn(u"lemma unknown")
            if self.replace_unknown_lemma:
                log.warn(u"using word {0} instead".format(word))
                lemma = word
       
        new_node = graph.add_source_node(word=word, lemma=lemma, pos=pos)
        
        if prev_node:
            graph.add_word_order_edge(prev_node, new_node)
        else:
            graph.set_source_start_node(new_node)
            
        return new_node, pos



class TreeTaggerEnglish(TreeTagger):
    """
    annotation of English input text with TreeTagger 
    """
    
    def __init__(self, 
                 command=config["tagger"]["en"]["command"], 
                 tagger_encoding=config["tagger"]["en"]["encoding"], 
                 eos_pos_tag="SENT",
                 *args, **kwargs):
        TreeTagger.__init__(self, command=command, 
                            tagger_encoding=tagger_encoding,
                            eos_pos_tag=eos_pos_tag, 
                            *args, **kwargs)
  


class TreeTaggerGerman(TreeTagger):
    """
    annotation of German input text with TreeTagger 
    """
    
    def __init__(self, 
                 command=config["tagger"]["de"]["command"], 
                 tagger_encoding=config["tagger"]["de"]["encoding"], 
                 eos_pos_tag="$.",
                 *args, **kwargs):
        TreeTagger.__init__(self, command=command, 
                            tagger_encoding=tagger_encoding,
                            eos_pos_tag=eos_pos_tag, 
                            *args, **kwargs)
        
        
class ILSP_NLP_Greek(Annotator):
    """
    annotation of Greek input text with ILSP NLP web service
    
    http://registry.elda.org/services/180
    http://sifnos.ilsp.gr/soaplab2-axis/
    """

    wsdl_url = "http://nlp.ilsp.gr/soaplab2-axis/services/ilsp.ilsp_nlp?wsdl"
    
    def __init__(self, *arg, **kargs):
        self.client = suds.client.Client(self.wsdl_url)
        
        
    def annot_text(self, text, encoding=None, errors='strict'):
        if encoding:
            text = text.decode(encoding, errors)
        else:
            assert isinstance(text, unicode)
        
        output_url = self._ilsp_nlp(text, input_type="txt")  
        return self._extract_from_xml(output_url, sent_tag="s")
                                      
                                      
    def annot_sentences(self, sentences, encoding=None):
        xml_source = self._embed_in_xml(sentences, encoding)
        output_url = self._ilsp_nlp(xml_source, input_type="xcesbasic")
        return self._extract_from_xml(output_url, sent_tag="p")
    
    def _embed_in_xml(self, sentences, encoding):
        # Embed sentences in minimal XCES xml .
        # Sentences are embedded as paragraphs (<p> rather than <s>), 
        # because ilsp_nlp performs both sentence chunking and tokenization.
        # Upon parsing the results, the sentence boundaries are ignored
        # by calling _extract_from_xml() with sent_tag="p".
        # This prevents sentences from being chunked by ilsp_nlp.
        xml_source = u"<cesDoc><cesHeader /><text><body>\n"
        
        for sent in sentences:
            if encoding:
                sent = sent.decode(encoding)
            else:
                assert isinstance(sent, unicode) 
            xml_source += u"<p>" + sent + u"</p>\n"
                
        xml_source += u"</body></text></cesDoc>"
        return xml_source

    def _create_map(self, input_data, input_type):
        # input type is either text ("txt") or XCES document ("xcesbasic")
        param_map = self.client.factory.create("ns3:Map")
        items = { "language" :"el",
                  "InputType": input_type,
                  "input_direct_data": input_data }
        
        for key, value in items.items():
            map_item = self.client.factory.create("ns3:mapItem")
            map_item.key = key
            map_item.value = value
            param_map.item.append(map_item)
            
        return param_map

    def _ilsp_nlp(self, input_data, input_type):
        log.debug(u"input data:\n" + input_data)
        param_map = self._create_map(input_data, input_type)
        job_id = self.client.service.createAndRun(param_map)
        log.debug("ilsp_nlp service job id:\n" + job_id)
        self.client.service.waitFor(job_id)
        results = self.client.service.getResults(job_id)
        report = results[0][2].value
        log.debug("ilsp_nlp service report:\n" + report)
        status = results[0][1].value
        log.debug("ilsp_nlp service return status: " + status)
        if status != "0":
            raise TGException("annotation of Greek text failed")
        output_url = results[0][0].value
        log.debug("ilsp_nlp service output URL: " + output_url)
        return output_url
        # TODO: are we suppposed to call destroy(xs:string jobId, )?
    
    def _extract_from_xml(self, output_url, sent_tag):
        graph_list = []
        sent_count = 0
        output = urllib.urlopen(output_url)
        
        # some silly moves to get the root element
        context = et.iterparse(output, 
                               events=("start", "end"))
        context = iter(context)
        _, root = context.next()
        
        # lousy way to handle namespaces
        name_space_prefix = "{http://www.xces.org/schema/2003}"
        sent_tag = name_space_prefix + sent_tag
        token_tag = name_space_prefix + "t"
    
        for event, elem in context:   
            if event == "start":
                if elem.tag == sent_tag:
                    graph_id = elem.get("id")
                    log.info("creating graph {}".format(graph_id))                    
                    graph = TransGraph(id=graph_id)
                    graph_list.append(graph)
                    prev_node = None
            elif event == "end":
                if elem.tag == token_tag:
                    new_node = graph.add_source_node(
                        word=elem.get("word"), 
                        lemma=elem.get("lemma"), 
                        pos=elem.get("tag")) 
                    if prev_node:
                        graph.add_word_order_edge(prev_node, new_node)
                    else:
                        graph.set_source_start_node(new_node)
                    prev_node = new_node
                    
        if log.isEnabledFor(logging.DEBUG):
            log.debug(u"result:\n" + 
                      et.tostring(root, encoding="utf-8").decode("utf-8"))
        
        output.close()
        return graph_list

    


def get_annotator(lang, *args, **kwargs):
    if lang == "en":
        return TreeTaggerEnglish(*args, **kwargs)
    elif lang == "gr":
        return ILSP_NLP_Greek(*args, **kwargs)
    elif lang == "de":
        return TreeTaggerGerman(*args, **kwargs)
    else:
        raise ValueError("no annotator for language {}".format(lang))