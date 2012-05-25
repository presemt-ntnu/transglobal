"""
constructing linguistically annotated translation graphs
"""

import xml.etree.cElementTree as et
import logging
import cStringIO
import subprocess
import urllib

import suds

from tg.config import config
from tg.graphproc import GraphProces
from tg.transgraph import TransGraph
from tg.exception import TGException



log = logging.getLogger(__name__)


class Annotator(GraphProces):
    """
    performs linguistic annotation (tokenization, lemmatization and tagging)
    and conversion to graph
    """
    
    def _single_run(self, text):
        log.info("starting text annotation with " +
                 self.__class__.__name__)
        return self.convert(self.annotate(text))
    
    def annotate(self, text):
        """
        call tagger to annotate text
        """
        raise NotImplementedError
    
    def convert(self, tagger_out):
        """
        convert tagger output to translation graphs
        """
        raise NotImplementedError
    
    
    
class TreeTagger(Annotator):
    """
    annotation of input text with TreeTagger 
    """
    
    lemma_unknown = "<unknown>"
    
    def __init__(self, command, encoding, xml_sent_tag=None,
                 eos_pos_tag=None, replace_unknown_lemma=True):
        Annotator.__init__(self)
        self.command = command
        self.encoding = encoding
        self.replace_unknown_lemma = replace_unknown_lemma
        
        if xml_sent_tag:
            self.convert = self.convert_from_xml
            self.xml_sent_tag = xml_sent_tag
        elif eos_pos_tag:
            self.convert = self.convert_from_text
            self.eos_pos_tag = eos_pos_tag
        else:
            raise ValueError("cannot convert annotation from unknown input "
                             "format; set either eos_sent_tag or xml_sent_tag")
        
    def annotate(self, text):
        log.debug("TreeTagger input:\n" + text)
        
        # convert from unicode to given encoding
        # we may loose some data here!
        text = text.encode(self.encoding, "backslashreplace")
        
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
        tagger_out = tagger_out.decode(self.encoding)        
        log.debug("TreeTagger standard output:\n" + tagger_out)
        log.debug("TreeTagger standard error:\n" + tagger_err)
        
        return tagger_out 
    
    def convert_from_text(self, tagger_out):
        graph_list = []
        start_new_graph = True
        
        for line in tagger_out.strip().split("\n"):
            if start_new_graph:
                graph = self.add_new_graph(n=len(graph_list) + 1)
                graph_list.append(graph)
                start_new_graph = False
                prev_node = None
                
            new_node, pos_tag = self.add_new_node(line, graph, prev_node)
            
            if pos_tag == self.eos_pos_tag:
                start_new_graph = True
                prev_node = None
            else:
                prev_node = new_node
                
        return graph_list
    
    def convert_from_xml(self, tagger_out):
        graph_list = []
        sent_count = 0
        # HACK: XML parser interprets "<unknown>" as an XML tag and crashes
        # when it finds no matching "<unknown/>" tag, so we escape the angled
        # brackets using entities. Angled brackets are later restored when
        # calling ".text" on a sentence element.
        tagger_out = tagger_out.replace("<unknown>", "&lt;unknown&gt;")
        # tagger_out is unicode but XML parser expects encoded input
        # TODO: char encoding specified in XML header may be different
        tagger_out = tagger_out.encode("utf-8")
        
        for _, elem in et.iterparse(cStringIO.StringIO(tagger_out)):
            if elem.tag == self.xml_sent_tag:
                sent_count += 1
                graph = self.add_new_graph(sent_count)
                graph_list.append(graph)
                prev_node = None
                
                for line in elem.text.strip().split("\n"):
                    prev_node, _ = self.add_new_node(line, graph, prev_node)
                    
        return graph_list

    def add_new_graph(self, graph_id=None, n=None):
        if n:
            graph_id = "{0:03}".format(n)
            
        log.info("creating graph {}".format(graph_id))
        return TransGraph(id=graph_id)

    def add_new_node(self, line, graph, prev_node):
        # fix: TreeTagger for English sometimes produces output like
        # 'that\t\tIN\tthat',
        line = line.replace("\t\t", "\t")                
        word, pos, lemma = line.split("\t")
        # TODO how to handle ambiguity in lemmatization as in er|sie|Sie   
   
        if lemma == self.lemma_unknown and self.replace_unknown_lemma:
            log.warn(u"lemma unknown;using word {0} instead".format(word))
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
                 encoding=config["tagger"]["en"]["encoding"], 
                 eos_pos_tag="SENT",
                 *args, **kwargs):
        TreeTagger.__init__(self, command=command, encoding=encoding,
                            eos_pos_tag=eos_pos_tag, *args, **kwargs)
  


class TreeTaggerGerman(TreeTagger):
    """
    annotation of German input text with TreeTagger 
    """
    
    def __init__(self, 
                 command=config["tagger"]["de"]["command"], 
                 encoding=config["tagger"]["de"]["encoding"], 
                 eos_pos_tag="$.",
                 *args, **kwargs):
        TreeTagger.__init__(self, command=command, encoding=encoding,
                            eos_pos_tag=eos_pos_tag, *args, **kwargs)



class ILSP_NLP_Greek(Annotator):
    """
    annotation of Greek input text with ILSP NLP web service
    
    http://registry.elda.org/services/180
    http://sifnos.ilsp.gr/soaplab2-axis/
    """

    wsdl_url = "http://nlp.ilsp.gr/soaplab2-axis/services/ilsp.ilsp_nlp?wsdl"
    
    def __init__(self, xml_sent_tag=None):
        self.client = suds.client.Client(self.wsdl_url)

    def _create_map(self, text):
        param_map = self.client.factory.create("ns3:Map")
        items = { "language" :"el",
                  "InputType": "txt",
                  "input_direct_data": text }
        
        for key, value in items.items():
            map_item = self.client.factory.create("ns3:mapItem")
            map_item.key = key
            map_item.value = value
            param_map.item.append(map_item)
            
        return param_map

    def annotate(self, text):
        # TODO: handle XMl input by converting to XCES format
        log.debug(u"input text:\n" + text)
        param_map = self._create_map(text)
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
    
    def convert(self, output_url):
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
        sent_tag = name_space_prefix + "s"
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