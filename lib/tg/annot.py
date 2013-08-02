# -*- coding: utf-8 -*-

"""
constructing linguistically annotated translation graphs
"""

# TODO:
# - more documentation

import codecs
import cStringIO
import logging
import re
import subprocess
import urllib
import xml.etree.ElementTree as et

import suds

from tg.config import config
from tg.transgraph import TransGraph
from tg.exception import TGException



log = logging.getLogger(__name__)



class Annotator(object):
    """
    Abstract base class for all annotators that annotate input text with
    part-of-speech tags and lemmas.
    """
    
    # string value to use for unknown values (POS tag or lemma)
    unknown = "__UNKNOWN__"    
    
    def annot_text(self, text, encoding=None, errors='strict'):
        """
        Annotate free text
        
        This allows free text input where sentence boundaries are not given.
        It therefore only works with annotators that can do sentence splitting.
        The character encoding should preferably be the same as the one
        supported by the tagger-lemmatizer to prevent conversion errors.
        
        Parameters
        ----------
        text: byte string
            encoded text
        encoding: string
            character encoding; if None, text is assumed to be unicode 
        errors: string
            how to handle encoding conversion errors (cf. str.encode())
        
        Returns
        -------
        graphs: list
            list of Transgraph instances
        """
        if encoding:
            text = text.decode(encoding, errors=errors)
        else:
            assert isinstance(text, unicode)
            
        return self._annot_text(text)
    
    def annot_text_file(self, inf, encoding="utf-8", errors='strict'):
        """
        Annotate free text from file
        
        This allows free text input where sentence boundaries are not given.
        The character encoding should preferably be the same as the one
        supported by the tagger-lemmatizer.
        
        Parameters
        ----------
        text: byte string
            encoded text
        encoding: string
            character encoding 
        errors: string
            how to handle encoding conversion errors (cf. str.encode())
        
        Returns
        -------
        graphs: list
            list of Transgraph instances
        """
        if not hasattr(inf, "read"):
            inf = codecs.open(inf, encoding=encoding, errors=errors)
        text = inf.read()    
        return self.annot_text(text)
    
    def annot_sentences(self, sentences, encoding=None, errors='strict',
                        ids=None):
        """
        Annotate sentences
        
        This allows free text input where sentence boundaries are not given.
        The character encoding should preferably be the same as the one
        supported by the tagger-lemmatizer.
        
        Parameters
        ----------
        sentences: iterable of unicode strings
            sequence of sentences
        encoding: string
            character encoding; if None, sentences assumed to be unicode strings
        ids: iterable of strings
            sentence identifiers
            
        Returns
        -------
        graphs: list
            list of Transgraph instances
        """
        if encoding:
            sentences = (s.decode(encoding, errors=errors) for s in sentences)
        # else there is no cheap way to check all sentences are unicode
        
        return self._annot_sentences(sentences, ids)
    
    def annot_xml(self, source, xml_sent_tag="seg", id_attr="id"):
        """
        Annotate sentences from XML string
        
        In contract to free text, this assumes sentence boundaries are given.
        Character encoding is assumed to be utf-8 unless specified otehrwise
        in the xml header.
        
        Parameters
        ----------
        source: byte string
            file or filename with xml input 
        xml_sent_tag: string
            xml tag for sentences
        id_attr: string
            attribute of xml sentence tag that identifies the sentence
        
        Returns
        -------
        graphs: list
            list of Transgraph instances
        """
        # Source MUST be a byte string (i.e. encoded).
        # If encoding is different from utf-8, 
        # then it must be specifid by the XMl header
        assert isinstance(source, str)
        inf = cStringIO.StringIO(source)
        return self.annot_xml_file(inf, xml_sent_tag, id_attr)
        
    def annot_xml_file(self, inf, xml_sent_tag="seg", id_attr="id"):
        """
        Annotate sentences from XML input
        
        In contract to free text, this assumes sentence boundaries are given.
        Character encoding is assumed to be utf-8 unless specified otehrwise
        in the xml header.
        
        Parameters
        ----------
        inf: file or string
            file or filename with xml input
        xml_sent_tag: string
            xml tag for sentences
        id_attr: string
            attribute of xml sentence tag that identifies the sentence
        
        Returns
        -------
        graphs: list
            list of Transgraph instances
        """
        sentences, ids = self._extract_sentences_from_xml(inf, xml_sent_tag,
                                                          id_attr)
        return self._annot_sentences(sentences, ids)
        
    def _extract_sentences_from_xml(self, inf, xml_sent_tag, id_attr):
        # this assumes that a sentence contains no internal XML markup 
        # (e.g. <b>...</b>)
        pairs =  ( ( elem.text, elem.get(id_attr))
                   for _, elem in et.iterparse(inf)
                   if elem.tag == xml_sent_tag )
        return zip(*pairs)
        
    def _add_new_graph(self, graph_id=None, n=None):
        log.info("creating graph (id={}, n={})".format(graph_id, n))
        return TransGraph(id=graph_id, n=n)   
    
    def _annot_text(self, text):
        pass
    
    def _annot_sentences(self, sentences, ids=None):
        pass
        
        
            
        
class TreeTagger(Annotator):
    """
    Annotation of input text with TreeTagger 
    """
    # The approach from the Annotator base class is to extract sentences from
    # the XML file and then call self.annot_sentences. However, TreeTagger
    # can process XML input directly, so instead we tag-lemmatize the xml
    # input and then create graphs directly from the XML output.
    
    # string used by TreeTagger for unknown lemmas
    unknown_lemma = u"<unknown>"
    
    def __init__(self, command, tagger_encoding, eos_pos_tag=None,
                 replace_unknown_lemma=True):
        Annotator.__init__(self)
        self.command = command
        self.tagger_encoding = tagger_encoding
        self.eos_pos_tag = eos_pos_tag
        self.replace_unknown_lemma = replace_unknown_lemma
        
    def annot_xml(self, source, xml_sent_tag="seg", id_attr="id"):
        #  strip utf-8 byte-order-mark
        source = source.lstrip(codecs.BOM_UTF8)
        
        # figure out encoding from xml header or default to utf-8
        m = re.match('.?<\?xml[^<>]+encoding="(.+)"', source)
        if m:
            encoding = m.groups()[0]
        else:
            encoding = "utf-8"
        source = source.decode(encoding)
        
        # run Treetagger
        tagger_out = self._tree_tagger(source)
        
        # XML parser interprets "<unknown>" as an XML tag and crashes when it
        # finds no matching "</unknown>" tag, so replace by __UNKNOWN__
        tagger_out = tagger_out.replace(self.unknown_lemma, self.unknown)
        
        # Replace illegal ampercents too
        tagger_out = tagger_out.replace(u"&", u"&amp;")

        # tagger_out is unicode but XML parser expects byte string
        # (cStringIO cannot cope with unicode either)
        tagger_out = tagger_out.encode("utf-8")
        graph_list = []        
        
        for _, elem in et.iterparse(cStringIO.StringIO(tagger_out)):
            if elem.tag == xml_sent_tag:
                graph = self._add_new_graph(graph_id=elem.get(id_attr),
                                            n=len(graph_list) + 1)
                graph_list.append(graph)
                prev_node = None
                text = elem.text.strip()
                
                if text:
                    for line in text.split(u"\n"):
                        prev_node, _ = self._add_new_node(line, graph, 
                                                          prev_node)
        return graph_list
        
    def annot_xml_file(self, inf, xml_sent_tag="seg", id_attr="id"):
        if isinstance(inf, basestring):
            inf = open(inf)
            
        source = inf.read()
        return self.annot_xml(source, xml_sent_tag, id_attr)
        
    def _annot_text(self, text):
        tagger_out = self._tree_tagger(text)
        return self._extract_sentences_from_text(tagger_out)
    
    def _annot_sentences(self, sentences, ids=None):
        xml_sent_tag = "seg"
        id_attr = "id"
        if not ids:
            ids = ("{:03d}".format(i+1) for i in range(len(sentences)))
        source = self._embed_in_xml(sentences, xml_sent_tag, id_attr, ids)
        return self.annot_xml(source, xml_sent_tag, id_attr)
    
    def _embed_in_xml(self, sentences, xml_sent_tag, id_attr, ids):
        # Embed sentences in a simple xml strcture.
        # Default Treetagger skips xml tags but keep them in its output,
        # so sentence boundaries are retained.
        text = u"<doc>\n"
            
        for sent, id in zip(sentences, ids):
            text += u'<{0} {1}="{2}">{3}</{4}>\n'.format(xml_sent_tag,
                                                         id_attr,
                                                         id,
                                                         sent, 
                                                         xml_sent_tag)
        text += u"</doc>\n"
        return text.encode("utf-8")
        
    def _tree_tagger(self, text):
        log.debug(u"TreeTagger input:\n" + text)
        
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
        log.debug(u"TreeTagger standard output:\n" + tagger_out)
        log.debug(u"TreeTagger standard error:\n" + tagger_err)
        
        return tagger_out 
    
    def _extract_sentences_from_text(self, tagger_out):
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
    
    def _add_new_node(self, line, graph, prev_node):
        # fix: TreeTagger for English sometimes produces output like
        # 'that\t\tIN\tthat',
        line = line.replace("\t\t", "\t")                
        word, pos, lemma = line.split("\t")
        # TODO how to handle ambiguity in lemmatization as in er|sie|Sie   
   
        if lemma == self.unknown:
            if self.replace_unknown_lemma:
                lemma = word
                log.warn(u'unknown lemma for word "{0}", '
                         u'using word instead'.format(word))
            else:
                log.warn(u'unknown lemma for word "{0}", '
                         u'using {1} instead'.format(word, self.unknown))
       
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
        
    def _annot_text(self, text):
        output_url = self._ilsp_nlp(text, input_type="txt")  
        return self._parse_ilsp_nlp_output(output_url, sent_tag="s")
                                      
    def _annot_sentences(self, sentences, ids=None):
        if not ids:
            ids = ("{:03d}".format(i+1) for i in range(len(sentences)))
        xml_source = self._embed_in_xml(sentences, ids)
        output_url = self._ilsp_nlp(xml_source, input_type="xcesbasic")
        return self._parse_ilsp_nlp_output(output_url, sent_tag="p")
    
    def _embed_in_xml(self, sentences, ids):
        # Embed sentences in minimal XCES xml .
        # Sentences are embedded as paragraphs (<p> rather than <s>), 
        # because ilsp_nlp performs both sentence chunking and tokenization.
        # Upon parsing the results, the sentence boundaries are ignored
        # by calling _extract_from_xml() with sent_tag="p".
        # This prevents sentences from being chunked by ilsp_nlp.
        xml_source = u"<cesDoc><cesHeader /><text><body>\n"
        
        for sent, id in zip(sentences, ids):
            xml_source += u'<p id="{}">'.format(id) + sent + u"</p>\n"
                
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
    
    def _parse_ilsp_nlp_output(self, output_url, sent_tag):
        graph_list = []
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
                    graph = self._add_new_graph(n=len(graph_list) + 1,
                                                graph_id=elem.get("id"))
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

    


class OsloBergenTagger(Annotator):
    """
    annotation of Norwegian (Bokmål) input text with Oslo-Bergen Tagger 
    """
    tagger_encoding="utf-8"
    eos_marker = u" END_OF_SENTENCE . "
    eos_line = ( u"END_OF_SENTENCE\tEND_OF_SENTENCE\tsubst_prop\n"
                 u".\t$.\t<punkt>_<<<\n" )
    
    def __init__(self, command=config["tagger"]["no"]["command"]):
        Annotator.__init__(self)    
        self.command = command
        
    def _annot_text(self, text):
        tagger_out = self._obt(text)
        return self._parse_obt_output(tagger_out)
                                       
    def _annot_sentences(self, sentences, ids=None):
        # There is no easy way to enforce sentence boundaries in OBT, so we
        # insert a silly eos_marker which always triggers a sentence
        # boundary. 
        text = self.eos_marker.join(sentences)
        tagger_out = self._obt(text)
        return self._parse_obt_output(tagger_out, self.eos_line, ids)
    
    def _obt(self, text):
        log.debug(u"OBT input:\n" + text)
        
        # convert from unicode to given encoding
        # we may loose some data here!
        text = text.encode(self.tagger_encoding, "backslashreplace")
        
        # create pipe to tagger
        log.debug("Calling OBT as " + self.command)
        tagger_proc = subprocess.Popen(self.command, 
                                       shell=True,
                                       stdin=subprocess.PIPE, 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

        # send text and retrieve tagger output 
        tagger_out, tagger_err = tagger_proc.communicate(text)
        
        # and convert back from given encoding to unicode
        tagger_err = tagger_err.decode(self.tagger_encoding, 
                                       "replace")        
        tagger_out = tagger_out.decode(self.tagger_encoding, 
                                       "replace")        
        log.debug(u"OBT standard output:\n" + tagger_out)
        log.debug(u"OBT standard error:\n" + tagger_err)       
        
        return tagger_out
    
    def _parse_obt_output(self, tagger_out, eos_line="\n\n", ids=None):
        graph_list = []
        annot_sentences = tagger_out.strip().split(eos_line)
        if not ids:
            ids = ("{:03d}".format(i+1) for i in range(len(annot_sentences)))
        
        for annot_sent, id in zip(annot_sentences, ids):
            graph = self._add_new_graph(n=len(graph_list) + 1,
                                        graph_id=id)
            graph_list.append(graph)
            prev_node = None
            
            for line in annot_sent.strip().split("\n"):
                # ignore sentence boundaries inserted by tagger (empty lines)
                # when using XML input
                if line.strip():
                    prev_node = self._add_new_node(line, graph, prev_node)
                
        return graph_list

    def _add_new_node(self, line, graph, prev_node):
        word, lemma, tag = line.split("\t")
        # use only the first part of the full tag as POS, but retain full tag
        pos = tag.split("_", 1)[0]
        new_node = graph.add_source_node(word=word, lemma=lemma, pos=pos,
                                         tag=tag)
        
        if prev_node:
            graph.add_word_order_edge(prev_node, new_node)
        else:
            graph.set_source_start_node(new_node)
            
        return new_node
    


def get_annotator(lang, *args, **kwargs):
    if lang == "en":
        return TreeTaggerEnglish(*args, **kwargs)
    elif lang == "gr":
        return ILSP_NLP_Greek(*args, **kwargs)
    elif lang == "de":
        return TreeTaggerGerman(*args, **kwargs)
    elif lang == "no":
        return OsloBergenTagger(*args, **kwargs)
    else:
        raise ValueError("no annotator for language {}".format(lang))