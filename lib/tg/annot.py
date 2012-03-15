"""
constructing linguistically annotated translation graphs
"""

import logging
import subprocess

import graphproc
import transgraph


log = logging.getLogger(__name__)



class Annotator(graphproc.GraphProces):
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
    
    
class TreeTaggerEnglish(Annotator):
    """
    annotation of English input text with TreeTagger 
    """
    
    def __init__(self, command="tree-tagger-english"):
        self.command = command
        self.sent_end_tag = "SENT"
        
        
    def annotate(self, text):
        log.debug("TreeTagger input:\n" + text)
        
        # convert from unicode to Latin1 encoding
        # we may loose some data here!
        text = text.encode("latin1", "backslashreplace")
        
        # create pipe to tagger
        log.debug("Calling TreeTagger as " + self.command)
        tagger_proc = subprocess.Popen(self.command, shell=True,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

        # send text and retrieve tagger output 
        tagger_out, tagger_err = tagger_proc.communicate(text)
        
        # and convert back from Latin1 to unicode
        tagger_out = tagger_out.decode("latin1")        
        log.debug("TreeTagger standard output:\n" + tagger_out)
        log.debug("TreeTagger standard error:\n" + tagger_err)
        
        return tagger_out
    
    
    def convert(self, tagger_out):
        graph_list = []
        make_new_graph = True
        
        for line in tagger_out.strip().split("\n"):
            if make_new_graph:
                graph_id = "{0:03}".format(len(graph_list) + 1)
                log.info("creating graph " + graph_id)
                graph = transgraph.TransGraph(id=graph_id)
                make_new_graph = False
                prev_node = None
                
            word, tag, lemma = line.split("\t")
            new_node = graph.add_source_node(word=word, tag=tag, lemma=lemma)
            
            if prev_node:
                graph.add_word_order_edge(prev_node, new_node)
            else:
                graph.set_source_start_node(new_node)
            
            if tag == self.sent_end_tag:
                graph_list.append(graph)
                make_new_graph = True
                prev_node = None
            else:
                prev_node = new_node
                
        return graph_list

    
    
    
