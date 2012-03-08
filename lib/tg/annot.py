"""
constructing linguistically annotated translation graphs
"""

import logging as log
import subprocess

import graphproc
import transgraph


class Annotator(graphproc.GraphProces):
    """
    performs linguistic annotation (tokenization, lemmatization and tagging)
    and conversion to graph
    """
    
    def _single_run(self, text):
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
        
        # convert from utf-8 to Latin1 encoding
        # we may loose some data here!
        text = text.encode("latin1", "backslashreplace")
        
        # create pipe to tagger
        log.debug("Calling TreeTagger as " + self.command)
        tagger_proc = subprocess.Popen(self.command, shell=True,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

        # send text and retrieve tagger output 
        tagger_out, tagger_err = tagger_proc.communicate(text)
        
        # and convert back from Latin1 encoding to utf-8    
        tagger_out = tagger_out.decode("latin1")        
        log.debug("TreeTagger standard output:\n" + tagger_out)
        log.debug("TreeTagger standard error:\n" + tagger_err)
        
        return tagger_out
    
    
    def convert(self, tagger_out):
        graph_list = []
        graph_count = 1
        graph_id = "{0:03}".format(graph_count)
        graph = transgraph.TransGraph(id=graph_id)
        prev_node = None
        
        for line in tagger_out.strip().split("\n"):
            word, tag, lemma = line.split("\t")
            new_node = graph.add_source_node(word=word, tag=tag, lemma=lemma)
            
            if prev_node:
                graph.add_word_order_edge(prev_node, new_node)
            else:
                graph.set_source_start_node(new_node)
                
            prev_node = new_node
            
            if tag == self.sent_end_tag:
                graph_list.append(graph)
                graph_count += 1
                graph = transgraph.TransGraph(id=graph_count)
                prev_node = None
                
        return graph_list

    
    
    
