"""
lookup of translation candidates in dictionary
"""

import logging as log

import graphproc


class Lookup(graphproc.GraphProces):
    """
    determine translation candidates by lexical lookup in dictionary and add
    them to the translation graph
    """ 

    def __init__(self, dictionary, max_n_gram_size=5):
        self.dictionary = dictionary
        self.max_n_gram_size = max_n_gram_size
        
    def _single_run(self, graph):
        source_nodes = []
        source_lempos = []
        delimiter = self.dictionary.delimiter
        
        for sn, data in graph.source_nodes_iter(with_data=True):
            source_nodes.append(sn)
            lempos = data["lemma"] + delimiter + data["tag"]
            source_lempos.append(lempos)
        
        for i in range(len(source_nodes)):
            for j in range(i, min(len(source_nodes), i + self.max_n_gram_size)):
                key = " ".join(source_lempos[i:j+1])
                
                try:
                    translations = self.dictionary[key]
                except KeyError:
                    continue
                
                if i == j:
                    sn = source_nodes[i]
                else:
                    sn = graph.add_hyper_node(source_nodes[i:j+1], is_source=False)
                
                for candidate in translations:
                    target_lempos = candidate.split()
                    
                    try:
                        lemma, tag = target_lempos[0].split(delimiter)
                    except ValueError:
                        log.error("skipping ill-formed candidate {0}".format(target_lempos[0]))
                        continue
                        
                    tn = graph.add_target_node(lemma=lemma, tag=tag)
                    
                    if len(target_lempos) > 1:
                        u = tn
                        target_nodes = [u]
                        
                        for lempos in target_lempos[1:]:
                            try:
                                lemma, tag = lempos.split(delimiter)
                            except ValueError:
                                log.error("skipping ill-formed candidate {0}".format(lempos))
                                continue
                            
                            v = graph.add_target_node(lemma=lemma, tag=tag)
                            target_nodes.append(v)
                            graph.add_word_order_edge(u,v)
                            u = v
                    
                        tn = graph.add_hyper_node(target_nodes)
                                                  
                                        
                    graph.add_translation_edge(sn, tn)        