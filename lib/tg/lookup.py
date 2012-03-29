"""
lookup of translation candidates in dictionary
"""

import logging

import graphproc

log = logging.getLogger(__name__)



class Lookup(graphproc.GraphProces):
    """
    determine translation candidates by lexical lookup in dictionary and add
    them to the translation graph
    """ 

    def __init__(self, dictionary, max_n_gram_size=5):
        self.dictionary = dictionary
        self.max_n_gram_size = max_n_gram_size
        
    def _single_run(self, graph):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        source_nodes, tagger_lempos_list = self._collect(graph)
        
        for i in range(len(source_nodes)):
            has_translation = False
            
            # Attempt to lookup tagger lempos n-grams starting with
            # tagger_lempos[i:i+1] up till max_n_gram_size.
            for j in range(i + 1, min(len(source_nodes), 
                                      i + 1 + self.max_n_gram_size)):
                lempos_subseq = tagger_lempos_list[i:j]
                entries = self._lookup_lempos_seq(lempos_subseq)
                
                if entries:
                    has_translation = True
                    self._add_translations(i, j, source_nodes, graph, entries)
                
            if not has_translation:
                log.warn(u"no translation found for {0}".format(
                    tagger_lempos_list[i]))

    def _collect(self, graph):
        # collect lists of ordered source nodes and corresponding tagger
        # lempos
        source_nodes = []
        tagger_lempos_list = []

        for sn in graph.source_nodes_iter(ordered=True):
            source_nodes.append(sn)
            tagger_lempos_list.append(graph.lempos(sn))
            
        return source_nodes, tagger_lempos_list
            
    def _lookup_lempos_seq(self, lempos_seq):
        # Try looking up lempos combinations first. Note that dictionary
        # lookup falls back on lemma only if any lempos is not found. The
        # dictionary may also map tagger pos tags to lexicon pos tags.
        
        # Result of lookup is list over entries consisting of
        # 1. a lexicon lempos (NB different from orginal tagger lempos!)
        # 2. a tuple of translation candidates in the form of 
        #    target lempos strings
        # For example:
        # [('rerun/vv', ('wiederholen/v*.full', 
        #                'wiederholt/v*.full', 
        #                u'wieder/v*.full auff\xfchren/v*.full', 
        #                u'f\xfchren/v*.full wieder/v*.full auf/v*.full', 
        #                'holen/v*.full wieder/v*.full')), 
        #  ('rerun/n', ('Wiederholungslauf/n', 
        #               'Wiederholung/n', 
        #               u'Wiederauff\xfchrung/n'))]   
        entries = []
        # convert list to string
        lempos_seq = " ".join(lempos_seq)
        log.debug(u"looking up lempos sequence: {}".format(lempos_seq))
        # dictionary actually returns an iterator, but we return a list of
        # entries rather than an iterator, because the caller needs to detect
        # if there are any results at all
        for pair in self.dictionary.get(lempos_seq, []):
            log.debug(u"found result: {0} --> {1}".format(
                pair[0],
                ", ".join(pair[1])))
            entries.append(pair)
                
        return entries
  
    def _add_translations(self, i, j, source_nodes, graph, entries):
        sn = self._get_source_node(i, j, source_nodes, graph)
        graph.node[sn]["lex_lempos"] = lex_lempos_list = []
            
        for lex_lempos, translations in entries:
            lex_lempos_list.append(lex_lempos)
            for target_lempos_seq in translations:
                tn = self._get_target_node(graph, target_lempos_seq)
                graph.add_translation_edge(sn, tn)

    def _get_target_node(self, graph, target_lempos_seq):
        target_lempos_seq = target_lempos_seq.split()
        tn = self._add_single_target_node(target_lempos_seq[0], graph)
        
        if len(target_lempos_seq) > 1:
            u = tn
            target_nodes = [u]
            
            for target_lempos in target_lempos_seq[1:]:
                v = self._add_single_target_node(target_lempos, graph)
                target_nodes.append(v)
                graph.add_word_order_edge(u, v)
                u = v
        
            tn = graph.add_hyper_target_node(target_nodes)
            
        return tn

    def _get_source_node(self, i, j, source_nodes, graph):
        if j == i + 1:
            # source node is a singular node (unigram)
            sn = source_nodes[i]
        else:
            # multiple nodes (n-gram with n>1), so source node becomes a
            # new hyper node combining singular source nodes
            sn = graph.add_hyper_source_node(source_nodes[i:j])
        return sn

    def _add_single_target_node(self, lempos, graph):
        lemma, pos = lempos.rsplit(self.dictionary.delimiter, 1)
        return graph.add_target_node(lemma=lemma, pos=pos) 
        