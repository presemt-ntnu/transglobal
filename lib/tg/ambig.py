"""
lempos ambiguity information
"""

import codecs
import cPickle
import logging

log = logging.getLogger(__name__)

from tg.config import config



class AmbiguityMap():
    """
    Holds lempos ambiguity information for the samples of a given language pair
    """

    def __init__(self, ambig_fname, lang_pair = None, subset = None, graphs_fname = None):
        """
        @param lang_pair: Read ambiguities from the file configured for the given language pair.
        @param fn: Read ambiguities from the specified file.
        """
        if ambig_fname:
            self.ambig_fname = ambig_fname
        elif lang_pair:
            self.ambig_fname = config["sample"][lang_pair]["ambig_fn"]
        else:
            raise ValueError
        
        if graphs_fname:
            subset = self.extract_source_lempos_subset(graphs_fname)

        self.source_target_map = self.read_ambig_file(self.ambig_fname, subset = subset)

    @staticmethod
    def read_ambig_file(ambig_fname, subset = None):
        source_target_map = {} 
        log.info("reading lempos ambiguity from file " + ambig_fname)        

        for line in codecs.open(ambig_fname, encoding="utf8"):
            if line.strip():
                source_label, target_label = line.rstrip().split("\t")[1:3]
                # strip corpus POS tag
                source_lempos = source_label.rsplit("/", 1)[0]
                target_lempos = target_label.rsplit("/", 1)[0]
    
                if subset and source_lempos not in subset:
                    continue
    
                source_target_map.setdefault(source_lempos,[]).append(target_lempos)

        return source_target_map
    
    @staticmethod
    def extract_source_lempos_subset(graphs_fname):
        """
        extract all required source lempos from pickled graphs,
        where POS tag is the *lexicon* POS tag
        """
        log.info("extracting source lempos subset from file " + graphs_fname)
        subset = set()
        
        for graph in cPickle.load(open(graphs_fname)):
            for _,node_attr in graph.source_nodes_iter(data=True, 
                                                       ordered=True):
                try:
                    subset.add(" ".join(node_attr["lex_lempos"]))
                except KeyError:
                    # not found in lexicon
                    pass
                
        return subset

    def __getitem__(self, item):
        return self.source_target_map[item]
    
    def __len__(self):
        return len(self.source_target_map)

    def source_target_pair_iter(self):
        """
        @return Iterator of all pairs of source and lemmas
        """
        return ((sl, tl) for sl, tl_list in self.source_target_map.iteritems() for tl in tl_list)

    def source_iter(self):
        """
        @return Iterator of all source lemmas
        """
        return self.source_target_map.iterkeys()
    
    def target_iter(self):
        """
        @return Iterator of all target lemmas
        """
        return (tl for tl_list in self.source_target_map.itervalues() for tl in tl_list)