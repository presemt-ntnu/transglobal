"""
lempos ambiguity information
"""

import codecs
import cPickle
import logging

from tg.config import config

log = logging.getLogger(__name__)



class AmbiguityMap():
    """
    Holds lempos ambiguity information for the samples of a given language pair
    
    Parameters
    ----------
    ambig_fname: str, optional
        Name of file containing ambiguity information
    lang_pair: str, optional
        Language pair used to retrieve ambig_fname from config
    subset: set, optional
        Restrict ambiguity map to given set of source lempos strings
    graphs: str or list, optional
        Name of file containing pickled graphs or list of TransGraph instances.
        Restrict ambiguity map to source lempos in these graphs.
        
    Notes
    -----
    Either ambig_fname or lang_pair must be given.        
    """

    def __init__(self, ambig_fname=None, lang_pair = None, subset = None, graphs = None):
        if ambig_fname:
            self.ambig_fname = ambig_fname
        elif lang_pair:
            self.ambig_fname = config["sample"][lang_pair]["ambig_fn"]
        else:
            raise ValueError
        
        if graphs:
            subset = self.extract_source_lempos_subset(graphs)

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
    
                source_target_map.setdefault(source_lempos, []).append(target_lempos)

        return source_target_map
    
    @staticmethod
    def extract_source_lempos_subset(graphs):
        """
        Extract all required source lempos pickled graphs.
        
        Parameters
        ----------
        graphs: str or list, optional
            Name of file containing pickled graphs or list of TransGraph instances
            
        Returns
        -------
        subset: set
            Subset of all source lempos strings occuring in graphs
            
        Notes
        -----
        The relevant POS tag in the lempos is the *lexicon* POS tag
        (lex_lempos attribute of nodes).
        """        
        if isinstance(graphs, basestring):
            log.info("extracting source lempos subset from file " + graphs)
            graphs = cPickle.load(open(graphs))

        subset = set()            
        
        for graph in graphs:
            for _, node_attr in graph.source_nodes_iter(data=True, 
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
    
    def __contains__(self, item):
        return item in self.source_target_map
    
    def __iter__(self):
        """
        Iterator of all pairs of source lemmas and their corresponding list of target lemmas
        """
        return self.source_target_map.iteritems()

    def source_target_pair_iter(self):
        """
        Iterator of all pairs of source and lemmas
        """
        return ((sl, tl) for sl, tl_list in self.source_target_map.iteritems() for tl in tl_list)

    def source_iter(self):
        """
        Iterator of all source lemmas
        """
        return self.source_target_map.iterkeys()
    
    def target_iter(self):
        """
        Iterator of all target lemmas
        """
        return (tl for tl_list in self.source_target_map.itervalues() for tl in tl_list)
    