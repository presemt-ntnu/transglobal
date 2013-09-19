"""
approximate maximum score  
"""

import logging 
import xml.etree.cElementTree as et
from collections import Counter

from tg.graphproc import GraphProcess
from tg.scorer import Scorer
from tg.mteval import read_ref_trans_counts
from tg.ambig import AmbiguityMap

log = logging.getLogger(__name__)


# TODO
# - counts for MWUs can be obtained by counting n-grams in the references
# - store doc-id and seg-id during annotation?

class _UpperScorer(Scorer):
    
    def __init__(self, ref_fname, score_attr):
        Scorer.__init__(self, score_attr)
        self.ref_fname = ref_fname
        
    def __call__(self, obj, *args, **kwargs):
        # It is assumed that the order of documents (by docid) in the source
        # and reference is the same
        self.counts = iter(read_ref_trans_counts(self.ref_fname, flatten=True))        
        Scorer.__call__(self, obj, *args, **kwargs)
        
    def _score_translations(self, graph, u, seg_counts):
        edge_data = []
        counts = []
        
        for u, v, data in graph.trans_edges_iter(u):
            edge_data.append(data)
            target_lemma = graph.lemma(v).lower()
            lemma_count = seg_counts.get(target_lemma)
            counts.append(lemma_count)
            
        return edge_data, counts
        


class DictUpperScorer(_UpperScorer):
    """
    Approximation of the maximal scoring obtainable.
    
    For each segment (sentence), we look at its reference translation(s) and
    count the number of times that a target lemma occurs. Next, when scoring
    translations candidates, we pick the translations with the highest count
    in the reference translation.
    
    Remarks:
    - This is an approximation. It may not work well for high frequency words 
      such as determiners or pronouns.
    - It is assumed that the order of documents (by docid) in the source
      and reference is the same. Therefore calling it on a single graph 
      will not work as expected.
    - In case of ties (candidates with equal counts), the choice is arbitrary.
    - In case all candidates have zero counts, the choice is arbitary
      (does not matter for score anyway).
    - Multi-word units are not taken into account yet.
    """
    
    def __init__(self, ref_fname, score_attr="dup_score"):
        _UpperScorer.__init__(self, ref_fname, score_attr)
                
    def _single_run(self, graph):
        # called for optional debug logging
        GraphProcess._single_run(self, graph)
        seg_counts = self.counts.next()
        
        for u in graph.source_nodes_iter():
            self._add_normalized_scores(
                *self._score_translations(graph, u, seg_counts))

    
    
class ModelUpperScorer(_UpperScorer): 
    """
    Similar to DictUpperScore, but the upper score is added only if the there
    is a model for the source lempos according to ambuity map for the
    language pair, optionally filtered.
    
    Remarks:
    - This score can only be compared with model scores obtained with the 
      same ambig map and the same filter.
    """
    
    def __init__(self, ref_fname, ambig_fname, filter=None, 
                 score_attr="mup_score"):
        _UpperScorer.__init__(self, ref_fname, score_attr)
        self.ambig_map = AmbiguityMap(ambig_fname)
        self.filter = filter or (lambda graph, node : False)
        
    def _single_run(self, graph):
        # called for optional debug logging
        GraphProcess._single_run(self, graph)
        seg_counts = self.counts.next()
        
        # skip if source lempos is not in ambiguity map or is filtered
        for u in graph.source_nodes_iter():
            try:
                source_lempos = u" ".join(graph.node[u]["lex_lempos"])
            except:
                # source lempos was not in lexicon (e.g. punctuation)
                continue
            
            if source_lempos not in self.ambig_map:
                continue
            
            if self.filter(graph, u):
                continue
            
            self._add_normalized_scores(
                *self._score_translations(graph, u, seg_counts))
    