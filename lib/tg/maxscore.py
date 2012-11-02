"""
approximate maximum score  
"""

import logging 
import xml.etree.cElementTree as et
from collections import Counter

from tg.graphproc import GraphProcess

log = logging.getLogger(__name__)


# TODO
# - counts for MWUs can be obtained by counting n-grams in the references
# - store doc-id and seg-id during annotation?


class MaxScore(GraphProcess):
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
      and reference is the same    
    - In case of ties (candidates with equal counts), the choice is arbitrary.
    - In case all candidates have zero conts, the choice is arbitary
      (does not matter for score anyway).
    - Multi-word units are not taken into account yet.
    - Scores are not normalized between zero and one.
    """
    
    def __init__(self, ref_fname, score_attr="max_score"):
        GraphProcess.__init__(self)
        self.score_attr = score_attr
        self.counts = self._get_counts(ref_fname)
        
    def _get_counts(self, ref_fname):
        """
        For each segment, create a bag (multi-set) from the tokens of its
        reference translation(s)
        """
        counts = {}
        doc_order = []
        
        for event, elem in et.iterparse(ref_fname, events=("start", "end")):
            if event == "start" and elem.tag == "doc":
                doc_id = elem.attrib["docid"]
                if doc_id not in doc_order:
                    doc_order.append(doc_id)
                try:
                    doc_counts = counts[doc_id]
                except KeyError:
                    doc_counts = counts[doc_id] = []
                seg_num = 0
            elif event == "end" and elem.tag == "seg":
                # tokens are lower-cased
                tokens = elem.text.lower().split()
                try:
                    seg_counts = doc_counts[seg_num]
                except IndexError:
                    seg_counts = Counter()
                    # seg_counts are kept in an ordered list rather than a
                    # dict keyed on seg_id, because the corresponding graphs
                    # may not have a seg:id attribute
                    doc_counts.append(seg_counts)
                seg_counts.update(tokens)
                seg_num += 1
                
        # It is assumed that the order of documents (by docid) in the source
        # and reference is the same
        ordered_counts = []
        for doc_id in doc_order:
            ordered_counts.extend(counts[doc_id])

        return ordered_counts
    
    def _batch_run(self, obj_list, *args, **kwargs):
        self.seg_num = 0
        GraphProcess._batch_run(self, obj_list, *args, **kwargs)

                
    def _single_run(self, graph):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        seg_counts = self.counts[self.seg_num]
        
        for u in graph.source_nodes_iter():
            for u,v,data in graph.trans_edges_iter(u):
                target_lemma = graph.lemma(v).lower()
                count = seg_counts.get(target_lemma, 0)
                data[self.score_attr] = count 
                
        self.seg_num += 1