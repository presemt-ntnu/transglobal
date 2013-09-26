"""
summarize properties of evaluation data sets 
"""

import cPickle
from tg.config import config


xx2lang = dict(
    de = "German",
    en = "English",
    gr = "Greek",
    no = "Norwegian",
    )

def data_props(graphs_fname):
    graphs = cPickle.load(open(graphs_fname))
    n_tokens = 0
    n_sent = len(graphs)
    
    for graph in graphs:
        n_tokens += len(graph.source_lemmas())
        
    return n_tokens, n_sent
        
        
def all_data_props():        
    data_sets=config["eval"]["data_sets"] 
    format_str = "{:<32} & {:<8} & {:<8} & {:>8d} & {:>8d} & {:>8.2f} \\\\" 
    
    for data in data_sets:
        lang_pairs = config["eval"][data].keys() 
        for lang in lang_pairs:
            graphs_fname = config["eval"][data][lang]["graphs_fname"]
            n_tokens, n_sent = data_props(graphs_fname)
            label = "\\textsc{" + data + "-" + lang + "}"
            sl, tl = lang.split("-")
            mean = n_tokens / float(n_sent)
            print format_str.format(
                label, 
                xx2lang[sl], 
                xx2lang[tl], 
                n_tokens, 
                n_sent, 
                mean)
       
            
if __name__ == "__main__":
    all_data_props()