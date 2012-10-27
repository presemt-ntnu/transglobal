"""
postprocessing of experimental data
"""

import os

from tg.config import config
from tg.draw import Draw
from tg.format import TextFormat, MtevalFormat
from tg.mteval import mteval, parse_total_scores


def postprocess(exp_name, data_set, lang_pair, graph_list, best_score_attr,
                base_score_attrs=[],
                sysid=None,
                base_fname=None,
                base_dir="./",
                out_dir=None, 
                draw=False,
                text=False):
    
    if not sysid:
        sysid = exp_name
        
    if not base_fname:
        base_fname = "_".join((exp_name, data_set, lang_pair))
    
    if not out_dir: 
        out_dir = os.path.join(base_dir, "_" + base_fname)
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    # draw graphs
    if draw:
        draw = Draw()
        draw(graph_list, out_format="pdf", best_score_attr=best_score_attr,
             base_score_attrs=base_score_attrs, out_dir=out_dir)
    
    # write translation output in plain text format
    if text:
        text_format = TextFormat(score_attr=best_score_attr)
        text_format(graph_list)
        text_format.write(os.path.join(out_dir, base_fname + ".txt"))
    
    # write translation output in Mteval format
    trglang = lang_pair.split("-")[1]
    mte_format = MtevalFormat(
        config["eval"][data_set][lang_pair]["src_fname"],
        trglang=trglang, 
        sysid=sysid,
        score_attr=best_score_attr)
    mte_format(graph_list)
    tst_fname = os.path.join(out_dir, base_fname + ".tst")
    mte_format.write(tst_fname)
    
    # calculate BLEU and NIST scores using mteval script
    scores_fname = os.path.join(out_dir, base_fname + ".scores")
    mteval(config["eval"][data_set][lang_pair]["lemma_ref_fname"],
           config["eval"][data_set][lang_pair]["src_fname"],
           tst_fname,
           scores_fname)
    scores = parse_total_scores(scores_fname)
    
    return scores[1:]

