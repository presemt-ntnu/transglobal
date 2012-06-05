"""
postprocessing of experimental data
"""

import os

from tg.config import config
from tg.draw import Draw
from tg.format import TextFormat, MtevalFormat
from tg.eval import mteval, get_scores, mteval_lang


def postprocess(data_set, lang_pair, out_dir, exp_name, graph_list,
                score_attr="freq_score", 
                sysid="most frequent translation", 
                draw=True):
    # draw graphs
    if draw:
        draw = Draw()
        draw(graph_list, out_format="pdf", best_score_attr=score_attr,
             out_dir=out_dir)
    
    # write translation output in plain text format
    text_format = TextFormat(score_attr=score_attr)
    text_format(graph_list)
    text_format.write(os.path.join(out_dir, exp_name + ".txt"))
    
    # write translation output in Mteval format
    trglang = lang_pair.split("-")[1]
    mte_format = MtevalFormat(
        config["eval"][data_set][lang_pair]["src_fname"],
        trglang=trglang, 
        sysid=sysid,
        score_attr=score_attr)
    mte_format(graph_list)
    tst_fname = os.path.join(out_dir, exp_name + ".tst")
    mte_format.write(tst_fname)
    
    # calculate BLEU and NIST scores using mteval script
    scores_fname = os.path.join(out_dir, exp_name + ".scores")
    mteval(config["eval"][data_set][lang_pair]["lemma_ref_fname"],
           config["eval"][data_set][lang_pair]["src_fname"],
           tst_fname,
           scores_fname)
    scores = get_scores(scores_fname)
    
    return scores