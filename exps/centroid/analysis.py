#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
generate graphs combining nc-1 and bounds scores
"""



import numpy as np
import matplotlib.pyplot as plt

from tg.config import config

nb = np.load("_nc-2.npy")
bounds = np.load("../bounds/_bounds.npy")


for measure in "accuracy", "nist", "bleu", :
    fig = plt.figure(figsize=(16,9))
    i = 0
    
    for data in config["eval"]["data_sets"]:
        for lang_pair in config["eval"][data].keys():
            source, target = lang_pair.split("-")
            subset = nb[(nb["data"] == data) & 
                        (nb["source"] == source) & 
                        (nb["target"] == target)]
            
            if not subset.size:
                continue
                    
            i += 1
            
            scores = subset[measure]
            labels = ["vect={} metric={}".format(r["vect_score_attr"],
                                                 r["metric"]) 
                      for r in subset]
            
            if measure != "accuracy":
                # No bounds for accuracy, because that depends on classifier.
                # Could be calculated for NC though, in principle.
                subset = bounds[(bounds["data"] == data) & 
                                (bounds["source"] == source) & 
                                (bounds["target"] == target)]
                scores = np.hstack([scores, subset[measure]])
                labels += [r["score_attr"] for r in subset]                        
            
            y_pos = np.arange(len(labels))
            ax = plt.subplot(3, 4, i)           

            if measure == "bleu":
                scores *= 100
                plt.xlim(xmax=40.0)
            elif measure == "nist":
                plt.xlim(xmax=9.0)                
            else:
                scores *= 100
                plt.xlim(xmax=60)                
                
            handles = plt.barh(y_pos, scores, align='center', height=0.5, 
                     color=("gray", "blue", "yellow", "green", "red", 
                            "orange", "purple", "white"),
                     alpha=0.8)
            
            for label, bar in zip(labels, handles):
                if "cosine" in label:
                    bar.set_hatch("//")
                
            scores = ["{:.2f}".format(s) for s in scores]
            plt.yticks(y_pos, scores)
            plt.xlabel(measure.capitalize())
            plt.grid(axis="x")
            plt.title("{} {}-{}".format(data.upper(), 
                                        source.capitalize(), 
                                        target.capitalize()))
            
    plt.figlegend(reversed(handles), reversed(labels), "lower center")
    plt.tight_layout()
    #plt.show()
    plt.savefig("_barchart_" + measure + ".pdf", format="pdf")