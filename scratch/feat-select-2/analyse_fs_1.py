import numpy as np
import matplotlib.pyplot as plt

from tg.config import config


def analyse_fs_1():
    plt.rcParams["axes.color_cycle"] = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'grey']
    
    results = np.load("_fs-1-acc.npy")
    
    plt.figure(figsize=(14,10))
               
    for i, measure in enumerate(("accuracy", "bleu")):
        plt.subplot(1, 2, i+1)
        for data in "metis", "presemt-dev":
            for lang in config["eval"][data].keys():
                source, target = lang.split("-")
                subset = results[(results["data"] == data) & 
                                 (results["source"] == source) &
                                 (results["target"] == target)]
                label = data + "_" + lang
                plt.plot(subset["alpha"], subset[measure] * 100, "o-", label=label)
                
        plt.xscale('log')
        plt.grid(axis="y", which="both")
        plt.xlabel("Alpha")
        if measure == "accuracy":
            plt.yticks(np.arange(25,75, 5))
            plt.ylim(25, 75)    
        else:
            plt.yticks(np.arange(0,30, 5))
            plt.ylim(0,30)    
        plt.ylabel(measure.capitalize() + " (%)")
        plt.legend(fontsize="small", ncol=3)
        
    plt.tight_layout()
    plt.show()
            
            
if __name__ == "__main__":
    analyse_fs_1()