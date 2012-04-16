#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
convert collection of samples in gzipped matrix market format to single hdf5 file
"""

# TODO: optimal compression and chunks
# TODO: use other data type instead of int32 ?


import glob
import codecs
import logging
import os

import numpy as np
import scipy.io
import h5py

log = logging.getLogger(__name__)

def convert(tab_fname, samp_dir, hdf_fname, samp_fpat="de-sample-{0}.mtx.gz", mode="a",
            max_samples=None,):
    hdfile = h5py.File(hdf_fname, mode)
    samp_count = 0
    
    log.info("opening " + hdf_fname)
    try:
        samples = hdfile["samples"]
    except KeyError:
        samples = hdfile.create_group("samples")
     
    for line in codecs.open(tab_fname, encoding="utf8"):
        if max_samples:
            samp_count += 1
            if samp_count > max_samples: 
                break
        
        _, _, target_label, samp_fid, new = line.rstrip().split("\t")
        
        # only if this is the first occurrence of this sample
        if new == "1":
            # remove deWac POS tag
            lempos, _ = target_label.rsplit("/",1)
            samp_fname = samp_dir + samp_fpat.format(samp_fid)
            
            try:
                m = scipy.io.mmread(samp_fname.encode("utf-8"))
            except IOError:
                log.error("no sample file " + samp_fname)
                continue
            
            # hdf group name can not contain forward slash
            name = lempos.replace("/", "_")
            
            if name not in samples:
                log.info("adding " + samp_fname)
                coo_data = np.array([m.row, m.col, m.data])
                sample = samples.create_dataset(name, data=coo_data, compression='gzip')
                sample.attrs["shape"] = m.shape
            else:
                log.info("skipping " + samp_fname + " because already present")
            
    log.info("closing " + hdf_fname)
    hdfile.close()          
    
    
if __name__ == "__main__":
    from tg.utils import set_default_log
    set_default_log(level=logging.INFO)
        
    # en-de
    tab_fname = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/en/de/en-de_ambig.tab"
    samp_dir = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/de/samples/"
    hdf_fname = "de_samples.hdf5"
    convert(tab_fname, samp_dir, hdf_fname, 
            mode="w", 
            # max_samples=10
            )
    