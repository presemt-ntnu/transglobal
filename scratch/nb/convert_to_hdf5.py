#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
convert collection of samples in gzipped matrix market format to single hdf5 file
"""

# TODO: optimal compression and chunks



import glob
import codecs
import logging
import os

import numpy as np
import scipy.io
import h5py

from tg.utils import coo_matrix_to_hdf5

log = logging.getLogger(__name__)


def convert(tab_fname, samp_dir, hdf_fname, samp_fpat="de-sample-{0}.mtx.gz", mode="a",
            max_samples=None):
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
            lempos, _ = target_label.rsplit("/", 1)
            samp_fname = samp_dir + samp_fpat.format(samp_fid)
            
            try:
                m = scipy.io.mmread(samp_fname.encode("utf-8"))
            except IOError:
                log.error("no sample file " + samp_fname)
                continue
            
            if lempos not in samples:
                # name contains "/" as separator, which h5py interpretes as a subgroup
                group = samples.create_group(lempos)
                log.info("adding " + samp_fname)
                # using 8-bit int to save space, assuming that no lemma will
                # occur over 256 times in a single context
                coo_matrix_to_hdf5(m, group, data_dtype="=i1", compression='gzip')
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
    
    # de-en
    tab_fname = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/de/en/de-en_ambig.tab"
    samp_dir = "/Users/erwin/Projects/Transglobal/github/transglobal/_data/corpmod/en/samples/"
    hdf_fname = "en_samples.hdf5"
    convert(tab_fname, samp_dir, hdf_fname, 
            mode="w", 
            samp_fpat="en-sample-{0}.mtx.gz"
            # max_samples=10
            )
    