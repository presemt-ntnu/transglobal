"""
parse NIST and BLEU scores from output of mteval scoring script
"""
  
      
from collections import OrderedDict, Counter
import codecs
import logging
import re
import subprocess
import sys
import xml.etree.cElementTree as et 

import numpy as np

from tg.config import config

log = logging.getLogger(__name__)

  
    
def mteval(ref_fname, src_fname, tst_fname, outf=sys.stdout,
           options=config["eval"]["mteval_opts"]):
    """
    Simple wrapper of NIST mteval-v13a.pl script
    
    Parameters
    ----------
    ref_fname: str
        file containing the reference translations for the documents
        to be evaluated
    src_fname: str
        file containing the source documents for which translations 
        are to be evaluated 
    tst_fname: str
        file containing the translations to be evaluated
    outf: file
        output file
    options: str
        eval script command line options
    
    Returns
    -------
    out, err: str, str
        strings received from standard output and error
    """
    command = '{0} {1} -r "{2}" -s "{3}" -t "{4}" {5}'.format(
        config["eval"]["perl_fname"],
        config["eval"]["mteval_fname"],
        ref_fname,
        src_fname,
        tst_fname,
        options or "")
        
    # create pipe to eval script
    log.debug("Calling evaluation script as as: " + command)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
      
    # send text and retrieve tagger output 
    out, err = proc.communicate()    
    
    log.debug(u"mteval standard output:\n" + out.decode("utf-8"))
    
    if err:
        log.error(u"mteval standard error:\n" + err.decode("utf-8"))
    
    if isinstance(outf, basestring):
        outf = open(outf, "w")
        close = True
    else:
        close = False
        
    log.info("writing mteval output to " + outf.name)
    outf.write(out)
    
    if close:
        outf.close()
    
    return out, err



def read_ref_trans(ref_fname, flatten=False):
    """
    Read reference translations from file in mteval xml format
    
    Parameters
    ----------
    ref_fname: file or filename
        reference translations in mteval xml format
        
    Returns
    -------
    refset: OrderedDict
        ordered dictionary mapping doc id to another ordered dict which
        maps seg id to a list of reference translations
    flatten: bool
        flatten refset by removing ordered dicts for documents and segments,
        returning only an ordered list of translations for each segment 
     
    Examples
    --------
    Get reference translations for doc with id "test" and segment with id "1":
    
    >>> refset["test"]["1"]    
    ['the European Union have react already .', 
     'There have be reaction on the side of the European Union .', 
     'the European union have already react .', 
     'the European Union have already react .', 
     'the European Union have respond already .']
    """
    refset = OrderedDict()
    
    for event, elem in et.iterparse(ref_fname, events=("start", "end")):
        if event == "start" and elem.tag == "doc":
            doc_id = elem.attrib["docid"]
            try:
                doc = refset[doc_id]
            except KeyError:
                doc = refset[doc_id] = OrderedDict()
        elif event == "end" and elem.tag == "seg":
            seg_id = elem.get("id")
            try:
                doc[seg_id].append(elem.text)
            except KeyError:
                doc[seg_id] = [elem.text]

    if flatten:
        refset = _flatten_ref_trans(refset)
        
    return refset



def read_ref_trans_counts(ref_fname, tok_func=lambda s: s.lower().split(), 
                          flatten=False):
    """
    Read reference translations counts from file in mteval xml format
    
    Parameters
    ----------
    ref_fname: file or filename
        reference translations in mteval xml format
    tok_func: func
        tokenisation function taking a sentence string as input and returning a 
        list of tokens 
    flatten: bool
        flatten refset by removing ordered dicts for documents and segments,
        returning only an ordered list of tokens counts for each segment 
        
    Returns
    -------
    refset: OrderedDict
        ordered dictionary mapping doc id to another ordered dict which
        maps seg id to Counter instances, each one holding token counts over all
        reference translations of a segment
     
    Examples
    --------
    Get tokens counts for document with id "test" and segment with id "1":
    
    >>> refset["test"]["1"]
    Counter({'the': 6, 'union': 5, '.': 5, 'have': 5, 'european': 5, 
    'already': 4, 'react': 3, 'respond': 1, 'be': 1, 'reaction': 1, 'of': 1, 
    'there': 1, 'on': 1, 'side': 1})   
    
    Get ordered list of segment ids in document with id "test":

    >>> refset["test"].keys()
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', 
    '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', 
    '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', 
    '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']
    
    Get count of token "european" in last segment of doc:
    
    >>> refset["test"].values()[-1]["european"]
    4
    """
    refset = OrderedDict()
    
    for event, elem in et.iterparse(ref_fname, events=("start", "end")):
        if event == "start" and elem.tag == "doc":
            doc_id = elem.attrib["docid"]
            try:
                doc = refset[doc_id]
            except KeyError:
                doc = refset[doc_id] = OrderedDict()
        elif event == "end" and elem.tag == "seg":
            seg_id = elem.get("id")
            # tokens are lower-cased
            tokens = tok_func(elem.text)
            try:
                doc[seg_id].update(tokens)
            except KeyError:
                doc[seg_id] = Counter(tokens)
                
    if flatten:
        refset = _flatten_ref_trans(refset)

    return refset


def _flatten_ref_trans(refset):
    """
    Flatten reference translations or their tokens counts
    """
    return [ seg
             for doc in refset.values()
             for seg in doc.values() ]
            


def parse_total_scores(file):
    """
    Get NIST and BLEU scores from score file
    
    Parameters
    ----------
    file: file or filename
        Output from mteval-v13a.pl scoring script
        produced with flag -d2 or -d3 for detailed output
        
    Returns
    -------
    (system, NIST, BLEU): tuple
        Tuple of system identifier and scores
    """
    if isinstance(file, basestring):
        file = open(file)
        
    total_score_pat = re.compile(
        'NIST score = ([0-9.]+)\s+BLEU score = ([0-9.]+)\s+for system "(.*)"')
    
    for line in file:
        try:
            nist, bleu, system = total_score_pat.match(line).groups()
        except AttributeError:
            pass
        else:
            return system, float(nist), float(bleu)
        
        
def parse_document_scores(file):
    """
    Get NIST and BLEU scores for documents from score file
    
    Parameters
    ----------
    file: file or filename
        Output from mteval-v13a.pl scoring script
        produced with flag -d2 or -d3 for detailed output 
    Returns
    -------
    scores: numpy.ndarray
        array with named fields "system", "document", "segments", "words", 
        "NIST" and "BLEU"
    """
    if isinstance(file, basestring):
        file = open(file)
        
    src_set_pat = re.compile(    
        '\s+src set ".*" \(([0-9.]+) docs, \d+ segs\)')        

    doc_score_pat = re.compile(
        '(NIST|BLEU) score using\s+\d+-grams = ([0-9.]+) for system "(.*)" '
        'on document "(.*)" \((\d+) segments, (\d+) words\)')  
    
    descriptor = {"names": ("system", "document", "segments", "words", 
                            "NIST", "BLEU", ), 
                  "formats": ("S64", "S64", "i4", "i4", "f4", "f4")}  
    
    doc_score_count = 0
    
    for line in file:
        try:
            n_docs = int(src_set_pat.match(line).group(1))
        except AttributeError:
            pass
        else:
            scores = np.zeros(n_docs, dtype=descriptor)
        
        try:
            metric, score, system, document, segments, words =\
            doc_score_pat.match(line).groups()
        except AttributeError:
            pass
        else:
            row = scores[doc_score_count % n_docs]
            row["system"] = system
            row["document"] = document
            row["segments"] = int(segments)
            row["words"] = int(words)
            row[metric] = float(score)
            doc_score_count += 1
            
    return scores


def parse_segment_scores(file):
    """
    Get NIST and BLEU scores for segments (sentences) from score file
    
    Parameters
    ----------
    file: file or filename
        Output from mteval-v13a.pl scoring script
        produced with flag -d2 or -d3 for detailed output
        
    Returns
    -------
    scores: numpy.ndarray
        array with named fields "system", "document", "segment", "words", 
        "NIST" and "BLEU"
    """
    if isinstance(file, basestring):
        file = open(file)
        
    src_set_pat = re.compile(    
        '\s+src set ".*" \(\d+ docs, ([0-9.]+) segs\)')        
    
    sent_score_pat = re.compile(
        '  (NIST|BLEU) score using \d+-grams = ([0-9.]+) for system "(.*)" '
        'on segment (\d+) of document "(.*)" \((\d+) words\)')
    
    descriptor = {"names": ("system", "document", "segment", "words", 
                            "NIST", "BLEU", ), 
                  "formats": ("S64", "S64", "S64", "i4", "f4", "f4")}  
    
    seg_score_count = 0
    
    for line in file:
        try:
            n_seg = int(src_set_pat.match(line).group(1))
        except AttributeError:
            pass
        else:
            scores = np.zeros(n_seg, dtype=descriptor)
        
        try:
            metric, score, system, segment, document, words =\
            sent_score_pat.match(line).groups()
        except AttributeError:
            pass
        else:
            row = scores[seg_score_count % n_seg]
            row["system"] = system
            row["document"] = document
            row["segment"] = segment
            row["words"] = int(words)
            row[metric] = float(score)
            seg_score_count += 1 
            
    return scores

