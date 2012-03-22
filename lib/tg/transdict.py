"""
read dicts
"""

# TODO
# - store dict more efficiently

import cPickle
import logging
import sys
from xml.etree import cElementTree as et

import configobj


log = logging.getLogger(__name__)


class TransDict(dict):
    """ 
    dictionary object that allows lookup of translation candidates on the
    basis of sl lemma and (optionally) POS tag
    """    
        
    delimiter = "/"    

    @staticmethod
    def load(pkl_fname):
        log.info("loading translation dictionary from " + pkl_fname)
        return cPickle.load(open(pkl_fname))


    @classmethod
    def from_xml(clas, dict_fname, reverse=False):
        # need to read in the whole dict, because same lemma may have multiple
        # entries and order is not guaranteed
        log.info("reading translation dictionary from " + dict_fname)
        
        sl_lem = tl_lem = ""
        sl_lempos = tl_lempos = ""
        trans_dict = clas()
        format_error = False
        entry_id = None
        
        # XML parsing is somewhat fuzzy, because lexicon is likely to contain
        # format errors
        for event, elem in et.iterparse(dict_fname):
            if elem.tag == "slLemma":
                # check that text is not None and contains no spaces
                if not elem.text or len(elem.text.split()) != 1:
                    format_error = True
                else:
                    sl_lem += elem.text.strip() + " "
                    sl_lempos += ( elem.text.strip() + 
                                   clas.delimiter + 
                                   elem.get("tag") + 
                                   " " )
            elif elem.tag == "tlLemma":
                if not elem.text or len(elem.text.split()) != 1:
                    format_error = True
                else:
                    tl_lem += elem.text.strip() + " "
                    tl_lempos += ( elem.text.strip() + 
                                   clas.delimiter + 
                                   elem.get("tag") + 
                                   " " )
            elif elem.tag == "entry":
                entry_id = elem.get("id")
                    
                if not format_error:
                    sl_lem = sl_lem.rstrip()
                    tl_lem = tl_lem.rstrip()
                    sl_lempos = sl_lempos.rstrip()
                    tl_lempos = tl_lempos.rstrip()
    
                    # store mapping from both sl lemma and sl lempos to tl lempos
                    if not reverse:
                        trans_dict.setdefault(sl_lem, set()).add(tl_lempos)
                        trans_dict.setdefault(sl_lempos, set()).add(tl_lempos)
                    else:
                        trans_dict.setdefault(tl_lem, set()).add(sl_lempos)
                        trans_dict.setdefault(tl_lempos, set()).add(sl_lempos)
                else:
                    # some format error, e.g. slLemma has no text
                    log.error("skiping ill-formed lexicon entry "
                              "with id {0}".format(entry_id))
                    format_error = False
    
                sl_lem = tl_lem = ""
                sl_lempos = tl_lempos = ""
                
        return trans_dict
    
        
    def dump(self, pkl_fname):
        log.info("dumping translation dictionary to " + pkl_fname)
        with open(pkl_fname, "wb") as inf:
            cPickle.dump(self, inf)
    
    

class DictAdaptor:
    
    def __init__(self, dictionary, posmap):
        if isinstance(dictionary, basestring):
            self.dict = TransDict.load(dictionary)
        else:
            self.dict = dictionary
            
        if isinstance(posmap, basestring):
            log.info("loading POS mapping from " + posmap)
            self.posmap = configobj.ConfigObj(posmap)
        else:
            self.posmap = posmap
            
        self.delimiter = self.dict.delimiter

            
    def __getitem__(self, key):
        lemmas = ""
        lempos = ""
        
        for pair in key.split():
            try:
                lemma, tag = pair.rsplit(self.delimiter, 1)
            except ValueError:
                # no tag specified
                lemma = pair
                mapped_tag = ""
            else:
                # if tag can not be mapped, then mapped_tag is empty,
                # thus lempos will never match
                mapped_tag = self.posmap.get(tag, "")                
                
            lemmas += lemma + " "
            lempos += lemma + self.delimiter + mapped_tag + " "
            
        try:
            return self.dict[lempos[:-1]]
        except KeyError:
            # no match on tag(s),
            # try result for lemma(s) only
            return self.dict[lemmas[:-1]]
            
            
    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default
        
        
        
#-----------------------------------------------------------------------------        
# Utility functions
#-----------------------------------------------------------------------------

    
def ambig_dist(trans_dict, with_lemma=True, with_lempos=True,
               with_single_word=True, with_multi_word=True, max_trans=100,):
    """
    Takes a TransDict objcet and the distribution of the translation ambiguity. 
    Returns a list where the first item represents the number of entries with
    zero translations, the second entry the number of entries with one
    translation, and so on until max_trans
    """
    # TODO: filter on certain POS tags
    dist = (max_trans + 1) * [0]
    
    for entry, values in trans_dict.iteritems():
        if trans_dict.delimiter in entry:
            if not with_lempos:
                continue
        else:
            if not with_lemma:
                continue
            
        if " " in entry:
            if not with_multi_word:
                continue
        else:
            if not with_single_word:
                continue
            
        dist[len(values)] += 1
        
    return dist


def ambig_dist_report(dist, outf=sys.stdout):
    """
    Report statistics on distribution of translation ambiguity,
    where 'dist' results from calling function 'ambig_dist'
    """
    total = sum(dist)
    outf.write("total number of entries: {0}\n".format(total))
    total_ambig = sum(dist[2:])
    outf.write("total number of ambiguous entries: {0} ({1:.2f}%)\n".format(
        total_ambig,
        total_ambig/float(total) * 100)) 
    outf.write("total number of non-ambiguous entries: {0} ({1:.2f}%)\n".format(
        total - total_ambig,
        (total - total_ambig)/float(total) * 100))
    
    # first one we look at is n + 2 = 0 + 2 = 2
    weighted_sum = sum([(n + 2) * c 
                        for n, c in enumerate(dist[2:])])
    av_ambig = weighted_sum / float(total_ambig)
    outf.write("average ambiguity (over ambiguous entries only): "
               "{0:.2f} translations )\n\n".format(av_ambig))
   
    outf.write(" n:          count:        %:\n")    
    for n, count in enumerate(dist):
        outf.write("{0:3d}    {1:12d}    {2:6.2f}\n".format(
            n,
            count,
            count/float(total) * 100))
         