"""
read dicts
"""

# TODO
# - store dict more efficiently

import configobj
import cPickle
import logging as log
from xml.etree import cElementTree as et



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