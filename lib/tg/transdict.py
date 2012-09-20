"""
read dicts
"""


import collections
import cPickle
import logging
import sys
from xml.etree import cElementTree as et

import configobj


log = logging.getLogger(__name__)


class TransDict(object):
    """ 
    dictionary object that allows lookup of translation candidates on the
    basis of sl lemma and (optionally) POS tag
    """    
    
    # delimiter between lemma and POS tag in lempos string 
    delimiter = "/"
    
    # replacement for delimiter if it occurs in original POS tag
    replacement = "|"
    
    def __init__(self, pos_map=None):
        self._lempos_dict = {}   
        self._lemma_dict = {}
        
        if isinstance(pos_map, basestring):
            log.info("loading POS mapping from " + pos_map)
            self.pos_map = configobj.ConfigObj(pos_map)
        else:
            self.pos_map = pos_map   

    def lookup_lempos(self, lempos):
        """
        lookup lempos combination and return a pair consisting of lempos
        string and a tuple of possible translations
        """
        if self.pos_map:
            lempos = self._map_pos(lempos)
            
        return lempos, self._lempos_dict[lempos]
    
    def lookup_lemma(self, lemma):
        """
        lookup lemma, get the corresponding lempos entries, and return an
        iterator over lookup for all items (see lookup_lempos)
        """
        return ( (lempos, self._lempos_dict[lempos])
                 for lempos in self._lemma_dict[lemma] )

    def __getitem__(self, key):
        """
        lookup lemma or lempos, get matching lempos entries, and return an
        iterator over lookup for all items (see lookup_lempos)
        """
        try:
            return iter([self.lookup_lempos(key)])
        except (KeyError, ValueError):
            # lempos not found or lempos ill-formed,
            # fall back to lemma only
            return self.lookup_lemma(self._strip_pos(key))
        
    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default
    
    def lempos_iteritems(self):
        return self._lempos_dict.iteritems()
    
    def lemma_iteritems(self):
        return ( self.lookup_lemma[lempos]
                 for lempos in self._lemma_dict.itervalues() )

    @staticmethod
    def load(pkl_fname):
        log.info("loading translation dictionary from " + pkl_fname)
        return cPickle.load(open(pkl_fname))

    @classmethod
    def from_xml(cls, dict_fname, reverse=False, pos_map=None):
        """
        create TransDict object from parsing a lexicon in Presemt XML format
        
        If reverse is True, the dictionary's direction is reversed.
        
        Optional argument pos_map must be a configobj or other dict-like
        object (or a config filename) defining a mapping from POS tags used
        by the tagger to POS tags used in the lexicon.
        """
        log.info("reading translation dictionary from " + dict_fname)
        
        # temporary dict mapping lempos to set of translations 
        lempos_dict = collections.defaultdict(set)
        # temporary dict mapping lemma to set of source lempos
        lemma_dict = collections.defaultdict(set)        
        
        # XML parsing is somewhat fuzzy, because lexicon is likely to contain
        # format errors
        for event, elem in et.iterparse(dict_fname, events=("start", "end")):
            if event == "start":
                if  elem.tag == "entry":
                    sl_lem = tl_lem = ""
                    sl_lempos = tl_lempos = ""
                    format_error = False
            # event == "end" for all cases below
            elif elem.tag == "slLemma":
                if cls._is_valid(elem):
                    sl_lem += elem.text.strip() + " "
                    sl_lempos += cls._lempos(elem) + " "
                else:
                    format_error = True
            elif elem.tag == "tlLemma":
                if cls._is_valid(elem):
                    tl_lem += elem.text.strip() + " "
                    tl_lempos += cls._lempos(elem) + " "
                else:
                    format_error = True
            elif elem.tag == "entry":
                if format_error:
                    # some format error, e.g. slLemma has no text
                    log.error("skiping ill-formed lexicon entry "
                              "with id {0}".format(elem.get("id")))
                    continue
                
                sl_lem = sl_lem.rstrip()
                tl_lem = tl_lem.rstrip()
                sl_lempos = sl_lempos.rstrip()
                tl_lempos = tl_lempos.rstrip()

                if not reverse:
                    lempos_dict[sl_lempos].add(tl_lempos)
                    lemma_dict[sl_lem].add(sl_lempos)
                else:
                    lempos_dict[tl_lempos].add(sl_lempos)
                    lemma_dict[tl_lem].add(tl_lempos)
                
        trans_dict = cls(pos_map=pos_map)                
        # convert default dicts to normal dicts and
        # convert values from sets to tuples to decrease storage space
        trans_dict._lempos_dict = dict( (lempos, tuple(translations))
                                        for lempos, translations in lempos_dict.iteritems() )
        trans_dict._lemma_dict = dict( (lemma, tuple(lempos_keys))
                                        for lemma, lempos_keys in lemma_dict.iteritems() )
        
        return trans_dict
    
    def dump(self, pkl_fname):
        log.info("dumping translation dictionary to " + pkl_fname)
        with open(pkl_fname, "wb") as inf:
            cPickle.dump(self, inf)
            
    # support methods
    
    @classmethod    
    def _lempos(cls, elem):
        """
        create elempos string
        """
        return ( elem.text.strip() + 
                 cls.delimiter + 
                 elem.get("tag").replace(cls.delimiter, cls.replacement) ) 
    
    @classmethod    
    def _is_valid(cls, elem):
        """
        check if format of slLemma or tlLemma element is valid
        """
        # text is not None? 
        if not elem.text:
            return False
        # text contains no whitespacs?
        if len(elem.text.split()) != 1:
            return False
        # no proper tag?
        if elem.get("tag", "").strip() == "":
            return False
        return True
            
    def _strip_pos(self, lempos):
        """
        strip pos tags from lempos string
        """
        return " ".join( pair.rsplit(self.delimiter, 1)[0]
                         for pair in lempos.split() )
            
    def _map_pos(self, lempos):
        """
        map all pos tags in lempos
        """
        mapped_lempos = ""
        
        # map pos tag
        for pair in lempos.split():
            lemma, pos = pair.rsplit(self.delimiter, 1)
            mapped_lempos += ( lemma + 
                               self.delimiter + 
                               self.pos_map.get(pos, "") + 
                               " " )
            
        return mapped_lempos[:-1]
    


class TransDictGreek(TransDict):
    """
    A hack that permits partial matching of Greek POS tags.
    
    The Greek tagger delivers tags such as:
    AsPpSp
    AdXxBa
    CjCo
    NoCmFeSgAc
    NoCmNeSgAc
    
    The list of possible tags is very long.
    
    The first two letters form the POS tag, the rest are features. However,
    sometimes features are relevant for mapping. For example, NoCm is a
    common noun (with tag "nocm" in the lexicon) and NoPr is a proper noun
    (with tag "nopr" in the lexicon). So simply splitting off the initial two
    chars won't do. 
    
    Instead we match the tagger's tag against each of the tags in the POS map
    and check if the first part matches. For example, a tag like NoCmFeSgAc
    matches NoCm and therefore gets mapped to the lexicon tag "nocm". Yes,
    this is ugly and expensive.
    """
    
    def _map_pos(self, lempos):
        """
        map all pos tags in lempos
        """
        mapped_lempos = ""
        
        # map pos tag
        for pair in lempos.split():
            lemma, pos = pair.rsplit(self.delimiter, 1)
            mapped_lempos += ( lemma + 
                               self.delimiter + 
                               self._map_single_pos(pos) + 
                               " " )
            
        return mapped_lempos[:-1]
    
    def _map_single_pos(self, pos):
        for from_pos, to_pos in self.pos_map.items():
            if pos.startswith(from_pos):
                return to_pos
        return ""
    
    
    

    
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
    
    if with_lempos:
        for entry, values in trans_dict.lempos_iteritems():
            if " " in entry:
                if not with_multi_word:
                    continue
            elif not with_single_word:
                continue
                
            dist[len(values)] += 1
            
    if with_lemma:
        for entry, values in trans_dict.lemma_iteritems():
            if " " in entry:
                if not with_multi_word:
                    continue
            elif not with_single_word:
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
         