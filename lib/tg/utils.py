import codecs

from os import makedirs
from os.path import exists, isfile, dirname


def create_dirs(path):
    """
    create directores if they do not exists yet
    """
    if isfile(path):
        path = dirname(path)
        
    if not exists(path):
        makedirs(path)
        



def read_map(fname, delimiter="\t", comment="#", encoding="utf-8"):
    """
    read a string-to-string mapping dict from file (e.g., a pos tag to pos
    tag mapping)
    """ 
    with codecs.open(fname, encoding=encoding) as inf:
        strmap = {}
        for line in inf:
            line = line.split(comment)[0].strip()
            if line:
                key, val = line.split(delimiter)
                strmap[key] = val
        return strmap