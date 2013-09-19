import logging

import networkx as nx

from tg.exception import TGException


log = logging.getLogger(__name__)


class TransGraph(nx.DiGraph):
    
    source_node_prefix = "s"
    target_node_prefix = "t"
    hyper_source_node_prefix = "hs"
    hyper_target_node_prefix = "ht"
    delimiter = u"/"
    max_scores_cache = "_max_scores"
    
    def __init__(self, data=None, **attr):
        nx.DiGraph.__init__(self, data, **attr) 
        self.source_node_count = 0   
        self.target_node_count = 0   
        self.hyper_source_node_count = 0   
        self.hyper_target_node_count = 0  
        self.source_start_node = None
        
    def __repr__(self):
        attrs = ", ".join("{}={!r}".format(k,v) 
                          for k,v in self.graph.items())
        return "{}({})".format(self.__class__.__name__, attrs)
    
    def __str__(self):
        return self.__repr__()
    
    #-------------------------------------------------------------------------
    # source nodes
    #-------------------------------------------------------------------------  
    
    def add_source_node(self, **attr):
        self.source_node_count += 1
        u = "{0}{1}".format(self.source_node_prefix,
                            self.source_node_count)
        self.add_node(u, **attr)
        return u
    
    def is_source_node(self, u):
        return u.startswith(self.source_node_prefix) 
    
    def set_source_start_node(self, u):
        self.source_start_node = u
    
    def source_nodes(self, data=False, ordered=False):
        return list(self.source_nodes_iter(data=data, ordered=ordered))
        
    def source_nodes_iter(self, data=False, ordered=False):
        if ordered:
            return self._ordered_source_nodes_iter(data=data)
        else:
            return self._unordered_source_nodes_iter(data=data)
                
    def _ordered_source_nodes_iter(self, data=False):
        u = self.source_start_node
        
        while u:
            if data:
                yield u, self.node[u]
            else:
                yield u
                
            for u, v, d in self.out_edges_iter(u, data=True):
                if d.get("name") == "next":
                    u = v
                    # assuming there is only one "next" node, ignore any
                    # remaining outgoing edges
                    break
            else:
                # none of the outgoing edges has name "next", so this was the
                # last token
                return
            
    def _unordered_source_nodes_iter(self, data=False):
        if data:
            return ( (u,d) for u, d in self.nodes_iter(data=True)
                     if self.is_source_node(u) )
        else:
            return ( u for u in self.nodes_iter()
                     if self.is_source_node(u) )
    
    #-------------------------------------------------------------------------
    # target nodes
    #-------------------------------------------------------------------------  
    
    def add_target_node(self, **attr):
        self.target_node_count += 1
        u = "{0}{1}".format(self.target_node_prefix,
                            self.target_node_count)
        self.add_node(u, **attr)
        return u    
    
    def is_target_node(self, u):
        return u.startswith(self.target_node_prefix) 
    
    #-------------------------------------------------------------------------
    # hyper nodes
    #-------------------------------------------------------------------------


    def add_hyper_source_node(self, nodes):
        self.hyper_source_node_count+= 1
        u = "{0}{1}".format(self.hyper_source_node_prefix,
                            self.hyper_source_node_count)
        for v in nodes:
            self.add_edge(v, u, name="part")
        return u
    
    def add_hyper_target_node(self, nodes):
        self.hyper_target_node_count += 1
        u = "{0}{1}".format(self.hyper_target_node_prefix,
                            self.hyper_target_node_count)
        for v in nodes:
            self.add_edge(u, v, name="part")
        return u
    
    def is_hyper_source_node(self, u):
        return u.startswith(self.hyper_source_node_prefix) 
    
    def is_hyper_target_node(self, u):
        return u.startswith(self.hyper_target_node_prefix) 
    
    def source_parts_iter(self, u):
        part_nodes = ( v for v, _, data in self.in_edges_iter(u, data=True)
                       if data.get("name") == "part" )
        return self.ordered_nodes_iter(part_nodes)
    
    def target_parts_iter(self, u):
        part_nodes = ( v for _, v, data in self.out_edges_iter(u, data=True)
                       if data.get("name") == "part" )
        return self.ordered_nodes_iter(part_nodes)
    
    #-------------------------------------------------------------------------
    # attributes
    #-------------------------------------------------------------------------
    
    # source only
    
    def source_words(self):
        return [ d["word"] 
                 for _,d  in self.source_nodes_iter(data=True, ordered=True) ]
    
    def source_lemmas(self):
        return [ d["lemma"] 
                 for _,d in self.source_nodes_iter(data=True, ordered=True) ]
    
    def source_lempos(self):
        return [ ( d["lemma"] + self.delimiter + d["pos"] )
                 for _,d in self.source_nodes_iter(data=True, ordered=True) ]
        
    def source_string(self):
        return " ".join(self.source_words())     

    # nodes
    
    def node_attrib(self, u, attrib, as_list=False):
        if self.is_source_node(u) or self.is_target_node(u):
            l = [ self.node[u][attrib] ]
        elif self.is_hyper_source_node(u):
            l = [ self.node[v][attrib]
                  for v in self.source_parts_iter(u) ]
        elif self.is_hyper_target_node(u):
            l =  [ self.node[v][attrib]
                   for v in self.target_parts_iter(u) ]
        else:
            raise ValueError("not a node")
        
        if as_list:
            return l
        else:
            return " ".join(l)
    
    def word(self, u, as_list=False):
        return self.node_attrib(u, "word", as_list=as_list)
    
    def lemma(self, u, as_list=False):
        return self.node_attrib(u, "lemma", as_list=as_list)
    
    def pos(self, u, as_list=False):
        return self.node_attrib(u, "pos", as_list=as_list)
    
    def lempos(self, u, as_list=False):
        l =  [ self.delimiter.join(pair)
               for pair in zip(self.node_attrib(u, "lemma", True),
                               self.node_attrib(u, "pos", True)) ]
        if as_list:
            return l
        else:
            return " ".join(l)
        
    
    def string(self, u):
        " ".join(self.node_attrib(u, "word"))
                
        
    #-------------------------------------------------------------------------
    # word order
    #-------------------------------------------------------------------------
            
    def add_word_order_edge(self, u, v, attr_dict=None, **attr):
        self.add_edge(u, v, name="next", attr_dict=attr_dict, **attr)
        
    def ordered_nodes_iter(self, nodes):
        """
        return an iterator over nodes in order (i.e. following edges named
        "next")
        """
        # find first node
        for u in nodes:
            if self.is_first_node(u):
                break
        else:
            # should never happen
            raise TGException("no first node among ordered nodes")
           
        while u:
            yield u
            
            for _,v,data in self.out_edges(u, data=True): 
                if data.get("name") == "next":
                    # found next node
                    u = v
                    break
            else:
                # reached final node
                return
                
    def is_first_node(self, u):
        """
        test if node is first (i.e. has predecessor with edge named "next")
        """
        for _, _, data in  self.in_edges_iter(u, data=True):
            if data.get("name") == "next":
                return False
        return True
    
    #-------------------------------------------------------------------------
    # translation
    #-------------------------------------------------------------------------
        
        
    def add_translation_edge(self, u, v, attr_dict=None, **attr):
        self.add_edge(u, v, name="trans", attr_dict=attr_dict, **attr)
        
    def trans_edges_iter(self, u=None):
        return ( (u,v,d) for u,v,d in self.out_edges_iter(u, data=True)
                 if d.get("name") == "trans" )
    
    def max_score(self, u, score_attr):
        """
        find max score
        
        Parameters
        ----------
        u: str
            Source node identifier
        score_attr: str
            Name of edge attribute containing the score.
            
        Returns
        -------
        t: tuple (score, node) or None
            A tuple t of a numerical score and target node identifier.
            If the score attribute is not found on any of the translation edges, 
            then both score and node are None.
            If there are no translation edges, 
            then both score and node are None.
            If there are multiple nodes with the same maximum score,
            one of them is chosen arbitrarily.
            
        Notes
        -----
        Max scores are cached with a dict on te source node, so make sure to 
        clear these caches if you rerun a scorer. 
        """
        try:
            return self.node[u][self.max_scores_cache][score_attr]
        except KeyError:
            pass
        
        max_score, max_node = None, None
        
        for _, v, d in self.trans_edges_iter(u):
            score = d.get(score_attr)
            if score > max_score:
                max_score, max_node = score, v
                max_node = v
                
        cache = self.node[u].setdefault(self.max_scores_cache, {})
        cache[score_attr] = max_score, max_node
        return max_score, max_node

        
        
        
