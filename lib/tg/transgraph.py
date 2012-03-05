import networkx as nx



class TransGraph(nx.DiGraph):
    
    def __init__(self, data=None, **attr):
        nx.DiGraph.__init__(self, data, **attr) 
        self.source_node_prefix = "s"
        self.target_node_prefix = "t"
        self.hyper_node_prefix = "h"
        self.source_node_count = 0   
        self.target_node_count = 0   
        self.hyper_node_count = 0   
        
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
    
    def source_token_nodes(self, with_data=False):
        return list(self.source_nodes_iter(with_data=with_data))
        
    def source_nodes_iter(self, with_data=False):
        this_node = self.source_start_node
        
        while this_node:
            for src_node, dest_node, data in self.out_edges_iter(this_node,
                                                                 data=True):
                if data.get("name") == "next":
                    if with_data:
                        yield this_node, self.node[this_node]
                    else:
                        yield this_node

                    this_node = dest_node
                    # assuming there is only one "next" node, ignore
                    # remaining ougoing edges
                    break
            else:
                # none of the outgoing edges has name "next", so this must be
                # last token
                if with_data:
                    yield this_node, self.node[this_node]
                else:
                    yield this_node
                    
                this_node = None
            
    def source_words(self):
        return [ self.node[u]["word"] for u in self.source_nodes_iter() ]
    
    def source_lemmas(self):
        return [ self.node[u]["lemma"] for u in self.source_nodes_iter() ]
    
    def source_lempos(self, delimiter="/"):
        return [ ( self.node[u]["lemma"] + delimiter + self.node[u]["tag"] )
                 for n in self.source_nodes_iter() ]
        
    def source_string(self):
        return " ".join(self.source_words())
    
    
    

    def add_target_node(self, **attr):
        self.target_node_count += 1
        u = "{0}{1}".format(self.target_node_prefix,
                            self.target_node_count)
        self.add_node(u, **attr)
        return u    
    
    def is_target_node(self, u):
        return u.startswith(self.target_node_prefix) 
    
    
    def add_hyper_node(self, nodes, is_source=True):
        self.hyper_node_count+= 1
        u = "{0}{1}".format(self.hyper_node_prefix,
                            self.hyper_node_count)
        for v in nodes:
            if is_source:
                self.add_edge(u, v, name="part")
            else:
                self.add_edge(v, u, name="part")
            
        return u
    
    def is_hyper_node(self, u):
        return u.startswith(self.hyper_node_prefix) 
    
    
            
    def add_word_order_edge(self, u, v, attr_dict=None, **attr):
        self.add_edge(u, v, name="next", attr_dict=attr_dict, **attr)
        
    def add_translation_edge(self, u, v, attr_dict=None, **attr):
        self.add_edge(u, v, name="trans", attr_dict=attr_dict, **attr)
        
    def trans_edges_iter(self, u):
        return ( (u,v,data) for u,v,data in self.out_edges_iter(u, data=True)
                 if data.get("name") == "trans" )
                
                
        
        
        
        
