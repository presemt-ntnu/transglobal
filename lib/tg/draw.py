import logging
import pydot

from tg.graphproc import GraphProces


log = logging.getLogger(__name__)


                


class DrawGV:
    """
    Draw a translation graph using Graphviz
    """
    
    GRAPH_DEFAULTS = dict(
         rankdir="LR", 
         pad=0.5,
        fontname="Helvetica",
        fontsize=8)
    
    NODE_DEFAULTS = dict(
        shape="box", 
        color="white",
        style="filled", 
        height=0,
        width=0,
        margin=0.05,
        fontname="Helvetica",
        fontsize=8)
    
    EDGE_DEFAULTS = dict(
        fontname="Helvetica",
        fontsize=8)
    
    SOURCE_COLOR = "#c2a5cf"
    TARGET_COLOR = "#abdba0"
    
    def __init__(self, nx_graph, score_attr="score"):
        self.score_attr = score_attr
        self.dot_graph = pydot.Dot('g0', graph_type='digraph', **self.GRAPH_DEFAULTS)   
        hypernodes = []
        # dot subgraph of all source nodes, which forces them on the same rank
        subgraph = pydot.Subgraph('', rank='same') 
    
        # 1. Add all nodes to the dot graph
        for u,data in nx_graph.nodes_iter(data=True):
            if nx_graph.is_source_node(u):
                node = self.source_node(u, data)
                subgraph.add_node(node)
            elif nx_graph.is_target_node(u):
                node = self.target_node(u, data)
            else:
                node = self.hyper_node(u, data)
                hypernodes.append(u)
                
            self.dot_graph.add_node(node)
                
        self.dot_graph.add_subgraph(subgraph)     
        
        # 2. Force all target nodes connected to the same hypernode 
        #    to be on the same rank
        for u in hypernodes:
            subgraph = pydot.Subgraph('', rank='same') 
            
            for v in nx_graph.successors_iter(u):
                # v must have a corresponding node in the dot graph; this is the reason that
                # this cannot be intergrated in step 1
                node = self.dot_graph.get_node(v)[0]
                subgraph.add_node(node)
    
            self.dot_graph.add_subgraph(subgraph)     
    
        # 3. Add edges to dot graph
        for u,v,data in nx_graph.edges_iter(data=True):
            if data.get("name") == "next":
                if nx_graph.is_source_node(u):
                    edge = self.source_next_edge(u, v, data)
                else:
                    edge = self.target_next_edge(u, v, data)
            else:
                edge = self.trans_edge(u, v, data)
                
            self.dot_graph.add_edge(edge)
            
             
    def write(self, out_fname, out_format="raw"):
        """
        write drawing to file in any of the following formats: 
        
        'canon', 'cmap', 'cmapx', 'cmapx_np', 'dia', 'dot', 'fig', 'gd',
        'gd2', 'gif', 'hpgl', 'imap', 'imap_np', 'ismap', 'jpe', 'jpeg',
        'jpg', 'mif', 'mp', 'pcl', 'pdf', 'pic', 'plain', 'plain-ext', 'png',
        'ps', 'ps2', 'svg', 'svgz', 'vml', 'vmlz', 'vrml', 'vtx', 'wbmp',
        'xdot', 'xlib'
        """
        log.info("writing drawing to " + out_fname)
        self.dot_graph.write(out_fname, format=out_format)
    
    def source_node(self, u, data):
        return pydot.Node(str(u), 
                          label=u"{}/{}\\n{}".format(
                              data["lemma"], 
                              data["pos"],
                              "|".join(data.get("lex_lempos", []))).encode("utf-8"), 
                          fillcolor=self.SOURCE_COLOR,
                          **self.NODE_DEFAULTS)        
    
    def target_node(self, u, data):
        return pydot.Node(str(u), 
                          label=u"{}/{}".format(data["lemma"], data["pos"]).encode("utf-8"), 
                          fillcolor=self.TARGET_COLOR,
                          **self.NODE_DEFAULTS)
    
    def hyper_node(self, u, data):
        if u.startswith("hs"):
            return pydot.Node(
                str(u), 
                label=u"|".join(data.get("lex_lempos", [])).encode("utf-8"), 
                fillcolor=self.SOURCE_COLOR,
                **self.NODE_DEFAULTS)
        else:
            return pydot.Node(str(u), shape="point", style="bold")
    
    def source_next_edge(self, u, v, data):
        return pydot.Edge(str(u), str(v), color=self.SOURCE_COLOR, style="bold",
                          **self.EDGE_DEFAULTS)
    
    def target_next_edge(self, u, v, data):
        return pydot.Edge(str(u), str(v), color=self.TARGET_COLOR, style="bold",
                          **self.EDGE_DEFAULTS)
    
    def trans_edge(self, u, v, data):
        try:
            label = "{0:.2f}".format(data[self.score_attr])
            penwidth = max(10 * data[self.score_attr], 1)
        except KeyError:
            label = ""
            penwidth = 1
                                     
        return pydot.Edge(str(u), str(v), color="gray", label=label,
                          penwidth=penwidth, **self.EDGE_DEFAULTS)
        




class Draw(GraphProces):
    
    def __init__(self, drawer=DrawGV):
        self.drawer = drawer
    
    def _single_run(self, graph, outf_fname=None, out_format="pdf", score_attr="score"):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        drawer = self.drawer(graph, score_attr=score_attr)
        if not outf_fname:
            out_fname = "graph-{}.{}".format(
                graph.graph["id"],
                out_format )
        drawer.write(out_fname, out_format)