"""
drawing of translation graphs using Graphviz
"""

# TODO:
# - clean up & speed up 


import logging
import os
import subprocess
import tempfile

import pydot

from tg.graphproc import GraphProcess


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
        color="gray",
        fontname="Helvetica",
        fontsize=8)
    
    SOURCE_COLOR = "#c2a5cf"
    TARGET_COLOR = "#abdba0"
    
    def __init__(self, nx_graph, best_score_attr="freq_score",
                 base_score_attrs=[]):
        self.base_score_attrs = base_score_attrs
        self.best_score_attr = best_score_attr
        self.dot_graph = pydot.Dot('g0', graph_type='digraph',
                                   **self.GRAPH_DEFAULTS)
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
                # v must have a corresponding node in the dot graph; this is
                # the reason that this cannot be intergrated in step 1
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
            elif data.get("name") == "part":
                edge = self.hyper_edge(u, v, data)
            else:
                edge = self.trans_edge(u, v, data)
                
            self.dot_graph.add_edge(edge)
            
        # 4. Mark best nodes
        self.mark_best_nodes(nx_graph)
        
        # Finally, some nasty hacks to get the source words drawn in the
        # direction from top to bottom. We introduce a invisible root node
        # with ordered invisible eges to all normal source nodes.
        # Disadvantage is that setting ordering=out messes up the order of
        # target words in multi-word expressions...
        self.dot_graph.set_ordering("out")
        self.dot_graph.add_node(pydot.Node("__root__", shape="point",
                                           style="invis"))
        
        for sn in nx_graph.source_nodes(ordered=True):
            if not nx_graph.is_hyper_source_node(sn):
                self.dot_graph.add_edge(
                    pydot.Edge("__root__", str(sn), style="invis", weight=1))

        
             
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
        label=u"{}\\n{}/{}\\n{}".format(
            data["word"],
            data["lemma"], 
            data["pos"],
            "|".join(data.get("lex_lempos", []))).encode("utf-8")
        return pydot.Node(str(u), 
                          label=label, 
                          fillcolor=self.SOURCE_COLOR,
                          **self.NODE_DEFAULTS)        
    
    def target_node(self, u, data):
        label=u"{}/{}".format(data["lemma"], 
                              data["pos"]).encode("utf-8")
        return pydot.Node(str(u), 
                          label=label, 
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
        edge = pydot.Edge(str(u), str(v), style="bold", **self.EDGE_DEFAULTS)
        edge.set_color(self.SOURCE_COLOR)
        return edge
    
    def target_next_edge(self, u, v, data):
        edge = pydot.Edge(str(u), str(v), style="bold", weight=0, **self.EDGE_DEFAULTS)
        edge.set_color(self.TARGET_COLOR)
        return edge
    
    def hyper_edge(self, u, v, data):
        return pydot.Edge(str(u), str(v), **self.EDGE_DEFAULTS)
    
    def trans_edge(self, u, v, data):
        try:
            label = "{0:.2f}".format(data[self.best_score_attr])
        except KeyError:
            label = "???"
            
        penwidth = max(10 * data.get(self.best_score_attr, 0), 1)
        
        if self.base_score_attrs:
            labels = []
            
            for score_attr in self.base_score_attrs:
                try:
                    labels.append("{0}={1:.2f}".format(score_attr,
                                                       data[score_attr]))
                except KeyError:
                    pass
                
            label += " (" + "; ".join(labels) + ")"
                                     
        return pydot.Edge(str(u), str(v), label=label, penwidth=penwidth,
                       **self.EDGE_DEFAULTS)
        
    def mark_best_nodes(self, nx_graph):
        for u in nx_graph.source_nodes_iter():
            score, v = nx_graph.max_score(u, self.best_score_attr)
            if score:
                node = self.overall_best_node(v)
                self.dot_graph.add_node(node)
                
            for score_attr in self.base_score_attrs:
                score, v = nx_graph.max_score(u, score_attr)
                if score:
                    node = self.base_best_node(v)
                    self.dot_graph.add_node(node)
                   
    def overall_best_node(self, u):
        return pydot.Node(str(u), fillcolor="#386CB0")
    
    def base_best_node(self, u):
        return pydot.Node(str(u), color="black")



class Draw(GraphProcess):
    
    def __init__(self, drawer=DrawGV):
        self.drawer = drawer
    
    def _single_run(self, graph, out_fname=None, out_format="pdf",
                    best_score_attr="freq_score", base_score_attrs=[], 
                    out_dir="", fname_prefix=None):
        log.debug("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        drawer = self.drawer(graph, best_score_attr=best_score_attr, 
                             base_score_attrs=base_score_attrs)
        
        if not out_fname:
            out_fname = "{}-{:03d}.{}".format(
                fname_prefix or "graph",
                graph.graph["n"],
                out_format )
            
        if out_dir:
            if not os.path.exists(out_dir):
                log.info("creating output dir " + out_dir)
                os.makedirs(out_dir)   
            out_fname = os.path.join(out_dir, out_fname)
            
        drawer.write(out_fname, out_format)
        
        
        
def draw(graph, base_score_attrs=[]):
    """
    debug function to draw graphs on Mac OS
    """
    drawer = Draw()
    out_fname=tempfile.NamedTemporaryFile(suffix=".pdf").name
    drawer(graph, out_fname, base_score_attrs=base_score_attrs)
    subprocess.call(["open", "-a", "Preview", out_fname])
    
    