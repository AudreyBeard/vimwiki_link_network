# Try with '/home/joshbeard/code/labnotes/vimwiki/index.wiki'

import argparse
import os
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain

parser = argparse.ArgumentParser(description="Creates a graph of Vimwiki Links")
parser.add_argument('entry_point', type=str, nargs=1, metavar='filename', help='file name of entry point for graph traversal')

fpath = '/home/joshbeard/code/labnotes/vimwiki/index.wiki'


class VimwikiGraph(nx.DiGraph):
    def __init__(self, entry_point_fpath, debug=False):
        super().__init__()
        self.root_dir, ep_fname = os.path.split(entry_point_fpath)
        self.ep = os.path.splitext(ep_fname)[0]
        self.simple_edges = []
        self._visited = set()
        self._to_visit = set()
        self._db = debug
        self.nx_graph = None
        self.expanded = False

    def __repr__(self):
        return super().__repr__()
        #rep = self.__class__.__name__
        #rep += ' rooted at {}.wiki'.format(os.path.join(self.root_dir, self.ep))
        #rep += '\nWith found nodes:\n'
        #for n in self.node_paths:
        #    rep += '  {}\n'.format(n)
        #return rep

    def get_path(self, relative_path, cur_dir=None):
        """ Gets the correct path for a link
        """
        # All links that start with '/' are at root directory
        if relative_path.startswith('/'):
            cur_dir = self.root_dir
            relative_path = relative_path[1:]

        # If current directory is not given, assume it's root
        if cur_dir is None:
            cur_dir = self.root_dir

        return os.path.realpath(os.path.join(cur_dir, relative_path))

    def visit(self, cur_node):
        """ Visits a single cur_node and updates set of not-yet-visited nodes
        """
        # Add this cur_node to list of found nodes
        self._visited.add(cur_node)

        # Get links at this cur_node:
        links = get_wiki_links(cur_node)
        if links is not None:
            # Get parent directory of current cur_node and get full link paths
            cur_dir = os.path.split(cur_node)[0]
            node_paths = [self.get_path(l, cur_dir) for l in links]

            # Add to set of not-yet-visited nodes as long as we haven't already
            # been there
            self._to_visit.update(set(node_paths).difference(self._visited))
            simple_cur_node = self._simple_path(cur_node)
            simple_edges = [[simple_cur_node, self._simple_path(n)] for n in node_paths]
            self.simple_edges.extend(simple_edges)

        return

    def _simple_path(self, node_path):
        return re.sub(self.root_dir, '', node_path)

    def organize(self):
        nlist = [['/index']]
        out_degs = np.array([d for n, d in self.out_degree()])
        in_degs = np.array([d for n, d in self.in_degree()])
        nodes = np.array([n for n, d in self.in_degree()])
        out_degs = out_degs[np.where(nodes != '/index')]
        in_degs = in_degs[np.where(nodes != '/index')]
        nodes = nodes[np.where(nodes != '/index')]
        ratio = out_degs / in_degs
        mean_rat = ratio.mean()
        std_rat = np.std(ratio)
        nlist.append(nodes[np.where(ratio > mean_rat + std_rat)])
        nlist.append(nodes[np.where(ratio <= mean_rat + std_rat * (ratio > mean_rat))])
        nlist.append(nodes[np.where(ratio <= mean_rat)])
        self.nlist = nlist
        self._out_degs = out_degs
        self._in_degs = in_degs
        self._nodes = nodes
        self._ratio = ratio

    def visit_all(self, expand_edges=False):
        if self._db:
            import ipdb
            ipdb.set_trace()

        # Enter graph
        self.visit(self.get_path(self.ep))

        # Explore each not-yet-visited node
        while len(self._to_visit) > 0:
            self.visit(self._to_visit.pop())

        if expand_edges and not self.expanded:
            self.expand_all_edges()

        self.add_edges_from(self.simple_edges)
        self.organize()

    def expand_edges(self, edge):
        """Here, edge should be a simple edge (that is, no root_dir)"""
        split_edge = [n.split(os.path.sep) for n in edge]
        split_edge = [i if len(i[0]) > 0 else i[1:] for i in split_edge]
        nodes = list(chain(*split_edge))
        expanded_edge = [[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)]
        return expanded_edge

    def expand_all_edges(self):
        self.simple_edges = list(chain(*[self.expand_edges(edge) for edge in self.simple_edges]))
        self.expanded = True

    def plot_nx_graph(self, draw_type=None):
        plt.subplot(111)
        if draw_type is None:
            nx.draw(self, with_labels=True)
        elif draw_type.lower() == 'shell':
            nx.draw_shell(self, with_labels=True, nlist=self.nlist)
        elif draw_type.lower() == 'circular':
            nx.draw_circular(self, with_labels=True)
        elif draw_type.lower() == 'spectral':
            nx.draw_spectral(self, with_labels=True)

        plt.show()


def get_wiki_links(fname):
    if not fname.endswith('.wiki'):
        fname = fname + '.wiki'

    if not os.path.exists(fname):
        return None

    with open(fname) as fid:
        text = fid.read()

    # Find all links
    links = re.findall('\[\[.*?\]\]', text)

    # Strip off brackets and disregard links to self
    links = [s.strip('[[').strip(']]') for s in links if not s[2] == '#']

    # Strip off link aliases
    links = [re.sub('\|.*', '', s) for s in links]

    return links


if __name__ == "__main__":
    args = parser.parse_args()
    ep = args.entry_point[0]

    graph = VimwikiGraph(ep)
