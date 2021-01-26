import time
from dataclasses import dataclass
from pathlib import Path

import logging

import math
import numpy as np
import networkx as nx
import igraph as ig
from typing import List, Dict, Tuple, Union, ClassVar
from os.path import join
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from collections import Counter


def dump_pickle(obj, fname):
    logging.error(f'Dumping pickle at {fname!r}')
    pickle.dump(obj, open(fname, 'wb'))
    return


def load_pickle(fname):
    return pickle.load(open(fname, 'rb'))


def get_graph(gname: str, basedir: str) -> Tuple[nx.Graph, str]:
    if gname == 'sample':
        g = nx.Graph()
        g.add_nodes_from(range(5), value='blue')
        g.add_nodes_from(range(5, 9), value='red')

        g.add_edges_from([(0, 1), (0, 3), (0, 4),
                          (1, 2), (1, 4), (1, 5),
                          (2, 3), (2, 4), (2, 8),
                          (3, 4),
                          (5, 6), (5, 7), (5, 8),
                          (6, 7), (6, 8),
                          (7, 8)])  # properly labeled
        g.name = 'sample'
    elif gname.endswith('.gpickle'):
        g = nx.read_gpickle(gname)
        g.name = Path(gname).stem
    else:
        if gname in ('waterloo', 'grenoble', 'uppsala'):
            g = nx.read_gpickle(f'../snap_data/cleaned/{gname}_lcc_attr.gpickle')
        try:
            g = nx.read_gml(join(basedir, 'input', f'{gname}.gml'))
        except Exception:
            path = join(basedir, 'input', f'{gname}.g')
            g = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())

        g.remove_edges_from(nx.selfloop_edges(g))
        if not nx.is_connected(g):
            nodes_lcc = max(nx.connected_components(g), key=len)
            g = g.subgraph(nodes_lcc).copy()
        name = g.name
        g = nx.convert_node_labels_to_integers(g, label_attribute='orig_label')
        g.name = name

    logging.error(f'{gname!r}, n = {g.order():,d}, m = {g.size():,d}')
    return g, 'value'


def draw_igraph(ig_g, color_map=None, bbox=(0, 0, 100, 100)):
    if color_map is not None:
        vals = ig_g.vs['value']
        vertex_color = [color_map[val] for val in vals]
    else:
        vertex_color = ['black' for v in ig_g.vs]

    fig = ig.plot(ig_g, bbox=bbox, vertex_color=vertex_color, )
    return fig


def igraph_read_gml(fname):
    ig_g: ig.Graph = ig.Graph.Read_GML(fname)
    vertex_attrs = ig_g.vs.attributes()
    keep = 'id', 'value'
    # turn everything into int, and drop anything that is not '
    for vertex_attr in vertex_attrs:
        if vertex_attr not in keep:
            del ig_g.vs[vertex_attr]
        else:
            ig_g.vs[vertex_attr] = list(map(lambda x: int(x), ig_g.vs[vertex_attr]))
    return ig_g


def draw_networkx_graph(nx_g: nx.Graph, color_map=None, node_color=None, ax=None, pos=None):
    if color_map is not None:
        vals = nx.get_node_attributes(nx_g, 'value').values()
        node_color = [color_map[val] for val in vals]
    else:
        if node_color is None: node_color = 'silver'

    node_shape = 'o' if nx_g.order() == 4 else 's'
    if pos is None:
        pos = nx.spring_layout(nx_g)
    if ax is None: ax = plt.gca()

    # pos = nx.rescale_layout_dict(pos, scale=0.2)
    ax.margins(0.4)
    ax.set_axis_off()

    nx.draw_networkx(nx_g, pos=pos,  ax=ax,
                     node_color=node_color, node_size=300, edgecolors='black', node_shape=node_shape,
                     edge_color='black',  width=2, with_labels=False)

    return


def networkx_to_igraph(nx_g):
    vals_dict = nx.get_node_attributes(nx_g, 'value')
    mapping = {uniq_val: i for i, uniq_val in enumerate(set(vals_dict.values()))}
    _vals = {n: mapping[v] for n, v in vals_dict.items()}
    nx.set_node_attributes(nx_g, name='_value', values=_vals)
    return ig.Graph.from_networkx(nx_g)


def get_color_mapping(ig_g: ig.Graph):
    # returns a color mapping for all the unique values in the graph
    vals = ig_g.vs['value']
    palette = sns.color_palette('pastel', len(set(vals)))
    mapping = {uniq_val: col for uniq_val, col in zip(set(vals), palette)}
    return mapping


class MotifCounter:
    """
    Class for counting motifs - both regular and colored
    """
    ISOCLASS_DICT = {
        2: {1: 'edge (g_21)'},
        3: {2: '2-star (g_32)', 3: 'tri (g_31)'},
        4: {6: '4-path (g_46)', 4: '3-star (g_45)', 10: '4-clique (g_41)',
            8: '4-cycle (g_44)', 7: '4-tailed-tri (g_43)', 9: '4-chordal-cycle (g_42)'}
    }

    def __init__(self, name: str, input_graph: ig.Graph, basedir: str) -> None:
        self.basedir = basedir
        self.name: str = name
        self.ig_g: ig.Graph = input_graph
        self.color_map = get_color_mapping(ig_g=self.ig_g)
        self.motif_hashes: Dict[str, Motif] = {}  # key: WL(motif) , val: (motif_graph, freq)
        self.sorted_motifs: List[Motif] = []   # all motifs sorted by freq
        return

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        st = f'name: {self.name!r}\t n: {self.ig_g.vcount():,d}\t m: {self.ig_g.ecount():,d}\n'
        st += ',\t'.join(f'({motif.size}) {motif.name!r} {motif.freq:,d}x' for motif in list(self.motif_hashes.values())[: 5])
        return st

    def plot_motifs(self, top_k=20):
        assert len(self.motif_hashes) != 0

        nrows = min(4, int(math.sqrt(len(self.motif_hashes))))
        ncols = top_k // nrows

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        if ncols == 1: axs = np.array([axs])
        axs = axs.flatten()
        axs = axs[: top_k]  # plot the top k most freq motifs
        fig.set_size_inches(ncols * 2.5, nrows * 2.5)

        suptitle = f'{self.name}\t\tn={self.ig_g.vcount():,d}\t\tm={self.ig_g.ecount():,d}\t\t{len(self.color_map.keys())} attributes'.expandtabs()

        if len(self.sorted_motifs) == 0:
            self.sorted_motifs = sorted(self.motif_hashes.values(), reverse=True)

        for motif, ax in zip(self.sorted_motifs, axs):
            color_map = self.color_map if motif.is_colored else None
            motif.plot(ax=ax, color_map=color_map)

        plt.suptitle(suptitle, fontsize=18, y=0.99)
        plt.tight_layout(h_pad=4, w_pad=2)
        plt.show()
        return

    def count(self, ks=(3, 4), overwrite=True) -> None:
        fname = Path(join(self.basedir, 'output', 'motifs', f'{self.name}_motif_hash.pkl'))
        if fname.exists() and not overwrite:
            print(f'Existing Motif pickle found at {fname.stem!r}, skipping')
            self.motif_hashes = load_pickle(fname)

        else:
            for k in ks:
                assert k in (3, 4), 'k must be 3 or 4'
                self.ig_g.motifs_randesu(size=k, callback=self.callback)

            dump_pickle(self.motif_hashes, fname)
        return

    def callback(self, ig_g: ig.Graph, nodes_in_motif: List[int], iso_class: int):
        k = len(nodes_in_motif)

        motif_sg: ig.Graph = ig_g.subgraph(nodes_in_motif)
        motif_sg_nx = motif_sg.to_networkx()

        motif_hash_normal = nx.weisfeiler_lehman_graph_hash(motif_sg_nx)
        motif_hash_colored = nx.weisfeiler_lehman_graph_hash(motif_sg_nx, node_attr='value')

        if motif_hash_normal not in self.motif_hashes:
            motif = Motif(graph=self.name, name=MotifCounter.ISOCLASS_DICT[k][iso_class], size=k,
                          iso_class=iso_class, freq=0, is_colored=False, sg_nx=motif_sg_nx, m=motif_sg_nx.size())
            self.motif_hashes[motif_hash_normal] = motif

        self.motif_hashes[motif_hash_normal].freq += 1  # increase the freq

        if motif_hash_colored not in self.motif_hashes:
            motif = Motif(graph=self.name, name=MotifCounter.ISOCLASS_DICT[k][iso_class], size=k,
                          iso_class=iso_class, freq=0, is_colored=True, sg_nx=motif_sg_nx, m=motif_sg_nx.size())
            self.motif_hashes[motif_hash_colored] = motif
        self.motif_hashes[motif_hash_colored].freq += 1  # increase the freq
        return


@dataclass
class Motif:
    """
    A class for motifs
    """
    graph: str
    name: str
    iso_class: int
    size: int
    sg_nx: nx.Graph
    m: int
    is_colored: bool
    freq: int
    NODE_POS: ClassVar[Dict] = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)}

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        if self.is_colored:
            wl_hash = nx.weisfeiler_lehman_graph_hash(self.sg_nx, node_attr='value')
        else:
            wl_hash = nx.weisfeiler_lehman_graph_hash(self.sg_nx)
        return wl_hash

    def __gt__(self, other):
        return self.freq > other.freq

    def plot(self, ax, color_map):
        pos = Motif.NODE_POS
        title = f'{self.name}  {self.freq:,d}x'
        draw_networkx_graph(self.sg_nx, ax=ax, pos=pos, color_map=color_map)
        ax.set_title(title)
        return


def main():
    basedir = '/data/ssikdar/Attributed-VRG/'

    names = ['karate', 'football', 'polbooks', 'wisconsin', 'texas', 'cornell',
             'polblogs', 'cora', 'citeseer', 'film', 'chameleon', 'pubmed', 'squirrel']

    for name in names[: 5]:
        graph_filename = join(basedir, 'input', f'{name}.gml')

        print(f'Reading ', name)
        ig_g: ig.Graph = igraph_read_gml(graph_filename)
        mc = MotifCounter(name=name, input_graph=ig_g, basedir=basedir)

        start = time.perf_counter()
        mc.count(ks=[3, 4], overwrite=False)
        # mc.plot_motifs()
        end = time.perf_counter()
        print(f'Counting motifs for {name!r} took {end - start:.2g} sec')
        break
    return


if __name__ == '__main__':
    main()
