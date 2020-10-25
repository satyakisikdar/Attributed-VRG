import functools
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import Set
from typing import Union, Tuple, List

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns;

sns.set_style('white')
from matplotlib import gridspec
from numpy import linalg as la
from scipy import sparse as sps
from scipy.sparse import issparse

from VRG.src.LightMultiGraph import LightMultiGraph

sns.set(); sns.set_style("darkgrid")

clusters = {}  # stores the cluster members
original_graph = None  # to keep track of the original edges covered


def get_compatibility_matrix(g: nx.Graph, attr_name: str):
    """
    From Danai's heterophily paper
    :param g:
    :param attr_name:
    :return:
    """
    values = set(nx.get_node_attributes(g, attr_name).values())
    mapping = {val: i for i, val in enumerate(values)}
    print(mapping)
    C = nx.attribute_mixing_matrix(g, attribute=attr_name, mapping=mapping, normalized=False)
    np.fill_diagonal(C, C.diagonal() / 2)

    D = np.diag(np.diag(C))
    e = np.ones(shape=(len(mapping), 1))

    h = float((e.T @ D @ e) / (e.T @ C @ e))

    Y = np.zeros(shape=(g.order(), len(mapping)))
    for n, d in g.nodes(data=True):
        attr = d[attr_name]
        Y[n, mapping[attr]] = 1
    A = nx.adjacency_matrix(g)
    E = np.ones(shape=(A.shape[0], len(mapping)))

    H = (Y.T @ A @ Y) / (Y.T @ A @ E)

    return_d = dict(homophily_ratio=h, compatibility_mat=H, attr_name=attr_name, mapping=mapping)
    return return_d


def find_boundary_edges(g: LightMultiGraph, nbunch: Set[int]) -> List[Tuple]:
    """
        Collect all of the boundary edges (i.e., the edges
        that connect the subgraph to the original graph)

        :param g: whole graph
        :param nbunch: set of nodes in the subgraph
        :return: boundary edges
    """
    nbunch = set(nbunch)
    if len(nbunch) == g.order():  # it's the entire node set
        return []

    boundary_edges = []
    for u, v in nx.edge_boundary(g, nbunch):
        edges = [(u, v)] * g.number_of_edges(u, v)
        boundary_edges.extend(edges)
    return boundary_edges


def set_boundary_degrees(g: LightMultiGraph, sg: LightMultiGraph) -> None:
    boundary_degree = {n: 0 for n in sg.nodes}  # by default every boundary degree is 0

    for u, v in nx.edge_boundary(g, sg.nodes()):
        if sg.has_node(u):
            boundary_degree[u] += g.number_of_edges(u, v)
        else:
            boundary_degree[v] += g.number_of_edges(u, v)
    nx.set_node_attributes(sg, values=boundary_degree, name='b_deg')
    return


def set_boundary_degrees_old(g: LightMultiGraph, sg: LightMultiGraph) -> None:
    """
        Find the nunber of boundary edges that each node participate in.
        This is stored as a node level attribute - 'b_deg' in nodes in g that are part of nbunch

        :param g: whole graph
        :param sg: the subgraph
        :return: nothing
    """
    boundary_degree = {}

    for u in sg.nodes():
        boundary_degree[u] = 0
        for v in g.neighbors(u):
            if not sg.has_node(v):
                boundary_degree[u] += g.number_of_edges(u, v)  # for a multi-graph

    nx.set_node_attributes(sg, values=boundary_degree, name='b_deg')
    return


def get_nodes_covered(sg: LightMultiGraph) -> Set[int]:
    """
        Get nodes covered by a non-terminal
        :param g:
        :param sg:
        :return:
    """
    nodes_covered = set()

    for node, data in sg.nodes(data=True):
        if 'nt' in data:  # if the subgraph has a non-terminal node, the new non-terminal now has all the nodes_covered of that non-terminal node
            nt = data['nt']
            assert isinstance(nt.nodes_covered, set) and len(
                nt.nodes_covered) > 0, 'non-terminal has invalid nodes_covered'
            nodes_covered.update(nt.nodes_covered)
        else:  # it's a regular node
            nodes_covered.add(node)

    return nodes_covered


def load_pickle(fname):
    return pickle.load(open(fname, 'rb'))


def dump_pickle(obj, fname):
    logging.error(f'Dumping pickle at {fname!r}')
    pickle.dump(obj, open(fname, 'wb'))
    return


def node_matcher_b_deg(node_attr_1: Dict, node_attr_2: Dict) -> bool:
    """
    Only match boundary degrees
    """
    return node_attr_1['b_deg'] == node_attr_2['b_deg']


def node_matcher_strict(node_attr_1: Dict, node_attr_2: Dict) -> bool:
    """
    If node n1 in G1 are the same as node n2 in G2
    :param node_attr_1: Dictionary of node attrs
    :param node_attr_2:
    :return:
    """
    # check if they have the same set of attributes
    # only check if their b_deg are the same
    same = set(node_attr_1.keys()) == set(node_attr_2.keys())
    if same:
        # check if the values are the same too
        for key, val_1 in node_attr_1.items():
            if key == 'actual_label':  # ignore actual_label attribute
                continue
            val_2 = node_attr_2[key]
            if val_1 != val_2:  # if the values don't match, break
                same = False
                break
    return same


def edge_matcher(edge_attr_1: Dict, edge_attr_2: Dict) -> bool:
    """
    If edge e1 in G1 is the same as edge e2 in G2
    :param edge_attr_1:
    :param edge_attr_2:
    :return:
    """
    # they must have the same set of edge attrs
    same = set(edge_attr_1.keys()) == set(edge_attr_1.keys())

    if same:
        # check if the values are the same too
        for key, val_1 in edge_attr_1.items():
            val_2 = edge_attr_2[key]
            if val_1 != val_2:  # if values dont match
                same = False
                break
    return same


def jaccard(set1: Set[int], set2: Set[int]):
    return len(set1 & set2) / len(set1 | set2)


def cvm_distance(data1, data2) -> float:
    data1, data2 = map(np.asarray, (data1, data2))
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    assert len(cdf1) == len(cdf2), 'CDFs should be of the same length'
    d = np.sum(np.absolute(cdf1 - cdf2)) / len(cdf1)
    return np.round(d, 3)


def ks_distance(data1, data2) -> float:
    data1, data2 = map(np.asarray, (data1, data2))
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    d = np.max(np.absolute(cdf1 - cdf2))
    return np.round(d, 3)


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        ColorPrint.print_bold(f'({func.__name__}) Start: {datetime.now().ctime()}')
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        ColorPrint.print_bold(f'({func.__name__}) End: {datetime.now().ctime()}, Elapsed time: {elapsed_time:0.4f}s')
        return value

    return wrapper_timer


def _pad(A, N):
    """Pad A so A.shape is (N,N)"""
    n, _ = A.shape
    if n >= N:
        return A
    else:
        if issparse(A):
            # thrown if we try to np.concatenate sparse matrices
            side = sps.csr_matrix((n, N - n))
            bottom = sps.csr_matrix((N - n, N))
            A_pad = sps.hstack([A, side])
            A_pad = sps.vstack([A_pad, bottom])
        else:
            side = np.zeros((n, N - n))
            bottom = np.zeros((N - n, N))
            A_pad = np.concatenate([A, side], axis=1)
            A_pad = np.concatenate([A_pad, bottom])
        return A_pad


def fast_bp(A, eps=None):
    n, m = A.shape
    degs = np.array(A.sum(axis=1)).flatten()
    if eps is None:
        eps = 1 / (1 + max(degs))
    I = sps.identity(n)
    D = sps.dia_matrix((degs, [0]), shape=(n, n))
    # form inverse of S and invert (slow!)
    Sinv = I + eps ** 2 * D - eps * A
    try:
        S = la.inv(Sinv)
    except:
        Sinv = sps.csc_matrix(Sinv)
        S = sps.linalg.inv(Sinv)
    return S


def check_file_exists(path: Union[Path, str]) -> bool:
    """
    Checks if file exists at path
    :param path:
    :return:
    """
    if isinstance(path, str):
        path = Path(path)
    return path.exists()


class ColorPrint:
    @staticmethod
    def print_red(message, end='\n'):
        sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_green(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_orange(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_blue(message, end='\n'):
        # pass
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_bold(message, end='\n'):
        sys.stdout.write('\x1b[1;37m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_none(message, end='\n'):
        pass
        # sys.stdout.write(message + end)


def plot_graph(g, ax=None, title='', attr_name='', mixing_dict={}):
    unique_values = set(mixing_dict.values())
    colors = sns.color_palette('pastel', n_colors=len(unique_values))
    color_map = {val: col for val, col in zip(unique_values, colors)}

    try:
        colors = [color_map[d[attr_name]] for _, d in g.nodes(data=True)]
    except KeyError:
        colors = '#77dd77'
    pos = nx.spring_layout(g)

    if ax is not None:
        ax.set_title(title, fontsize=20)
    nx.draw_networkx_nodes(g, pos=pos, node_size=100, node_color=colors, alpha=0.7, ax=ax)
    nx.draw_networkx_edges(g, pos=pos, edge_color='gray', alpha=0.7, ax=ax)
    return


def grid_plot(graphs, graph_name='', attr_name='', mixing_dict={}):
    # todo keep the positions of the constant nodes fixed
    rows, cols = 2, 4
    plt.rcParams['figure.figsize'] = [30, 15]

    grid = gridspec.GridSpec(rows, cols)
    fig = plt.figure()
    for i, box in enumerate(grid):
        ax = fig.add_subplot(box)
        g = graphs[i]
        deg_as = round(nx.degree_assortativity_coefficient(g), 3)
        attr_as = round(nx.attribute_assortativity_coefficient(g, attribute=attr_name), 3)
        plot_graph(g, ax=ax, title=f'g{i + 1} {g.order(), g.size()} degree as: {deg_as} att as: {attr_as}')

    plt.tight_layout()
    plt.suptitle(f'{graph_name}', y=1, fontsize=10)
    plt.show()


def check_isomorphism(ref_g: nx.Graph, graphs: List[nx.Graph]) -> bool:
    return all(nx.is_isomorphic(ref_g, g) for g in graphs)


def nx_to_igraph(nx_g: Union[nx.Graph, nx.DiGraph]) -> ig.Graph:
    """
    Convert networkx graph to an equivalent igraph Graph
    attributes are stored as vertex sequences
    """
    nx_g = nx.convert_node_labels_to_integers(nx_g, label_attribute='old_label')
    old_label = nx.get_node_attributes(nx_g, 'old_label')

    weights = nx.get_edge_attributes(nx_g, name='wt')  # WEIGHTS are stored in WT
    if len(weights) == 0:
        is_weighted = False
        edge_list = list(nx_g.edges())
    else:
        is_weighted = True
        edge_list = [(u, v, w) for (u, v), w in weights.items()]

    is_directed = nx_g.is_directed()
    ig_g = ig.Graph.TupleList(edges=edge_list, directed=is_directed,
                              weights=is_weighted)

    logging.error(f'iGraph: n={ig_g.vcount()}\tm={ig_g.ecount()}\tweighted={is_weighted}\tdirected={is_directed}')

    for v in ig_g.vs:
        v['name'] = str(old_label[v.index])  # store the original labels in the name attribute
        v['label'] = str(v['name'])

    return ig_g


def nx_to_lmg(nx_g: nx.Graph) -> LightMultiGraph:
    lmg = LightMultiGraph()
    lmg.name = nx_g.name
    for n, d in nx_g.nodes(data=True):
        lmg.add_node(n, **d)
    lmg.add_edges_from(nx_g.edges(data=True))
    return lmg


def igraph_to_nx(ig_g: ig.Graph) -> nx.Graph:
    """
    Convert an igraph graph into a networkx graph
    """
    nx_g = nx.Graph()
    name = {}
    for v in ig_g.vs:
        v: ig.VertexSeq
        name[v.index] = int(v['name'])  # store the mapping

    for e in ig_g.es:
        e:ig.EdgeSeq
        nx_g.add_edge(name[e.source], name[e.target])

    return nx_g


def get_assortativity(g: nx.Graph, attr_name: str):
    if attr_name == 'degree':
        return nx.degree_assortativity_coefficient(g)
    else:
        return nx.attribute_assortativity_coefficient(g, attribute=attr_name)


def get_mixing_dict(g: nx.Graph, attr_name: str) -> Dict:
    """
    Row normalized mixing dict akin to a transition matrix
    :return:
    """
    attr_dict = nx.attribute_mixing_dict(g, attribute=attr_name, normalized=False)
    row_norm_attr_dict = {}
    for key1, dict1 in attr_dict.items():
        row_norm_attr_dict[key1] = {}
        for key2, val in dict1.items():
            row_norm_attr_dict[key1][key2] = val / sum(dict1.values())
    return row_norm_attr_dict
