import math

import networkx as nx
import sys; sys.path.append('../')
from VRG.src.LightMultiGraph import LightMultiGraph


def gamma_code(n):
    bits = math.log2(n)
    return 2 * bits + 1


def nbits(x):
    """
    Returns the number of bits to encode x in binary
    :param x: argument
    :return: number of bits required to encode x in binary
    """
    if x == 0: return 0
    return math.log2(x)


def graph_dl(g: LightMultiGraph, attributed: bool = False):
    """
     Get DL for graphs using Gamma coding
     :param g:  a multigraph
     :return: Length in bits to represent graph g in binary
    """
    n = g.order()
    m = len(g.edges())  # here we dont use size because it will throw the algorithm into a whack

    l_u = find_lu(g, attributed)

    # encoding the nodes
    dl_v = nbits(n) + n * nbits(l_u)

    # encoding the edges
    dl_edges = 0
    for u, v in g.edges():
        k = g.number_of_edges(u, v)
        dl_edges += 2 * gamma_code(k + 1)  # 2 because the graph is undirected
    nnz = 2 * m  # the number of non-zero entries in the matrix
    dl_edges += (n ** 2 - nnz) * gamma_code(0 + 1)

    dl_e = nbits(m) + nbits(l_u) * dl_edges  # added the l_u factor

    return dl_v + dl_e


def find_lu(g, attributed: bool = False) -> int:
    l_u = 1  # for edges
    node_types = set()

    for n, d in g.nodes(data=True):
        node_types.add(f'nt-{d["nt"]}') if 'nt' in d else node_types.add('t')
        if attributed:
            if 'attr_dict' in d: d = d['attr_dict']
            for k, v in d.items():
                if k != 'nt' and k != 'orig_label': node_types.add(v)

    l_u += len(node_types)
    return l_u


if __name__ == '__main__':
    g = nx.read_gml('../input/football.gml')
    print(graph_dl(g), graph_dl(g, attributed=True))
    # lmg = LightMultiGraph()
    # lmg.add_edges_from(g.edges())
    # print(graph_dl(lmg))
    # h = nx.path_graph(10)
    # g.add_nodes_from(range(5), color='red')
    # g.add_nodes_from(range(5, 10), color='blue')
    # nt = NonTerminal(size=5, nodes_covered={1, 2})
    # g.add_node('a', nt=nt)
    # print(find_lu(g))
