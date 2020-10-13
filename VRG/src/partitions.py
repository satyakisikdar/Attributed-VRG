"""
Contains the different partition methods
1. Conductance based partition
2. Spectral K-means
3. Leiden and Louvain methods
"""
import logging
import math
import random
from typing import Union

import igraph as ig
import networkx as nx
import scipy.sparse.linalg
import sklearn.preprocessing
from sklearn.cluster import KMeans

from VRG.src.LightMultiGraph import LightMultiGraph
from VRG.src.utils import nx_to_igraph


# LightMultiGraph = nx.Graph

def louvain_leiden_infomap_label_prop(g: Union[ig.Graph, nx.Graph, nx.DiGraph], max_size: int, method: str = 'leiden'):
    is_weighted = False
    if isinstance(g, nx.Graph) or isinstance(g, nx.DiGraph):  # turn it into an igraph Graph
        if len(nx.get_edge_attributes(g, name='wt')) == 0:
            is_weighted = False
        g = nx_to_igraph(g)

    weights = 'weight' if is_weighted else None
    return _get_list_of_lists(ig_g=g, method=method, weights=weights, max_size=max_size)


def _get_list_of_lists(ig_g, max_size, method='leiden', weights=None):
    tree = []

    if method == 'leiden':
        clusters = ig_g.community_leiden(weights=weights, objective_function='modularity', n_iterations=-1)
    elif method == 'louvain':
        clusters = ig_g.community_multilevel(weights=weights)
    elif method == 'labelprop':
        clusters = ig_g.community_label_propagation(weights=weights)
    elif method == 'infomap':
        clusters = ig_g.community_infomap(edge_weights=weights)
    elif method == 'leading_eigenvector':
        clusters = ig_g.community_leading_eigenvector(weights=weights)
    else:
        raise NotImplementedError(f'Improper method: {method!r}')

    if len(clusters) == 1:
        sg = clusters.subgraphs()[0]
        comms = [[int(n['name'])] for n in sg.vs()]
        return comms
        # if clusters.sizes()[0] <= max_size:
        #     assert sg.is_connected(mode='WEAK'), 'subgraph is disconnected'
        #     comms = [[int(n['name'])] for n in sg.vs()]90
        #     return comms
        # else:
        #     # try clustering 'sg' again using conductance
        #     nx_sg = igraph_to_nx(ig_g=sg)
        #     comms = approx_min_conductance_partitioning(g=nx_sg, max_k=max_size)
        #     return comms

    for sg in clusters.subgraphs():
        sg: ig.Graph
        if not sg.is_connected(mode='WEAK'):
            logging.error('subgraph is disconnected')
        tree.append(_get_list_of_lists(sg, method=method, weights=weights, max_size=max_size))

    return tree


def get_random_partition(g: LightMultiGraph, seed=None):
    nodes = list(g.nodes())
    if seed is not None:
        random.seed(seed)
    random.shuffle(nodes)
    return random_partition(nodes)


def random_partition(nodes):
    tree = []
    if len(nodes) < 2:
        return nodes

    left = nodes[: len(nodes) // 2]
    right = nodes[len(nodes) // 2:]

    tree.append(random_partition(left))
    tree.append(random_partition(right))
    return tree


def approx_min_conductance_partitioning(g: nx.Graph):
    """
    Approximate minimum conductance partinioning. I'm using the median method as referenced here:
    http://www.ieor.berkeley.edu/~goldberg/pubs/krishnan-recsys-final2.pdf
    :param g: graph to recursively partition
    :return: a dendrogram
    """
    lvl = []
    node_list = list(g.nodes())
    if len(node_list) == 1:
        return node_list

    if not nx.is_connected(g):
        for nodes_cc in nx.connected_components(g):
            p = g.subgraph(nodes_cc).copy()
            lvl.append(approx_min_conductance_partitioning(p))
        assert len(lvl) > 0
        return lvl

    assert nx.is_connected(g), "g is not connected in cond"

    fiedler_vector = nx.fiedler_vector(g, method='lanczos')

    p1, p2 = set(), set()

    fiedler_dict = {}
    for idx, n in enumerate(fiedler_vector):
        fiedler_dict[idx] = n
    fiedler_vector = [(k, fiedler_dict[k]) for k in sorted(fiedler_dict,
                                                           key=fiedler_dict.get, reverse=True)]
    half_idx = len(fiedler_vector) // 2  # floor division

    for idx, _ in fiedler_vector:
        if half_idx > 0:
            p1.add(node_list[idx])
        else:
            p2.add(node_list[idx])
        half_idx -= 1  # decrement so halfway through it crosses 0 and puts into p2

    sg1 = g.subgraph(p1).copy()
    sg2 = g.subgraph(p2).copy()

    iter_count = 0
    while not (nx.is_connected(sg1) and nx.is_connected(sg2)):
        sg1 = g.subgraph(p1).copy()
        sg2 = g.subgraph(p2).copy()

        # Hack to check and fix non connected subgraphs
        if not nx.is_connected(sg1):
            for nodes_cc in sorted(nx.connected_components(sg1), key=len, reverse=True)[1:]:
                sg = sg1.subgraph(nodes_cc).copy()
                p2.update(sg.nodes())
                for n in sg.nodes():
                    p1.remove(n)

            sg2 = g.subgraph(p2).copy()  # updating sg2 since p2 has changed

        if not nx.is_connected(sg2):
            for nodes_cc in sorted(nx.connected_components(sg2), key=len, reverse=True)[1:]:
                sg = sg2.subgraph(nodes_cc).copy()
                p1.update(sg.nodes())
                for n in sg.nodes():
                    p2.remove(n)

        iter_count += 1
    if iter_count > 2:
        print('it took {} iterations to stabilize'.format(iter_count))

    assert nx.is_connected(sg1) and nx.is_connected(sg2), "subgraphs are not connected in cond"

    lvl.append(approx_min_conductance_partitioning(sg1))
    lvl.append(approx_min_conductance_partitioning(sg2))

    assert (len(lvl) > 0)
    return lvl


def spectral_kmeans(g: LightMultiGraph, K: int):
    """
    k-way ncut spectral clustering Ng et al. 2002 KNSC1
    :param g: graph g
    :param K: number of clusters
    :return:
    """
    tree = []

    if g.order() <= K:   # not more than k nodes, return the list of nodes
        if g.order() == 1:
            clusters = list(g.nodes())
        else:
            clusters = [[n] for n in g.nodes()]
        return clusters

    if K == 2:  # if K is two, use approx min partitioning
        return approx_min_conductance_partitioning(g)

    if not nx.is_connected(g):
        for nodes_cc in nx.connected_components(g):
            p = g.subgraph(nodes_cc).copy()
            if p.order() > K + 1:   # if p has more than K + 1 nodes, use spectral K-means
                tree.append(spectral_kmeans(p, K))
            else:   # try spectral K-means with a lesser K
                tree.append(spectral_kmeans(p, K - 1))
        assert len(tree) > 0
        return tree

    if K >= g.order() - 2:
        return spectral_kmeans(g, K - 1)

    assert nx.is_connected(g), "g is not connected in spectral kmeans"

    L = nx.laplacian_matrix(g)

    assert K < g.order() - 2, "k is too high"

    _, eigenvecs = scipy.sparse.linalg.eigsh(L.asfptype(), k=K + 1, which='SM')  # compute the first K+1 eigenvectors
    eigenvecs = eigenvecs[:, 1:]  # discard the first trivial eigenvector

    U = sklearn.preprocessing.normalize(eigenvecs)  # normalize the eigenvecs by its L2 norm

    kmeans = KMeans(n_clusters=K).fit(U)

    cluster_labels = kmeans.labels_
    clusters = [[] for _ in range(max(cluster_labels) + 1)]

    for u, clu_u in zip(g.nodes(), cluster_labels):
        clusters[clu_u].append(u)

    for cluster in clusters:
        sg = g.subgraph(cluster).copy()
        # assert nx.is_connected(sg), "subgraph not connected"
        if len(cluster) > K + 1:
            tree.append(spectral_kmeans(sg, K))
        else:
            tree.append(spectral_kmeans(sg, K - 1))

    return tree


if __name__ == '__main__':
    g = nx.karate_club_graph()
    # l = approx_min_conductance_partitioning(g)
    K = int(math.sqrt(g.order() // 2))
    l = spectral_kmeans(g, K)
    print(l)
