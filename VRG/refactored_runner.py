from os.path import join

import networkx as nx

from VRG.src.HierClustering import HierarchicalClustering
from VRG.src.graph_io import GraphReader


def read_graph(name: str, path: str = '', basedir: str = '/data/ssikdar/Attributed-VRG') -> nx.Graph:
    if path == '':
        path = join(basedir, 'input', f'{name}.gml')
    greader = GraphReader(filename=path, gname=name, reindex_nodes=True)
    return greader.graph


def main():
    basedir = '/data/ssikdar/Attributed-VRG'
    name = 'wisconsin'
    clustering = 'leiden'
    input_graph = read_graph(name=name)
    hc = HierarchicalClustering(input_nx_graph=input_graph, clustering=clustering, name=name)
    root = hc.get_clustering(use_pickle=False)
    print(hc.stats)
    return


if __name__ == '__main__':
    main()
