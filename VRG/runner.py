import argparse
import logging
import math
import os
import sys;

from VRG.src.consensus_clustering import get_consensus_root

sys.path.extend(['../', '../../'])
from pathlib import Path

from time import time
from typing import Any, List, Union, Dict

import networkx as nx
import seaborn as sns; sns.set_style('white')

from tqdm import tqdm

import VRG.src.partitions as partitions
from VRG.src.LightMultiGraph import LightMultiGraph
from VRG.src.Tree import create_tree
from VRG.src.VRG import VRG, NCE, AttributedVRG
from VRG.src.extract import NCEExtractor, VRGExtractor, AVRGExtractor
from VRG.src.generate import RandomGenerator, NCEGenerator, AttributedRandomGenerator
from VRG.src.utils import dump_pickle, check_file_exists, load_pickle, timer

sys.setrecursionlimit(1_000_000)
# logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logging.basicConfig(level=logging.ERROR, format="%(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True


def get_graph(gname: str = 'sample') -> nx.Graph:
    start_time = time()
    if gname == 'sample':
        g = nx.Graph()
        g.add_nodes_from(range(5), color='blue')
        g.add_nodes_from(range(5, 9), color='red')

        g.add_edges_from([(0, 1), (0, 3), (0, 4),
                          (1, 2), (1, 4), (1, 5),
                          (2, 3), (2, 4), (2, 8),
                          (3, 4),
                          (5, 6), (5, 7), (5, 8),
                          (6, 7), (6, 8),
                          (7, 8)])  # properly labeled
        g.name = 'sample'
    elif gname == 'karate':
        g = nx.karate_club_graph()
    elif gname == 'BA':
        g = nx.barabasi_albert_graph(10, 2, seed=42)
        # g = nx.MultiGraph(g)
        g = nx.Graph()
    elif gname.endswith('.gpickle'):
        g = nx.read_gpickle(gname)
        g.name = Path(gname).stem
    else:
        if gname in ('waterloo', 'grenoble', 'uppsala'):
            g = nx.read_gpickle(f'../snap_data/cleaned/{gname}_lcc_attr.gpickle')
        elif gname in ('polblogs', 'polbooks', 'football'):
            g = nx.read_gml(f'./input/{gname}.gml')
        else:
            path = f'./input/{gname}.g'
            g = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())

        g.remove_edges_from(nx.selfloop_edges(g))
        if not nx.is_connected(g):
            nodes_lcc = max(nx.connected_components(g), key=len)
            g = g.subgraph(nodes_lcc).copy()
        name = g.name
        g = nx.convert_node_labels_to_integers(g, label_attribute='orig_label')
        g.name = name

    end_time = round(time() - start_time, 2)
    logging.error(f'Graph: {gname}, n = {g.order():_d}, m = {g.size():_d} read in {round(end_time, 3):_g}s.')

    return g


@timer
def get_clustering(g: nx.Graph, outdir: str, clustering: str, use_pickle: bool, max_size=None) -> Any:
    """
    wrapper method for getting dendrogram. uses an existing pickle if it can.
    :param g: graph
    :param outdir: output directory where picles are stored
    :param clustering: name of clustering method
    :param use_pickle: flag to whether or not to use the pickle
    :return: root node of the dendrogram
    """
    if g.name == 'sample':
        list_of_list_clusters = [
            [
                [[0], [1]],
                [[2], [[3], [4]]]
            ],
            [
                [[5], [6]],
                [[7], [8]]
            ]
        ]
        return list_of_list_clusters

    list_of_list_pickle = f'./{outdir}/{clustering}_list.pkl'

    if not os.path.exists(f'./{outdir}'):
        os.makedirs(f'./{outdir}')

    if check_file_exists(list_of_list_pickle) and use_pickle:
        logging.error(f'Using existing pickle for {clustering!r} clustering\n')
        list_of_list_clusters = load_pickle(list_of_list_pickle)

    else:
        tqdm.write(f'Running {clustering!r} clustering...')
        if clustering == 'random':
            list_of_list_clusters = partitions.get_random_partition(g)
        elif clustering == 'consensus':
            list_of_list_clusters = get_consensus_root(g=g, gname=g.name)
        elif clustering in ('leiden', 'louvain', 'infomap', 'labelprop'):
            assert max_size is not None
            list_of_list_clusters = partitions.louvain_leiden_infomap_label_prop(g, method=clustering, max_size=max_size)
        elif clustering == 'cond':
            list_of_list_clusters = partitions.approx_min_conductance_partitioning(g)
        elif clustering == 'spectral':
            list_of_list_clusters = partitions.spectral_kmeans(g, K=int(math.sqrt(g.order() // 2)))
        else:
            raise NotImplementedError(f'Invalid clustering algorithm {clustering!r}')
        dump_pickle(list_of_list_clusters, list_of_list_pickle)

    return list_of_list_clusters


def make_dirs(outdir: str, name: str) -> None:
    """
    Make the necessary directories
    :param outdir:
    :param name:
    :return:
    """
    subdirs = ('grammars', 'graphs', 'rule_orders', 'trees', 'grammar_stats')

    for dir in subdirs:
        dir_path = f'./{outdir}/{dir}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if dir == 'grammar_stats':
            continue
        dir_path += f'{name}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return


def get_grammars(name: str, clustering: str, grammar_type: str, mu: int, input_graph: nx.Graph, use_grammar_pickle: bool,
                 use_cluster_pickle: bool, count: int = 1, attr_name: str = '') -> List[Union[VRG, NCE]]:
    """
    Dump the stats
    :return:
    """
    outdir = 'dumps'
    make_dirs(outdir, name)  # make the directories if needed

    print(f'Extracting {count} grammars')
    grammars = []

    for i in range(count):
        grammar_filename = f'{outdir}/grammars/{name}/{grammar_type}_{clustering}_{mu}_{i}.pkl'
        if use_grammar_pickle and check_file_exists(grammar_filename):
            logging.error(f'Using pickled grammar from {grammar_filename!r}')
            grammar = load_pickle(grammar_filename)
        else:
            g_copy = input_graph.copy()
            g_copy.name = input_graph.name
            list_of_list_clusters = get_clustering(g=g_copy, outdir=f'{outdir}/trees/{name}', clustering=clustering,
                                                   use_pickle=use_cluster_pickle, max_size=mu)
            if isinstance(list_of_list_clusters, list):
                root = create_tree(list_of_list_clusters)
            else:
                root = list_of_list_clusters
            g_copy = LightMultiGraph()
            for n, d in input_graph.nodes(data=True):
                g_copy.add_node(n, **d)
            g_copy.add_edges_from(input_graph.edges(data=True))
            g_copy.name = name
            if grammar_type == 'VRG':
                extractor = VRGExtractor(g=g_copy, type='mu_random', mu=mu, root=root, clustering=clustering)
            elif grammar_type == 'NCE':
                extractor = NCEExtractor(g=g_copy, type='mu_random', mu=mu, root=root, clustering=clustering)
            elif grammar_type == 'AVRG':
                assert attr_name != ''
                extractor = AVRGExtractor(g=g_copy, attr_name=attr_name, type='mu_random', clustering=clustering,
                                          mu=mu, root=root)
            else:
                raise NotImplementedError(f'Invalid grammar type {grammar_type!r}')

            grammar = extractor.extract()
            dump_pickle(grammar, grammar_filename)
        grammars.append(grammar)
    return grammars


def generate_graphs(grammar: Union[VRG, NCE, AttributedVRG], num_graphs: int, outdir: str = 'dumps',
                    mixing_dict: Union[None, Dict] = None, attr_name: Union[str, None] = None) -> List[nx.Graph]:

    if isinstance(grammar, AttributedVRG):
        assert attr_name != ''
        gen = AttributedRandomGenerator(grammar=grammar, mixing_dict=mixing_dict, attr_name=attr_name)
    elif isinstance(grammar, VRG):
        gen = RandomGenerator(grammar=grammar)
    elif isinstance(grammar, NCE):
        gen = NCEGenerator(grammar=grammar)
    else:
        raise NotImplementedError(f'Invalid grammar type {type(grammar)!r}')

    graphs = gen.generate(num_graphs=num_graphs)

    graphs_filename = f'{outdir}/graphs/{name}/{grammar_type}_{clustering}_{mu}_{len(graphs)}.pkl'
    dump_pickle(graphs, graphs_filename)

    return graphs


def parse_args():
    clustering_algs = ['consensus', 'leiden', 'louvain', 'spectral', 'cond', 'random']
    # grammar_types = ('mu_random', )  # 'mu_level', 'mu_dl', 'mu_level_dl', 'local_dl', 'global_dl')
    grammar_types = ('VRG', 'NCE')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # formatter class shows defaults in help

    # using choices we can control the inputs. metavar='' prevents printing the choices in the help preventing clutter
    parser.add_argument('-g', '--graph', help='Name of the graph or path to gpickle file', default='karate', metavar='')
    parser.add_argument('-c', '--clustering', help='Clustering method to use', default='consensus', choices=clustering_algs,
                        metavar='')
    parser.add_argument('-m', '--mu', help='Size of RHS (mu)', default=4, type=int)
    parser.add_argument('-t', '--type', help='Grammar type', default='VRG', choices=grammar_types, metavar='')
    parser.add_argument('-o', '--outdir', help='Name of the output directory', default='output')
    parser.add_argument('-n', help='Number of graphs to generate', default=5, type=int)
    parser.add_argument('-p', '--grammar-pickle', help='Use pickled grammar?', action='store_true')
    parser.add_argument('-d', '--cluster-pickle', help='Use pickled dendrogram?', action='store_true')
    parser.add_argument('-a', '--attr-name', help='Name of Attribute', default='', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    name, attr_name, use_cluster_pickle, \
        use_grammar_pickle, clustering, grammar_type, mu, n = args.graph, args.attr_name, args.cluster_pickle,\
                                                              args.grammar_pickle, args.clustering, args.type, args.mu, args.n
    print('Command line args:', args)
    # name = 'sample'; attr_name = 'color'; mu =3; grammar_type = 'AVRG'
    name = 'polblogs'; attr_name = 'value'; mu = 60; grammar_type = 'AVRG'
    use_grammar_pickle = False; use_cluster_pickle = True; n = 10

    g = get_graph(name)
    g.name = name
    if attr_name != '':
        mix_dict = nx.attribute_mixing_dict(g, attribute=attr_name, normalized=True)
        print('Mixing dict:', mix_dict)
    else:
        mix_dict = None

    vrg = get_grammars(name=name, clustering=clustering, grammar_type=grammar_type, input_graph=g, mu=mu,
                       use_grammar_pickle=use_grammar_pickle, use_cluster_pickle=use_cluster_pickle, attr_name=attr_name)[0]

    print(vrg)
    graphs = generate_graphs(grammar=vrg, num_graphs=n, mixing_dict=mix_dict, attr_name=attr_name)
    print(graphs)
