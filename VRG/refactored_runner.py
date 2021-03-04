import logging
import os
import sys
from os.path import join
from pathlib import Path

import networkx as nx

from VRG.src.HierClustering import HierarchicalClustering
from VRG.src.extract import AVRGExtractor
from VRG.src.generate import GreedyAttributeRandomGenerator, AttributedRandomGenerator
from VRG.src.graph_io import GraphReader
from VRG.src.program_args import ProgramArgs, GrammarArgs, GenerationArgs
from VRG.src.utils import check_file_exists, load_pickle, dump_pickle

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


def read_graph(name: str, path: str = '', basedir: str = '/data/ssikdar/Attributed-VRG') -> nx.Graph:
    if path == '':
        path = join(basedir, 'input', f'{name}.gml')
    greader = GraphReader(filename=path, gname=name, reindex_nodes=True)
    return greader.graph


def ensure_dirs(basedir: str, name: str):
    subdirs = 'grammars', 'trees', 'graphs', 'snapshots', 'jsons'

    for dir_ in subdirs:
        dir_path = Path(join(basedir, 'output', dir_))
        if not dir_path.exists():
            logging.error(f'Making new directory: {dir_path!r}')
            os.makedirs(dir_path, exist_ok=True)

        name_dir_path = dir_path / name
        if not name_dir_path.exists():
            logging.error(f'Making new directory: {name_dir_path.stem!r}')
            os.makedirs(name_dir_path, exist_ok=True)
    return


def get_clustering(prog_args: ProgramArgs, input_graph: nx.Graph) -> HierarchicalClustering:
    assert prog_args.clustering != '', f'Clustering not properly set in ProgramArgs'
    hc = HierarchicalClustering(prog_args=prog_args, input_graph=input_graph)
    hc.get_clustering()
    # hc.calculate_cost()
    print(hc.stats)
    return hc


def get_grammar(gram_args: GrammarArgs):
    """
    Extract a grammar based on the parameters grammar args
    :param gram_args:
    :return:
    """
    assert gram_args.grammar_filename != '', f'Grammar filename is empty'
    prog_args = gram_args.program_args

    # use existing grammar
    if prog_args.use_grammar_pickle and check_file_exists(gram_args.grammar_filename):
        logging.error(f'Using pickled grammar for {prog_args.name!r} {prog_args.clustering!r}')
        grammar = load_pickle(gram_args.grammar_filename)
        return grammar

    # generate new grammar
    # 1. Get dendrogram from the grammar_args.hc object
    if gram_args.hc_obj.root is None:
        gram_args.hc_obj.root = gram_args.hc_obj.get_clustering()

    assert gram_args.hc_obj.root is not None, 'Clustering failed'

    # 2. extract the grammar
    if gram_args.grammar_type == 'AVRG':
        extractor = AVRGExtractor(grammar_args=gram_args)
    else:
        raise NotImplementedError('Only AVRGs for now')
    grammar = extractor.extract()

    # 3. write it to disk
    if prog_args.write_grammar:
        dump_pickle(grammar, gram_args.grammar_filename)
    return grammar


def get_graphs(gen_args: GenerationArgs):
    assert gen_args.graphs_filename != '', 'Empty graphs filename'

    prog_args = gen_args.program_args
    if prog_args.use_graphs_pickle and check_file_exists(gen_args.graphs_filename):
        if prog_args.write_snapshots and check_file_exists(gen_args.snapshots_filename):
            logging.error(f'Both graphs and snapshots found! {prog_args.name!r} {prog_args.clustering!r}')
            return load_pickle(gen_args.graphs_filename)
        else:
            logging.error(f'Using pickled graphs for {prog_args.name!r} {prog_args.clustering!r}')
            return load_pickle(gen_args.graphs_filename)

    if gen_args.inp_mixing_dict is None:
        gen_args.inp_mixing_dict = nx.attribute_mixing_dict(gen_args.input_graph, attribute=prog_args.attr_name)

    if 'greedy' in gen_args.gen_type:
        if gen_args.inp_degree_ast is None:
            gen_args.inp_degree_ast = nx.degree_assortativity_coefficient(gen_args.input_graph)
        if gen_args.inp_attr_ast is None:
            gen_args.inp_attr_ast = nx.attribute_assortativity_coefficient(gen_args.input_graph,
                                                                           attribute=prog_args.attr_name)

        if gen_args.gen_type == 'greedy-deg':
            gen_args.alpha = 1
        elif gen_args.gen_type == 'greedy-attr':
            gen_args.alpha = 0
        else:
            assert gen_args.gen_type == 'greedy-50', f'invalid greedy gen_type: {gen_args.gen_type!r}'
            gen_args.alpha = 0.5

        gen = GreedyAttributeRandomGenerator(gen_args=gen_args)

    elif gen_args.gen_type in ('mix', 'random'):
        gen = AttributedRandomGenerator(gen_args=gen_args)
    else:
        raise NotImplementedError(f'Invalid gen type {gen_args.gen_type!r}')

    graphs = gen.generate(num_graphs=gen_args.num_graphs)

    if prog_args.write_graphs:
        dump_pickle(graphs, gen_args.graphs_filename)
    if prog_args.write_snapshots:
        dump_pickle(gen.all_gen_snapshots, gen_args.snapshots_filename)
    return graphs


def main():
    name = 'karate'
    clustering = 'cond'

    input_graph = read_graph(name=name)
    prog_args = ProgramArgs(name=name, clustering=clustering)

    prog_args.use_graphs_pickle = True
    prog_args.write_snapshots = True
    ensure_dirs(basedir=prog_args.basedir, name=name)

    hc = get_clustering(prog_args=prog_args, input_graph=input_graph)

    extract_type = 'mu-level'
    mu = 5
    grammar_args = GrammarArgs(program_args=prog_args, extract_type=extract_type, hc_obj=hc, mu=mu)
    print(grammar_args)
    grammar = get_grammar(gram_args=grammar_args)
    print(grammar)

    gen_type = 'random'
    gen_args = GenerationArgs(grammar_args=grammar_args, grammar=grammar, gen_type=gen_type)
    graphs = get_graphs(gen_args=gen_args)

    snapshots = load_pickle(gen_args.snapshots_filename)[0]  # load one
    write_snapshot_jsons(snapshots, gen_args=gen_args)
    return


def write_snapshot_jsons(graphs, gen_args):
    import json

    for i, g in enumerate(graphs):
        snapshots_json_filename = join(gen_args.program_args.basedir, 'output/jsons', gen_args.name,
                                       f'graph_{i + 1}.json')

        processed_graph = nx.Graph()
        for n, data in g.nodes(data=True):
            d = {}
            if 'attr_dict' in data:
                data = data['attr_dict']
            if 'nt' in data:
                d['nt'] = data['nt'].size
            if 'value' in data:
                d['value'] = data['value']
            processed_graph.add_node(n, **d)

        for u, v, data in g.edges(data=True):
            d = {'weight': g[u][v]['weight']}
            processed_graph.add_edge(u, v, **d)

        processed_graph = nx.convert_node_labels_to_integers(processed_graph)
        d = nx.node_link_data(processed_graph)
        json.dump(d, open(snapshots_json_filename, 'w'), indent=4)
    return


if __name__ == '__main__':
    main()
