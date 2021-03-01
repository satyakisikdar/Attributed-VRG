import logging
import os
import sys
from os.path import join
from pathlib import Path

import networkx as nx

from VRG.src.HierClustering import HierarchicalClustering
from VRG.src.extract import AVRGExtractor
from VRG.src.graph_io import GraphReader
from VRG.src.program_args import ProgramArgs, GrammarArgs
from VRG.src.utils import check_file_exists, load_pickle, dump_pickle

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


def read_graph(name: str, path: str = '', basedir: str = '/data/ssikdar/Attributed-VRG') -> nx.Graph:
    if path == '':
        path = join(basedir, 'input', f'{name}.gml')
    greader = GraphReader(filename=path, gname=name, reindex_nodes=True)
    return greader.graph


def ensure_dirs(prog_args: ProgramArgs):
    subdirs = 'grammars', 'trees', 'graphs'

    for dir_ in subdirs:
        dir_path = Path(join(prog_args.basedir, 'output', dir_))
        if not dir_path.exists():
            logging.error(f'Making new directory: {dir_path!r}')
            os.makedirs(dir_path, exist_ok=True)

        name_dir_path = dir_path / prog_args.name
        if not name_dir_path.exists():
            logging.error(f'Making new directory: {name_dir_path.stem!r}')
            os.makedirs(name_dir_path, exist_ok=True)
    return


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


def main():
    name = 'wisconsin'
    clustering = 'cond'
    input_graph = read_graph(name=name)
    prog_args = ProgramArgs(name=name, clustering=clustering)

    ensure_dirs(prog_args)

    hc = HierarchicalClustering(prog_args=prog_args, input_graph=input_graph)
    root = hc.get_clustering()
    hc.calculate_cost()
    print(hc.stats)

    extract_type = 'mu-level'
    mu = 5
    grammar_args = GrammarArgs(program_args=prog_args, extract_type=extract_type, hc_obj=hc, mu=mu)
    print(grammar_args)
    grammar = get_grammar(gram_args=grammar_args)
    print(grammar)
    return


if __name__ == '__main__':
    main()
