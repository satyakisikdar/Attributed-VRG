"""
Refactored batch runner -- clustering, grammar extraction, graph generation
"""
import logging
import sys
from glob import glob
from os.path import join
from pathlib import Path
from typing import List, Iterable

from VRG.src.VRG import AttributedVRG
from VRG.src.utils import load_pickle

sys.path.extend(['../', '../../', './', '../../../', '/home/ssikdar/tmp_dir'])

from VRG.refactored_runner import read_graph, ensure_dirs, get_clustering, get_grammar, get_graphs
from VRG.src.parallel import parallel_async
from VRG.src.program_args import ProgramArgs, GrammarArgs, GenerationArgs

sys.setrecursionlimit(1_000_000)


def run_batch_clusters(basedir: str, names: List[str], clusterings: List[str], num_workers: int = 8,
                       overwrite: bool = False):
    args_list = []

    skip_list = []  # skip reading the datasets which are already done
    if not overwrite:
        for name in names:
            skip = True
            for clustering in clusterings:
                root_pickle_path = join(basedir, 'output', 'trees', name, f'{clustering}_root.pkl')
                if not Path(root_pickle_path).exists():  # all the clusterings must exist to skip the whole dataset
                    skip = False
                    break
            if skip:
                skip_list.append(name)

    for name in names:
        if name in skip_list:
            print(f'Skipping {name!r}', end='\t', flush=True)
            continue

        ensure_dirs(basedir=basedir, name=name)
        input_graph = read_graph(name=name)

        for clustering in clusterings:
            prog_args = ProgramArgs(name=name, clustering=clustering)
            root_pickle_path = join(basedir, 'output', 'trees', name, f'{clustering}_root.pkl')

            if not overwrite and Path(root_pickle_path).exists():  # skip over the clusterings that are already computed
                continue

            args_list.append((prog_args, input_graph))

    try:
        parallel_async(func=get_clustering, args=args_list, num_workers=num_workers)
    except Exception as e:
        logging.error(e)
    return


def run_batch_grammars(basedir: str, names: List[str], clusterings: List[str], mus: Iterable, num_workers: int = 8,
                       overwrite: bool = False):
    extract_types = 'mu-level', 'mu-random'  # , 'all-tnodes'

    grammar_args_list = []
    for name in names:
        input_graph = read_graph(name=name, basedir=basedir)
        for clustering in clusterings:
            prog_args = ProgramArgs(name=name, clustering=clustering)
            hc = get_clustering(prog_args=prog_args, input_graph=input_graph)
            if hc.root is None:
                logging.error(f'Error in {clustering!r} alg for {name!r}. Skipping!')
                continue

            for extract_type in extract_types:
                for mu in mus:
                    grammar_args = GrammarArgs(program_args=prog_args, hc_obj=hc, extract_type=extract_type, mu=mu)
                    if overwrite and Path(grammar_args.grammar_filename).exists():
                        continue
                    grammar_args_list.append((grammar_args,))  # has to be a tuple

            # make one for the all-tnodes
            grammar_args = GrammarArgs(program_args=prog_args, hc_obj=hc, extract_type='all-tnodes', mu=0)
            if overwrite and Path(grammar_args.grammar_filename).exists():
                continue
            grammar_args_list.append((grammar_args,))  # has to be a tuple

    try:
        parallel_async(func=get_grammar, args=grammar_args_list, num_workers=num_workers)
    except Exception as e:
        print(e)

    return


def run_batch_generations(basedir: str, names: List[str], clusterings: List[str], mus: Iterable, num_workers: int = 8,
                          overwrite: bool = False):
    # run generation only for existing grammars
    # get grammar_args from the pickled grammars
    gen_args_list = []
    gen_types = 'mix', 'random', 'greedy-deg', 'greedy-50', 'greedy-attr'

    grammar_dir = join(basedir, 'output/grammars')
    for name in names:
        for grammar_filename in glob(f'{grammar_dir}/{name}/*.pkl'):
            grammar: AttributedVRG = load_pickle(grammar_filename)
            if grammar is None:
                continue

            grammar_args: GrammarArgs = grammar.grammar_args

            for gen_type in gen_types:
                gen_args = GenerationArgs(gen_type=gen_type, grammar=grammar, grammar_args=grammar_args)
                if not overwrite and Path(gen_args.graphs_filename).exists():
                    logging.error(f'Skipping {name!r} {grammar.clustering!r} {gen_type!r}')
                    continue
                gen_args_list.append((gen_args,))  # has to be a tuple

    parallel_async(func=get_graphs, args=gen_args_list, num_workers=num_workers)
    return


def main():
    basedir = '/data/ssikdar/AVRG'
    names = ['football', 'polbooks', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer',
             'polblogs', 'airports', 'film', 'chameleon', 'squirrel', 'pubmed'][: 3]

    clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random']
    mus = range(3, 5)

    # run_batch_clusters(names=names, clusterings=clusterings, num_workers=4, basedir=basedir)
    # run_batch_grammars(basedir=basedir, names=names, clusterings=clusterings, mus=mus, num_workers=4, overwrite=False)
    run_batch_generations(basedir=basedir, names=names, clusterings=clusterings, mus=mus, num_workers=4,
                          overwrite=False)
    return


if __name__ == '__main__':
    main()
