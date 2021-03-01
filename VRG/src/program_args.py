"""
Program Args that get tossed around in functions
"""
from dataclasses import dataclass
from os.path import join
from typing import Any

from VRG.src.utils import nx_to_lmg


@dataclass
class ProgramArgs:
    name: str  # name of the dataset
    clustering: str  # clustering alg
    basedir: str = '/data/ssikdar/AVRG'  # basedir where the directories are made
    attr_name: str = 'value'  # each node should have a 'value' attribute

    use_cluster_pickle: bool = True  # use a pickled root by default
    use_grammar_pickle: bool = True  # use a pickled grammar
    use_graphs_pickle: bool = True  # use the set of pickled graphs

    write_grammar: bool = True  # write grammar to disk
    write_graphs: bool = True  # write graphs to disk


@dataclass
class GrammarArgs:
    program_args: ProgramArgs
    extract_type: str  # extraction type: mu-random, mu-level, all-tnodes ...
    mu: int  # size of grammar rules
    hc_obj: Any  # associated hierarchical clustering object
    grammar_type: str = 'AVRG'  # by default we want AVRGs
    grammar_info: str = ''
    grammar_filename: str = ''

    def __post_init__(self):
        self.name = self.program_args.name
        self.input_graph = self.hc_obj.input_graph
        self.clustering = self.program_args.clustering
        self.lmg = nx_to_lmg(self.input_graph)

        self.grammar_info = f'{self.grammar_type}_{self.extract_type}_{self.clustering}_{self.mu}'
        self.grammar_filename = join(self.program_args.basedir, 'output/grammars', self.name,
                                     f'{self.grammar_info}.pkl')
        return


@dataclass
class GenerationArgs:
    grammar_args: GrammarArgs
    gen_type: str  # generation type: regular, fancy, greedy-deg, greedy-attr, greedy-50
    num_graphs: int = 10  # number of graphs to generate
    graphs_info: str = ''
    graphs_filename: str = ''

    def __post_init__(self):
        self.program_args = self.grammar_args.program_args
        self.graphs_info = f'{self.gen_type}_{self.num_graphs}'
        self.graphs_filename = join(self.program_args.basedir, 'output/graphs', self.program_args.name,
                                    f'{self.grammar_args.grammar_info}_{self.graphs_info}.pkl')
        return
