import abc
import logging
import random
from collections import Counter
from typing import List, Dict, Tuple, Set, Union, Any

import networkx as nx
import numpy as np
from tqdm import tqdm

from VRG.src.LightMultiGraph import LightMultiGraph
from VRG.src.NonTerminal import NonTerminal
from VRG.src.Rule import VRGRule, NCERule
from VRG.src.VRG import VRG, NCE
from VRG.src.program_args import GenerationArgs
from VRG.src.utils import find_boundary_edges, timer, CustomCounter, \
    incremental_degree_assortativity, incremental_attr_assortativity


class BaseGenerator(abc.ABC):
    """
    Base class for generating graphs from a RandomVRG
    """
    __slots__ = 'grammar', 'gen_type', '_gen_graph', 'current_non_terminal_nodes', 'rule_ordering', \
                'generated_graphs', 'save_snapshots'
    allowed_strategies = 'random', 'greedy-deg', 'greedy-attr', 'greedy-50', 'mix'

    def __init__(self, gen_args: GenerationArgs) -> None:
        assert gen_args.gen_type in BaseGenerator.allowed_strategies, f'Invalid strategy {gen_args.gen_type}, ' \
                                                                      f'choose from {BaseGenerator.allowed_strategies}'
        self.gen_args = gen_args
        self.name = gen_args.name
        self.clustering = gen_args.clustering
        self.extract_type = gen_args.grammar_args.extract_type
        self.grammar: Union[VRG, NCE] = gen_args.grammar  # RandomVRG object
        self.gen_type: str = gen_args.gen_type  # strategy for generation
        self.save_snapshots = gen_args.program_args.write_snapshots

        self.current_non_terminal_nodes: Set[int] = set()  # set of non-terminal nodes in the current graph
        self.generated_graphs: List[nx.Graph] = []  # list of generated graphs
        self._gen_graph: Union[None, LightMultiGraph] = None  # generated graph
        self.rule_ordering: List[int] = []  # sequence of rule ids used to generate the graph
        self.gen_snapshots: List[LightMultiGraph] = []  # stores the individual stages for one generation
        self.all_gen_snapshots: List[List[LightMultiGraph]] = []  # stores the snapshots for all  different generations
        return

    @abc.abstractmethod
    def select_rule(self) -> Tuple[int, Union[VRGRule, NCERule], Dict]:
        """
        Selects a rule from the grammar with the matching LHS
        """
        pass

    def update_graph(self, chosen_nt_node: int, chosen_rule: Union[VRGRule, NCERule],
                     node_correspondence: Union[None, Dict] = None) -> None:
        """
        Update the current graph by
            - removing the node corresponding to the NonTerminal
            - replacing that with the RHS graph of the rule
            - rewiring the broken edges
            - node labels are carried
        """
        existing_node = chosen_nt_node
        logging.debug(f'Replacing node: {existing_node} with rule {chosen_rule}')

        chosen_nt: NonTerminal = self._gen_graph.nodes[chosen_nt_node]['nt']
        assert self._gen_graph.degree_(existing_node) == chosen_nt.size, 'Degree of non-terminal must match its size'

        broken_edges = find_boundary_edges(self._gen_graph, {existing_node})  # find the broken edges
        assert len(broken_edges) == chosen_nt.size, f'Incorrect #broken edges: {len(broken_edges)} != {chosen_nt.size}'

        node_count = max(self._gen_graph.nodes) + 1  # number of nodes in the present graph
        self._gen_graph.remove_node(existing_node)  # remove the existing node from the graph

        node_label = {}  # empty dictionary

        for node, d in chosen_rule.graph.nodes(data=True):  # add the nodes from rule
            node_label[node] = node_count

            if 'nt' in d:
                self.current_non_terminal_nodes.add(node_label[node])  # add the nonterminal node to the set
            if 'attr_dict' in d:
                d = d['attr_dict']
            self._gen_graph.add_node(node_label[node], **d)
            node_count += 1

        for u, v, d in chosen_rule.graph.edges(data=True):  # add edges from rule
            self._gen_graph.add_edge(node_label[u], node_label[v], **d)

        if len(broken_edges) != 0:
            random.shuffle(broken_edges)  # shuffle the broken edges
            # randomly joining the new boundary edges from the RHS to the rest of the graph - uniformly at random
            for node, d in chosen_rule.graph.nodes(data=True):
                num_boundary_edges = d['b_deg']
                if num_boundary_edges == 0:  # there are no boundary edges incident to that node
                    continue

                assert len(broken_edges) >= num_boundary_edges

                edge_candidates = broken_edges[: num_boundary_edges]  # picking the first num_broken edges
                broken_edges = broken_edges[num_boundary_edges:]  # removing them from future consideration

                for u, v in edge_candidates:  # each edge is either (node_sample, v) or (u, node_sample)
                    if u == existing_node:  # u is the existing node, rewire it to node
                        u = node_label[node]
                    else:
                        v = node_label[node]
                    logging.debug(f'adding broken edge ({u}, {v})')
                    self._gen_graph.add_edge(u, v)
        return

    def _generate(self, i: int):
        self._gen_graph = LightMultiGraph()  # reset the _gen_graph
        self.gen_snapshots = []  # reset the snapshot list
        g = self._gen()

        nx_g = nx.Graph()
        for k, v in g.graph.items():  # copy over the graph level attributes
            nx_g.graph[k] = v

        for n, d in g.nodes(data=True): nx_g.add_node(n, **d)
        nx_g.add_edges_from(g.edges())
        nx_g.remove_edges_from(nx.selfloop_edges(g))
        # logging.error(f'({i+1:02d}) Generated graph: n={nx_g.order():_d} m={nx_g.size():_d}')
        return nx_g, self.gen_snapshots

    @timer
    def generate(self, num_graphs: int) -> List[nx.Graph]:
        """
        Calls _gen() num_graphs many times
        :param num_graphs:
        :return:
        """
        self.generated_graphs: List[nx.Graph] = []
        # with Parallel(n_jobs=5, backend='multiprocessing') as parallel:
        #     things = parallel(delayed(self._generate)(i) for i in range(num_graphs))
        # tqdm.write(f'Generating {num_graphs} graphs for')

        with tqdm(total=num_graphs, ncols=150) as pbar:
            things = []
            for i in range(num_graphs):
                things.append(self._generate(i))
                pbar.set_description(f'{self.name!r} {self.clustering!r} {self.extract_type!r} {self.gen_type!r}')
                pbar.update(1)

        for g, snapshot in things:
            self.generated_graphs.append(g)
            self.all_gen_snapshots.append(snapshot)

        return self.generated_graphs

    @abc.abstractmethod
    def _gen(self) -> LightMultiGraph:
        """
        Generate a graph from the rules
        """
        pass


class RandomGenerator(BaseGenerator):
    def __init__(self, gen_args: GenerationArgs) -> None:
        super().__init__(gen_args=gen_args)
        return

    def select_rule(self) -> Tuple[int, VRGRule, Set[int], Dict]:
        """
        For random selection, pick based on the frequency of the rules
        returns the non-terminal node, PartRule, nodes covered
        """
        chosen_nt_node = random.sample(self.current_non_terminal_nodes, 1)[0]  # choose a non terminal node at random

        chosen_nt = self._gen_graph.nodes[chosen_nt_node]['nt']
        rule_candidates = self.grammar.rule_dict[chosen_nt.size]

        if len(rule_candidates) == 1:
            rule_idx = 0  # pick the only rule in the list
        else:
            weights = np.array([rule.frequency for rule in rule_candidates])
            weights = weights / np.sum(weights)  # normalize into probabilities
            rule_idx = int(
                np.random.choice(range(len(rule_candidates)), size=1, p=weights))  # pick based on probability

        return chosen_nt_node, rule_candidates[rule_idx], set(), {}

    def _gen(self) -> LightMultiGraph:
        starting_nt = self.grammar.rule_dict[0][0].lhs_nt

        self._gen_graph = LightMultiGraph()
        self._gen_graph.add_node(0, nt=starting_nt)
        if self.save_snapshots: self.gen_snapshots.append(self._gen_graph.copy())

        self.current_non_terminal_nodes = {0}  # first non-terminal is node 0

        while len(self.current_non_terminal_nodes) != 0:  # continue until there are non-terminals remaining
            chosen_nt_node, chosen_rule, _, _ = self.select_rule()  # throw out the set of covered nodes and node correspondence
            self.current_non_terminal_nodes.remove(chosen_nt_node)  # remove the non-terminal from the set
            self.update_graph(chosen_rule=chosen_rule, chosen_nt_node=chosen_nt_node)
            if self.save_snapshots:
                self.gen_snapshots.append(self._gen_graph.copy())  # add the current graph after update

        logging.debug(f'Generated graph: n={self._gen_graph.order():_d} m={self._gen_graph.size():_d}')
        return self._gen_graph


class AttributedRandomGenerator(RandomGenerator):
    """
    Attributed graphs generated Random style
    """

    def __init__(self, gen_args: GenerationArgs):
        super().__init__(gen_args=gen_args)
        self.mixing_dict = gen_args.inp_mixing_dict
        self.attr_name = gen_args.program_args.attr_name
        self.fancy_rewirings = 0
        self.total_rewirings = 0
        self.use_fancy_rewiring = gen_args.use_fancy_rewiring
        self.save_snapshots = gen_args.program_args.write_snapshots
        return

    def _gen(self) -> LightMultiGraph:
        self.fancy_rewirings = 0  # reset the counts
        self.total_rewirings = 0
        starting_nt = self.grammar.rule_dict[0][0].lhs_nt

        self._gen_graph = LightMultiGraph()
        self._gen_graph.add_node(0, nt=starting_nt)
        if self.save_snapshots: self.gen_snapshots.append(self._gen_graph.copy())

        self.current_non_terminal_nodes = {0}  # first non-terminal is node 0

        while len(self.current_non_terminal_nodes) != 0:  # continue until there are non-terminals remaining
            chosen_nt_node, chosen_rule, _, _ = self.select_rule()  # throw out the set of covered nodes and node correspondence
            self.current_non_terminal_nodes.remove(chosen_nt_node)  # remove the non-terminal from the set
            self.update_graph(chosen_rule=chosen_rule, chosen_nt_node=chosen_nt_node)
            if self.save_snapshots: self.gen_snapshots.append(self._gen_graph.copy())
        if self.fancy_rewirings == 0:
            fancy_frac = 0
        else:
            fancy_frac = 100 * self.fancy_rewirings/self.total_rewirings
        logging.debug(f'Generated graph: n={self._gen_graph.order():,d} m={self._gen_graph.size():,d} fancy rewirings'
                      f'({self.fancy_rewirings:,d}/{self.total_rewirings:,d}) {round(fancy_frac, 3)}%')

        self._gen_graph.graph['total_rewirings'] = self.total_rewirings
        self._gen_graph.graph['fancy_rewirings'] = self.fancy_rewirings
        return self._gen_graph

    def select_rule(self) -> Tuple[int, VRGRule, Set[int], Dict]:
        """
        For random selection, pick based on the frequency of the rules
        returns the non-terminal node, PartRule, nodes covered
        """
        chosen_nt_node = random.sample(self.current_non_terminal_nodes, 1)[0]  # choose a non terminal node at random

        chosen_nt = self._gen_graph.nodes[chosen_nt_node]['nt']
        rule_candidates = self.grammar.rule_dict[chosen_nt.size]

        if len(rule_candidates) == 1:
            rule_idx = 0  # pick the only rule in the list
        else:
            weights = np.array([rule.frequency for rule in rule_candidates])
            weights = weights / np.sum(weights)  # normalize into probabilities
            rule_idx = int(
                np.random.choice(range(len(rule_candidates)), size=1, p=weights))  # pick based on probability

        return chosen_nt_node, rule_candidates[rule_idx], set(), {}

    # @profile
    def update_graph(self, chosen_nt_node: int, chosen_rule: VRGRule,
                     node_correspondence: Union[None, Dict] = None) -> None:
        """
        Update the current graph by
            - removing the node corresponding to the NonTerminal
            - replacing that with the RHS graph of the rule
            - rewiring the broken edges
            - node labels are carried
        """
        existing_node = chosen_nt_node
        logging.debug(f'Replacing node: {existing_node} with rule {chosen_rule}')

        chosen_nt: NonTerminal = self._gen_graph.nodes[chosen_nt_node]['nt']
        assert self._gen_graph.degree_(existing_node) == chosen_nt.size, 'Degree of non-terminal must match its size'

        broken_edges = find_boundary_edges(self._gen_graph, {existing_node})  # find the broken edges
        broken_edges: Counter[tuple, int] = CustomCounter(broken_edges)  # TODO change to a Counter object to speed things up

        assert sum(broken_edges.values()) == chosen_nt.size, f'Incorrect #broken edges: {len(broken_edges)} != {chosen_nt.size}'

        self.total_rewirings += broken_edges.positive_values_len()  # all the broken edges need to be rewired

        node_count = max(self._gen_graph.nodes) + 1  # number of nodes in the present graph
        self._gen_graph.remove_node(existing_node)  # remove the existing node from the graph

        node_label = {}  # empty dictionary
        b_degs = {}  # this is needed to not modify the boundary degrees of rules in place
        for node, d in chosen_rule.graph.nodes(data=True):  # add the nodes from rule
            node_label[node] = node_count
            b_degs[node] = d['b_deg']
            if 'nt' in d:
                self.current_non_terminal_nodes.add(node_label[node])  # add the nonterminal node to the set
            if 'attr_dict' in d:
                d = d['attr_dict']
            self._gen_graph.add_node(node_label[node], **d)
            node_count += 1

        for u, v, d in chosen_rule.graph.edges(data=True):  # add edges from rule
            self._gen_graph.add_edge(node_label[u], node_label[v], **d)

        if broken_edges.positive_values_len() != 0:
            if self.use_fancy_rewiring:
                # Figure out if there are terminal nodes on both RULE and the current graph
                rule_terminals = [node for node, d in chosen_rule.graph.nodes(data=True) if 'nt' not in d]
                graph_terminal_ctr = CustomCounter()
                # for u, v in broken_edges:
                for u, v in broken_edges.elements():
                    node = v if u == existing_node else u
                    if 'nt' not in self._gen_graph.nodes[node]:
                        graph_terminal_ctr[node] += 1  # .append(node)

                if len(rule_terminals) > 0 and len(graph_terminal_ctr) > 0:
                    # this is where the probabilistic matching needs to happen
                    # rule_terminal_list = []  # we need b_deg copies of each terminal in the rules
                    rule_terminal_ctr = CustomCounter()  # we need b_deg copies of each terminal in the rules # TODO changed
                    for rule_t in rule_terminals:
                        b_deg = b_degs[rule_t]
                        if b_deg > 0:
                            rule_terminal_ctr[rule_t] = b_deg

                    # assert graph_terminal_ctr.positive_values_len() <= rule_terminal_ctr.positive_values_len()

                    # do the matching up
                    while True:
                        if graph_terminal_ctr.positive_values_len() == 0 or rule_terminal_ctr.positive_values_len() == 0:
                            break
                        # 1. pick a graph terminal at random and remove it
                        graph_t = random.choice(list(graph_terminal_ctr.keys()))

                        graph_terminal_ctr[graph_t] -= 1
                        attr_1 = self._gen_graph.nodes[graph_t][self.attr_name]

                        # 2. pick a rule terminal with probability proportional to the mixing matrix
                        wts = []
                        for rule_t in rule_terminal_ctr:
                            try:  # this weird reason why sometimes the node attributes are nested
                                attr_2 = chosen_rule.graph.nodes[rule_t][self.attr_name]
                            except KeyError:
                                attr_2 = chosen_rule.graph.nodes[rule_t]['attr_dict'][self.attr_name]
                            p = self.mixing_dict[attr_1].get(attr_2, 0)  # probability of edge  # TODO: check for bipartite graphs
                            wts.append((rule_t, p))

                        rule_t = random.choices(population=tuple(map(lambda x: x[0], wts)),
                                                weights=tuple(map(lambda x: x[1], wts)), k=1)[0]
                        rule_terminal_ctr[rule_t] -= 1  # .remove(rule_t)

                        # 3. Add the edge to gen_graph
                        logging.debug(f'Adding broken edge {(node_label[rule_t], graph_t)} the fancy way')
                        self._gen_graph.add_edge(graph_t, node_label[rule_t])  # node_label stores the actual label in gen_graph
                        self.fancy_rewirings += 1  # fancy rewiring

                        # 3. update b_deg for the rule terminal
                        b_degs[rule_t] -= 1

                        # 4. Remove the boundary edge from list of boundary edges
                        if (graph_t, existing_node) in broken_edges:
                            # broken_edges.remove((graph_t, existing_node))
                            broken_edges[(graph_t, existing_node)] -= 1
                        else:
                            # broken_edges.remove((existing_node, graph_t))
                            broken_edges[(existing_node, graph_t)] -= 1

            # continue the regular random rewiring with the remaining broken edges
            broken_edges: list = list(broken_edges.elements())  # turn broken edges back into a list
            random.shuffle(broken_edges)  # shuffle the broken edges
            # randomly joining the new boundary edges from the RHS to the rest of the graph - uniformly at random

            for node, d in chosen_rule.graph.nodes(data=True):
                num_boundary_edges = b_degs[node]
                if num_boundary_edges == 0:  # there are no boundary edges incident to that node
                    continue

                assert len(broken_edges) >= num_boundary_edges

                edge_candidates = broken_edges[: num_boundary_edges]  # picking the first num_broken edges
                broken_edges = broken_edges[num_boundary_edges:]  # removing them from future consideration

                for u, v in edge_candidates:  # each edge is either (node_sample, v) or (u, node_sample)
                    if u == existing_node:  # u is the existing node, rewire it to node
                        u = node_label[node]
                    else:
                        v = node_label[node]
                    logging.debug(f'adding broken edge ({u}, {v})')
                    self._gen_graph.add_edge(u, v)
        return


class GreedyAttributeRandomGenerator(AttributedRandomGenerator):
    def __init__(self, gen_args: GenerationArgs):
        super().__init__(gen_args=gen_args)
        self.inp_deg_ast = gen_args.inp_degree_ast  # input degree assortativity
        self.inp_attr_ast = gen_args.inp_attr_ast  # input attribute assortativity
        self.alpha = gen_args.alpha
        self.mapping = {val: i for i, val in
                        enumerate(gen_args.inp_mixing_dict.keys())}  # maps attr values to numbers from 0
        return

    # @profile
    def update_graph(self, chosen_nt_node: int, chosen_rule: VRGRule,
                     node_correspondence: Union[None, Dict] = None) -> None:
        """
        Update the current graph by
            - removing the node corresponding to the NonTerminal
            - replacing that with the RHS graph of the rule
            - rewiring the broken edges
            - node labels are carried
        """
        existing_node = chosen_nt_node
        logging.debug(f'Replacing node: {existing_node} with rule {chosen_rule}')

        chosen_nt: NonTerminal = self._gen_graph.nodes[chosen_nt_node]['nt']
        assert self._gen_graph.degree_(existing_node) == chosen_nt.size, 'Degree of non-terminal must match its size'

        broken_edges = find_boundary_edges(self._gen_graph, {existing_node})  # find the broken edges
        broken_edges: Counter[tuple, int] = CustomCounter(broken_edges)  # TODO change to a Counter object to speed things up

        assert sum(broken_edges.values()) == chosen_nt.size, f'Incorrect #broken edges: {len(broken_edges)} != {chosen_nt.size}'

        self.total_rewirings += broken_edges.positive_values_len()  # all the broken edges need to be rewired

        node_count = max(self._gen_graph.nodes) + 1  # number of nodes in the present graph
        self._gen_graph.remove_node(existing_node)  # remove the existing node from the graph

        node_label = {}  # empty dictionary
        b_degs = {}  # this is needed to not modify the boundary degrees of rules in place
        for node, d in chosen_rule.graph.nodes(data=True):  # add the nodes from rule
            node_label[node] = node_count
            b_degs[node] = d['b_deg']
            if 'nt' in d:
                self.current_non_terminal_nodes.add(node_label[node])  # add the nonterminal node to the set
            if 'attr_dict' in d:
                d = d['attr_dict']
            self._gen_graph.add_node(node_label[node], **d)
            node_count += 1

        for u, v, d in chosen_rule.graph.edges(data=True):  # add edges from rule
            self._gen_graph.add_edge(node_label[u], node_label[v], **d)

        if broken_edges.positive_values_len() != 0:
            if self.use_fancy_rewiring:
                # Figure out if there are terminal nodes on both RULE and the current graph
                rule_terminals = [node for node, d in chosen_rule.graph.nodes(data=True) if 'nt' not in d]
                graph_terminal_ctr = CustomCounter()
                # for u, v in broken_edges:
                for u, v in broken_edges.elements():
                    node = v if u == existing_node else u
                    if 'nt' not in self._gen_graph.nodes[node]:
                        graph_terminal_ctr[node] += 1  # .append(node)

                if len(rule_terminals) > 0 and len(graph_terminal_ctr) > 0:
                    # this is where the probabilistic matching needs to happen
                    # rule_terminal_list = []  # we need b_deg copies of each terminal in the rules
                    rule_terminal_ctr = CustomCounter()  # we need b_deg copies of each terminal in the rules # TODO changed
                    for rule_t in rule_terminals:
                        b_deg = b_degs[rule_t]
                        if b_deg > 0:
                            rule_terminal_ctr[rule_t] = b_deg

                    # assert graph_terminal_ctr.positive_values_len() <= rule_terminal_ctr.positive_values_len()

                    # do the matching up
                    while True:
                        if graph_terminal_ctr.positive_values_len() == 0 or rule_terminal_ctr.positive_values_len() == 0:
                            break
                        # 1. pick a graph terminal at random and remove it
                        graph_t = random.choice(list(graph_terminal_ctr.keys()))

                        graph_terminal_ctr[graph_t] -= 1

                        # 2. pick a rule terminal based on incremental assortativity
                        best_rule_terminal = None, np.inf

                        # we actually dont need to calculate assortativity mixing mat from scratch here
                        # TODO
                        attr_mixing_mat = nx.attribute_mixing_matrix(self._gen_graph, attribute=self.attr_name,
                                                                     mapping=self.mapping, normalized=False)
                        deg_mixing_mat = nx.degree_mixing_matrix(self._gen_graph, normalized=False)
                        ######

                        for rule_t in rule_terminal_ctr:
                            # calculate degree and attribute ast after edge (graph_t, node_label[rule_t])
                            u, v = graph_t, node_label[rule_t]
                            attr_u, attr_v = self._gen_graph.nodes[u][self.attr_name], self._gen_graph.nodes[v][self.attr_name]

                            new_deg_ast, new_deg_mixing_mat = incremental_degree_assortativity(g=self._gen_graph, u=u, v=v, M_curr=deg_mixing_mat)
                            new_attr_ast, new_attr_mixing_mat = incremental_attr_assortativity(g=self._gen_graph, attr_name=self.attr_name,
                                                                                               attr_u=attr_u, attr_v=attr_v, mapping=self.mapping,
                                                                                               M=attr_mixing_mat)

                            deg_ast_diff = np.abs(self.inp_deg_ast - new_deg_ast)
                            attr_ast_diff = np.abs(self.inp_attr_ast - new_attr_ast)

                            cost = -1 if np.isnan(deg_ast_diff) else self.alpha * deg_ast_diff

                            if not np.isnan(attr_ast_diff):
                                cost += (1 - self.alpha) * attr_ast_diff

                            if cost < best_rule_terminal[1]:
                                best_rule_terminal = rule_t, cost
                                best_deg_mixing_mat = new_deg_mixing_mat
                                best_attr_mixing_mat = new_attr_mixing_mat

                        rule_t = best_rule_terminal[0]
                        assert rule_t is not None
                        rule_terminal_ctr[rule_t] -= 1  # .remove(rule_t)
                        logging.debug(f'Best rule terminal {node_label[rule_t]} cost = {best_rule_terminal[1]}')

                        # 3. Add the edge to gen_graph
                        logging.debug(f'Adding broken edge {(node_label[rule_t], graph_t)} the fancy way')
                        self._gen_graph.add_edge(graph_t, node_label[rule_t])  # node_label stores the actual label in gen_graph

                        # update the mixing matrices
                        # deg_mixing_mat = best_deg_mixing_mat.copy()
                        # attr_mixing_mat = best_attr_mixing_mat.copy()

                        # assert np.allclose(nx.degree_mixing_matrix(self._gen_graph, normalized=False), deg_mixing_mat)

                        self.fancy_rewirings += 1  # fancy rewiring

                        # 3. update b_deg for the rule terminal
                        b_degs[rule_t] -= 1

                        # 4. Remove the boundary edge from list of boundary edges
                        if (graph_t, existing_node) in broken_edges:
                            # broken_edges.remove((graph_t, existing_node))
                            broken_edges[(graph_t, existing_node)] -= 1
                        else:
                            # broken_edges.remove((existing_node, graph_t))
                            broken_edges[(existing_node, graph_t)] -= 1

            # continue the regular random rewiring with the remaining broken edges
            broken_edges: list = list(broken_edges.elements())  # turn broken edges back into a list
            random.shuffle(broken_edges)  # shuffle the broken edges
            # randomly joining the new boundary edges from the RHS to the rest of the graph - uniformly at random

            for node, d in chosen_rule.graph.nodes(data=True):
                num_boundary_edges = b_degs[node]
                if num_boundary_edges == 0:  # there are no boundary edges incident to that node
                    continue

                assert len(broken_edges) >= num_boundary_edges

                edge_candidates = broken_edges[: num_boundary_edges]  # picking the first num_broken edges
                broken_edges = broken_edges[num_boundary_edges:]  # removing them from future consideration

                for u, v in edge_candidates:  # each edge is either (node_sample, v) or (u, node_sample)
                    if u == existing_node:  # u is the existing node, rewire it to node
                        u = node_label[node]
                    else:
                        v = node_label[node]
                    logging.debug(f'adding broken edge ({u}, {v})')
                    self._gen_graph.add_edge(u, v)
        return


class GreedyGenerator(BaseGenerator):
    """
    Makes sure a certain set of nodes are generated
    """
    def __init__(self, grammar: NCE, input_graph: Union[LightMultiGraph, nx.Graph], fraction: Union[None, float] = None,
                 keep_nodes: Union[Set, None] = None, save_snapshots: bool = False):
        # assert isinstance(grammar, NCE), 'Incorrect variant of grammar. Needs NCE'
        super().__init__(grammar, strategy='greedy', save_snapshots=save_snapshots)
        self.input_graph: Union[nx.Graph, LightMultiGraph] = input_graph

        if fraction is None and keep_nodes is None:
            raise NotImplementedError(f'Need either the fraction or keep_nodes')
        elif keep_nodes is not None:
            self.missing_nodes = set(keep_nodes)  # we need these nodes
        elif fraction is not None:
            if fraction == 1:
                self.missing_nodes = input_graph.nodes
            else:
                random.seed(0)
                self.missing_nodes: Set[int] = set(random.sample(input_graph.nodes,
                                                             int(fraction*input_graph.order())))
        self.keep_these_nodes = set(self.missing_nodes)  # keep a copy of the missing nodes for labeling later
        return

    def select_rule(self) -> Tuple[int, NCERule, Set]:
        """
        Pick rules that prioritizes the missing nodes - pick the ones with the most overlap
        :return:
        """
        found = False
        if len(self.missing_nodes) != 0:  # there are still nodes that are missing
            for nt_node in self.current_non_terminal_nodes:
                nt: NonTerminal = self._gen_graph.nodes[nt_node]['nt']
                nodes_covered = nt.nodes_covered
                if len(nodes_covered) > 0 and nodes_covered.issubset(self.missing_nodes):
                    chosen_nt_node = nt_node
                    chosen_nt = nt
                    logging.debug(f'nonterminal "{nt}" chosen greedily')
                    found = True
                    self.missing_nodes -= nodes_covered  # update the missing nodes
                    rule_idx = chosen_nt.id - 1
                    break
        if not found:  # nt not found, pick one at random - not truly random since it's only looking at the current non-termials
            chosen_nt_node = random.sample(self.current_non_terminal_nodes, 1)[0]
            chosen_nt = self._gen_graph.nodes[chosen_nt_node]['nt']

            logging.debug(f'nonterminal "{chosen_nt}" chosen at random')
            rule_candidates = self.grammar.rule_dict[chosen_nt.size]

            if len(rule_candidates) == 1:
                rule_idx = rule_candidates[0].id - 1  # pick the only rule in the list
            else:
                rule_idx = random.choice(rule_candidates).id - 1  # pick based on probability

        chosen_rule = self.grammar.rule_list[rule_idx]
        assert chosen_rule.lhs_nt.size == chosen_nt.size, 'Improper node selection'
        return chosen_nt_node, chosen_rule, chosen_rule.nodes_covered

    def update_graph(self, chosen_nt_node: int, chosen_rule: NCERule,
                     node_correspondence: Union[None, Dict] = None) -> None:
        """
        Update the current graph by
            - removing the node corresponding to the NonTerminal
            - replacing that with the RHS graph of the rule
            - rewiring the broken edges
            - node labels are carried
        """
        existing_node = chosen_nt_node
        logging.debug(f'Replacing node: {existing_node} with rule {chosen_rule}')

        chosen_nt: NonTerminal = self._gen_graph.nodes[chosen_nt_node]['nt']
        assert self._gen_graph.degree_(existing_node) == chosen_nt.size, 'Degree of non-terminal must match its size'

        broken_edges = find_boundary_edges(self._gen_graph, {existing_node})  # find the broken edges
        assert len(broken_edges) == chosen_nt.size, f'Incorrect #broken edges: {len(broken_edges)} != {chosen_nt.size}'

        self._gen_graph.remove_node(existing_node)  # remove the existing node from the graph

        node_label = {}  # empty dictionary

        for node, d in chosen_rule.graph.nodes(data=True):  # add the nodes from rule
            if node in self.keep_these_nodes:
                label = node
            else:
                label = f'{node}_'  # add underscores to the nodes which are not in the keep list
            while self._gen_graph.has_node(label):  # necessary since now the same rule can be invoked multiple times - causing two nodes to be labeled same
                label = f'{label}_'
            node_label[node] = label
            if 'nt' in d:
                self.current_non_terminal_nodes.add(node_label[node])  # add the nonterminal node to the set
            self._gen_graph.add_node(node_label[node], **d)

        for u, v, d in chosen_rule.graph.edges(data=True):  # add edges from rule
            self._gen_graph.add_edge(node_label[u], node_label[v], **d)

        if len(broken_edges) != 0:
            random.shuffle(broken_edges)  # shuffle the broken edges
            # randomly joining the new boundary edges from the RHS to the rest of the graph - uniformly at random
            for node, d in chosen_rule.graph.nodes(data=True):
                num_boundary_edges = d['b_deg']
                if num_boundary_edges == 0:  # there are no boundary edges incident to that node
                    continue

                assert len(broken_edges) >= num_boundary_edges

                edge_candidates = broken_edges[: num_boundary_edges]  # picking the first num_broken edges
                broken_edges = broken_edges[num_boundary_edges:]  # removing them from future consideration

                for u, v in edge_candidates:  # each edge is either (node_sample, v) or (u, node_sample)
                    if u == existing_node:  # u is the existing node, rewire it to node
                        u = node_label[node]
                    else:
                        v = node_label[node]
                    # logging.debug(f'adding broken edge ({u}, {v})')
                    self._gen_graph.add_edge(u, v)
        return

    def _gen(self) -> LightMultiGraph:
        starting_nt = self.grammar.rule_dict[0][0].lhs_nt
        self.missing_nodes = set(self.keep_these_nodes)  # reset the missing nodes for each gen
        self._gen_graph = LightMultiGraph()
        self._gen_graph.add_node(0, nt=starting_nt)

        self.current_non_terminal_nodes = {0}  # first non-terminal is node 0

        while len(self.current_non_terminal_nodes) != 0:  # continue until there are non-terminals remaining
            chosen_nt_node, chosen_rule, nodes_covered = self.select_rule()
            self.current_non_terminal_nodes.remove(chosen_nt_node)  # remove the non-terminal from the set

            logging.debug(f'Selected nt_node: {chosen_nt_node}, rule: {chosen_rule}, nodes: {nodes_covered}, missing: {self.missing_nodes}')

            # update the node correspondences here
            self.update_graph(chosen_rule=chosen_rule, chosen_nt_node=chosen_nt_node)

        colors = {}
        for n in self._gen_graph.nodes():
            if n in self.keep_these_nodes:
                colors[n] = 'red'
            else:
                colors[n] = 'blue'
        nx.set_node_attributes(self._gen_graph, name='colors', values=colors)

        return self._gen_graph


class EnsureAllNodesGenerator(GreedyGenerator):
    """
    Use the select rule policy from NCE generator but keep the rewiring random
    """
    def select_rule(self) -> Tuple[int, NCERule, Set, Any]:
        """
        Pick rules that prioritizes the missing nodes - pick the ones with the most overlap
        :return:
        """
        chosen_nt_node = max(self.current_non_terminal_nodes,
                             key=lambda nt_node: self._gen_graph.nodes[nt_node]['nt'].id)  # choose the nt with max id
        logging.debug(f'Picking nt: {chosen_nt_node}')
        chosen_nt: NonTerminal = self._gen_graph.nodes[chosen_nt_node]['nt']
        rule_idx = chosen_nt.id - 1

        chosen_rule = self.grammar.rule_list[rule_idx]
        assert chosen_rule.lhs_nt.size == chosen_nt.size, 'Non-terminal size mismatch'

        return chosen_nt_node, chosen_rule, chosen_rule.nodes_covered, None


class AttributedEnsureAllNodesGenerator(AttributedRandomGenerator):
    def select_rule(self) -> Tuple[int, VRGRule, Set[int], Any]:
        """
        Pick rules that prioritizes the missing nodes - pick the ones with the most overlap
        :return:
        """
        chosen_nt_node = max(self.current_non_terminal_nodes,
                             key=lambda nt_node: self._gen_graph.nodes[nt_node]['nt'].id)  # choose the nt with max id
        logging.debug(f'Picking nt: {chosen_nt_node}')
        chosen_nt: NonTerminal = self._gen_graph.nodes[chosen_nt_node]['nt']
        rule_idx = chosen_nt.id - 1

        chosen_rule = self.grammar.rule_list[rule_idx]
        assert chosen_rule.lhs_nt.size == chosen_nt.size, 'Non-terminal size mismatch'

        return chosen_nt_node, chosen_rule, chosen_rule.nodes_covered, None


class GreedyAttributedEnsureAllNodesGenerator(GreedyAttributeRandomGenerator):
    def select_rule(self) -> Tuple[int, VRGRule, Set[int]]:
        """
        Pick rules that prioritizes the missing nodes - pick the ones with the most overlap
        :return:
        """
        chosen_nt_node = max(self.current_non_terminal_nodes,
                             key=lambda nt_node: self._gen_graph.nodes[nt_node]['nt'].id)  # choose the nt with max id
        logging.debug(f'Picking nt: {chosen_nt_node}')
        chosen_nt: NonTerminal = self._gen_graph.nodes[chosen_nt_node]['nt']
        rule_idx = chosen_nt.id - 1

        chosen_rule = self.grammar.rule_list[rule_idx]
        assert chosen_rule.lhs_nt.size == chosen_nt.size, 'Non-terminal size mismatch'

        return chosen_nt_node, chosen_rule, chosen_rule.nodes_covered, None


class NCEGenerator(BaseGenerator):
    """
    NCE generator for guaranteeing isomorphism
    """
    def __init__(self, grammar: Union[VRG, NCE], strategy: str='random'):
        super().__init__(grammar, strategy=strategy)
        return

    def _gen(self) -> LightMultiGraph:
        starting_nt = self.grammar.rule_dict[0][0].lhs_nt

        self._gen_graph = LightMultiGraph()
        self._gen_graph.add_node(0, nt=starting_nt)
        self.current_non_terminal_nodes = {0}  # first non-terminal is node 0
        self.exhausted_rules = set()  # reset

        while len(self.current_non_terminal_nodes) != 0:  # continue until there are non-terminals remaining
            chosen_nt_node, chosen_rule, nodes_covered = self.select_rule()
            self.current_non_terminal_nodes.remove(chosen_nt_node)  # remove the non-terminal from the set

            logging.debug(
                f'Selected nt_node: {chosen_nt_node}, rule: {chosen_rule}, nodes: {nodes_covered}')

            self.update_graph(chosen_rule=chosen_rule, chosen_nt_node=chosen_nt_node)

        logging.debug(f'Generated graph: n={self._gen_graph.order()} m={self._gen_graph.size()}')
        return self._gen_graph

    def select_rule(self) -> Tuple[int, NCERule, Set]:
        """
        Select a rule from grammar by matching the boundary nodes
        :return:
        """
        chosen_nt_node = max(self.current_non_terminal_nodes,
                             key=lambda nt_node: self._gen_graph.nodes[nt_node]['nt'].id)  # choose the non-terminal with max id
        logging.debug(f'Picking nt: {chosen_nt_node}')
        chosen_nt: NonTerminal = self._gen_graph.nodes[chosen_nt_node]['nt']
        rule_idx = chosen_nt.id - 1

        chosen_rule = self.grammar.rule_list[rule_idx]
        assert chosen_rule.lhs_nt.size == chosen_nt.size, 'Non-terminal size mismatch'

        return chosen_nt_node, chosen_rule, chosen_rule.nodes_covered

    def update_graph(self, chosen_nt_node: int, chosen_rule: NCERule, node_correspondence: Union[None, Dict] = None) -> None:
        """
        Update the current graph by
            - removing the node corresponding to the NonTerminal
            - replacing that with the RHS graph of the rule
            - rewiring the broken edges
            - node labels are carried
        """
        existing_node = chosen_nt_node
        logging.debug(f'Replacing node: {existing_node} with rule {chosen_rule}')

        chosen_nt: NonTerminal = self._gen_graph.nodes[chosen_nt_node]['nt']
        assert self._gen_graph.degree_(existing_node) == chosen_nt.size, 'Degree of non-terminal must match its size'

        broken_edges = find_boundary_edges(self._gen_graph, {existing_node})  # find the broken edges
        assert len(broken_edges) == chosen_nt.size, f'Incorrect #broken edges: {len(broken_edges)} != {chosen_nt.size}'

        self._gen_graph.remove_node(existing_node)  # remove the existing node from the graph

        for node, d in chosen_rule.graph.nodes(data=True):  # add the nodes from rule
            if 'nt' in d:
                self.current_non_terminal_nodes.add(node)  # add the nonterminal node to the set
            self._gen_graph.add_node(node, **d)

        for u, v, d in chosen_rule.graph.edges(data=True):  # add edges from rule
            assert self._gen_graph.has_node(u) and self._gen_graph.has_node(v), 'Graph does not have the reqd nodes'
            self._gen_graph.add_edge(u, v, **d)

        for u, v in chosen_rule.boundary_edges:
            assert self._gen_graph.has_node(u) and self._gen_graph.has_node(v), 'Graph does not have the reqd nodes'
            # logging.debug(f'adding broken edge ({u}, {v})')
            self._gen_graph.add_edge(u, v)

        return


def get_node_correspondence(rule: VRGRule, nodes: Set[int]) -> Dict:
    """
    Returns the node correspondence b/w terminals and actual labels
    :param rule: chosen rule
    :param nodes: set of the nodes covered by the rule
    :return:
    """
    node_correspondence = {}  # stores the mapping between the rules and the actual labels in the input graph
    for node, d in rule.graph.nodes(data=True):
        if 'nt' not in d:  # it is a terminal node, it should have an actual label
            for label in d['actual_labels']:  # run thru the labels
                if label in nodes:
                    node_correspondence[node] = label
    return node_correspondence
