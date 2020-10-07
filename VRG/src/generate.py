import abc
import logging
import math
import random
from typing import List, Dict, Tuple, Set, Union

import networkx as nx
import numpy as np

from VRG.src.LightMultiGraph import LightMultiGraph
from VRG.src.NonTerminal import NonTerminal
from VRG.src.Rule import VRGRule, NCERule
from VRG.src.VRG import VRG, NCE
from VRG.src.utils import find_boundary_edges, timer


class BaseGenerator(abc.ABC):
    """
    Base class for generating graphs from a RandomVRG
    """
    __slots__ = 'grammar', 'strategy', '_gen_graph', 'current_non_terminal_nodes', 'rule_ordering',\
                'generated_graphs',
    allowed_strategies = 'random', 'greedy'

    def __init__(self, grammar: Union[VRG, NCE], strategy: str) -> None:
        assert strategy in BaseGenerator.allowed_strategies, f'Invalid strategy {strategy}, ' \
                                                             f'choose from {BaseGenerator.allowed_strategies}'
        self.grammar: Union[VRG, NCE] = grammar  # RandomVRG object
        self.strategy: str = strategy  # strategy for generation
        self.current_non_terminal_nodes: Set[int] = set()  # set of non-terminal nodes in the current graph
        self.generated_graphs: List[nx.Graph] = []  # list of generated graphs
        self._gen_graph: Union[None, LightMultiGraph] = None  # generated graph
        self.rule_ordering: List[int] = []  # sequence of rule ids used to generate the graph
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

    @timer
    def generate(self, num_graphs: int) -> List[nx.Graph]:
        """
        Calls _gen() num_graphs many times
        :param num_graphs:
        :return:
        """
        self.generated_graphs: List[nx.Graph] = []

        for i in range(num_graphs):
            self._gen_graph = LightMultiGraph()  # reset the _gen_graph
            g = self._gen()
            # nx_g = nx.Graph(g)  # turn the graph into NetworkX graph
            nx_g = nx.Graph()
            for k, v in g.graph.items():  # copy over the graph level attributes
                nx_g.graph[k] = v

            for n, d in g.nodes(data=True):
                nx_g.add_node(n, **d)
            nx_g.add_edges_from(g.edges())
            nx_g.remove_edges_from(nx.selfloop_edges(g))
            # logging.error(f'({i+1:02d}) Generated graph: n={nx_g.order():_d} m={nx_g.size():_d}')
            self.generated_graphs.append(nx_g)

        return self.generated_graphs

    @abc.abstractmethod
    def _gen(self) -> LightMultiGraph:
        """
        Generate a graph from the rules
        """
        pass


class RandomGenerator(BaseGenerator):
    def __init__(self, grammar: VRG) -> None:
        super().__init__(grammar=grammar, strategy='random')

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
        self.current_non_terminal_nodes = {0}  # first non-terminal is node 0

        while len(self.current_non_terminal_nodes) != 0:  # continue until there are non-terminals remaining
            chosen_nt_node, chosen_rule, _, _ = self.select_rule()  # throw out the set of covered nodes and node correspondence
            self.current_non_terminal_nodes.remove(chosen_nt_node)  # remove the non-terminal from the set
            self.update_graph(chosen_rule=chosen_rule, chosen_nt_node=chosen_nt_node)

        logging.debug(f'Generated graph: n={self._gen_graph.order():_d} m={self._gen_graph.size():_d}')
        return self._gen_graph


class AttributedRandomGenerator(RandomGenerator):
    """
    Attributed graphs generated Random style
    """
    def __init__(self, grammar: VRG, mixing_dict: Dict, attr_name: str):
        super().__init__(grammar)
        self.mixing_dict = mixing_dict
        self.attr_name = attr_name
        self.fancy_rewirings = 0
        self.total_rewirings = 0
        return

    def _gen(self) -> LightMultiGraph:
        self.fancy_rewirings = 0  # reset the counts
        self.total_rewirings = 0
        starting_nt = self.grammar.rule_dict[0][0].lhs_nt

        self._gen_graph = LightMultiGraph()
        self._gen_graph.add_node(0, nt=starting_nt)
        self.current_non_terminal_nodes = {0}  # first non-terminal is node 0

        while len(self.current_non_terminal_nodes) != 0:  # continue until there are non-terminals remaining
            chosen_nt_node, chosen_rule, _, _ = self.select_rule()  # throw out the set of covered nodes and node correspondence
            self.current_non_terminal_nodes.remove(chosen_nt_node)  # remove the non-terminal from the set
            self.update_graph(chosen_rule=chosen_rule, chosen_nt_node=chosen_nt_node)

        if self.fancy_rewirings == 0:
            fancy_frac = 0
        else:
            fancy_frac = 100 * self.fancy_rewirings/self.total_rewirings
        logging.error(f'Generated graph: n={self._gen_graph.order():_d} m={self._gen_graph.size():_d} fancy rewirings'
                      f'({self.fancy_rewirings:_d} / {self.total_rewirings:_d}) {round(fancy_frac, 3)}%')

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
        assert len(broken_edges) == chosen_nt.size, f'Incorrect #broken edges: {len(broken_edges)} != {chosen_nt.size}'

        self.total_rewirings += len(broken_edges)  # all the broken edges need to be rewired

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

        if len(broken_edges) != 0:
            # Figure out if there are terminal nodes on both RULE and the current graph
            rule_terminals = [node for node, d in chosen_rule.graph.nodes(data=True) if 'nt' not in d]
            graph_terminal_list = []
            for u, v in broken_edges:
                node = v if u == existing_node else u
                if 'nt' not in self._gen_graph.nodes[node]:
                    graph_terminal_list.append(node)

            if len(rule_terminals) > 0 and len(graph_terminal_list) > 0:
                # this is where the probabilistic matching needs to happen
                rule_terminal_list = []  # we need b_deg copies of each terminal in the rules
                for rule_t in rule_terminals:
                    b_deg = b_degs[rule_t]
                    if b_deg > 0:
                        rule_terminal_list.extend([rule_t] * b_deg)

                assert len(graph_terminal_list) >= len(graph_terminal_list)

                # do the matching up
                while True:
                    if len(graph_terminal_list) == 0 or len(rule_terminal_list) == 0:
                        break
                    # 1. pick a graph terminal at random and remove it
                    graph_t = random.choice(graph_terminal_list)

                    graph_terminal_list.remove(graph_t)
                    attr_1 = self._gen_graph.nodes[graph_t][self.attr_name]

                    # 2. pick a rule terminal with probability proportional to the mixing matrix
                    wts = []
                    for rule_t in rule_terminal_list:
                        try:  # this weird reason why sometimes the node attributes are nested
                            attr_2 = chosen_rule.graph.nodes[rule_t][self.attr_name]
                        except KeyError:
                            attr_2 = chosen_rule.graph.nodes[rule_t]['attr_dict'][self.attr_name]
                        p = self.mixing_dict[attr_1].get(attr_2, 0)  # probability of edge  # TODO: check for bipartite graphs
                        wts.append(p)

                    rule_t = random.choices(rule_terminal_list, weights=wts, k=1)[0]
                    rule_terminal_list.remove(rule_t)

                    # 3. Add the edge to gen_graph
                    logging.debug(f'Adding broken edge {(node_label[rule_t], graph_t)} the fancy way')
                    self._gen_graph.add_edge(graph_t, node_label[rule_t])  # node_label stores the actual label in gen_graph
                    self.fancy_rewirings += 1  # fancy rewiring

                    # 3. update b_deg for the rule terminal
                    b_degs[rule_t] -= 1

                    # 4. Remove the boundary edge from list of boundary edges
                    if (graph_t, existing_node) in broken_edges:
                        broken_edges.remove((graph_t, existing_node))
                    else:
                        broken_edges.remove((existing_node, graph_t))

            # continue the regular random rewiring with the remaining broken edges
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
    def __init__(self, grammar: VRG, mixing_dict: Dict, attr_name: str, inp_deg_ast: float, inp_attr_ast: float):
        super().__init__(grammar, mixing_dict, attr_name)
        self.inp_deg_ast = inp_deg_ast  # input degree assortativity
        self.inp_attr_ast = inp_attr_ast  # input attribute assortativity
        return

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
        assert len(broken_edges) == chosen_nt.size, f'Incorrect #broken edges: {len(broken_edges)} != {chosen_nt.size}'

        self.total_rewirings += len(broken_edges)  # all the broken edges need to be rewired

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

        if len(broken_edges) != 0:
            # Figure out if there are terminal nodes on both RULE and the current graph
            rule_terminals = [node for node, d in chosen_rule.graph.nodes(data=True) if 'nt' not in d]
            graph_terminal_list = []
            for u, v in broken_edges:
                node = v if u == existing_node else u
                if 'nt' not in self._gen_graph.nodes[node]:
                    graph_terminal_list.append(node)

            if len(rule_terminals) > 0 and len(graph_terminal_list) > 0:
                # this is where the probabilistic matching needs to happen
                rule_terminal_list = []  # we need b_deg copies of each terminal in the rules
                for rule_t in rule_terminals:
                    b_deg = b_degs[rule_t]
                    if b_deg > 0:
                        rule_terminal_list.extend([rule_t] * b_deg)

                assert len(graph_terminal_list) >= len(graph_terminal_list)

                # do the matching up
                while True:
                    if len(graph_terminal_list) == 0 or len(rule_terminal_list) == 0:
                        break
                    # 1. pick a graph terminal at random and remove it
                    graph_t = random.choice(graph_terminal_list)

                    graph_terminal_list.remove(graph_t)
                    attr_1 = self._gen_graph.nodes[graph_t][self.attr_name]

                    # TODO: NEW STUFF
                    # 2. pick a rule terminal which minimizes the sum of degree and attribute assortativity difference
                    best_rule_terminal: Tuple[int, float] = (None, np.inf)
                    for rule_t in rule_terminal_list:
                        self._gen_graph.add_edge(graph_t, node_label[rule_t])  # add the edge
                        # calculate change in assortativity
                        new_deg_asst = nx.degree_assortativity_coefficient(self._gen_graph)
                        new_attr_asst = nx.attribute_assortativity_coefficient(self._gen_graph, self.attr_name)

                        cost = math.fabs(self.inp_deg_ast - new_deg_asst) + math.fabs(self.inp_attr_ast - new_attr_asst)
                        logging.debug(f'Rule_t: {rule_t}, cost: {cost!r}, prev best: {best_rule_terminal!r}')

                        if cost < best_rule_terminal[1]:
                            best_rule_terminal = rule_t, cost
                        self._gen_graph.remove_edge(graph_t, node_label[rule_t])  # remove the edge

                    assert best_rule_terminal[0] is not None
                    rule_t = best_rule_terminal[0]
                    rule_terminal_list.remove(rule_t)

                    # 3. Add the edge to gen_graph
                    logging.debug(f'Adding broken edge {(node_label[rule_t], graph_t)} the fancy way')
                    self._gen_graph.add_edge(graph_t, node_label[rule_t])  # node_label stores the actual label in gen_graph
                    self.fancy_rewirings += 1  # fancy rewiring

                    # 3. update b_deg for the rule terminal
                    b_degs[rule_t] -= 1

                    # 4. Remove the boundary edge from list of boundary edges
                    if (graph_t, existing_node) in broken_edges:
                        broken_edges.remove((graph_t, existing_node))
                    else:
                        broken_edges.remove((existing_node, graph_t))

            # continue the regular random rewiring with the remaining broken edges
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
                 keep_nodes: Union[Set, None] = None):
        assert isinstance(grammar, NCE), 'Incorrect variant of grammar. Needs NCE'
        super().__init__(grammar, strategy='greedy')
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
    def select_rule(self) -> Tuple[int, NCERule, Set]:
        """
        Pick rules that prioritizes the missing nodes - pick the ones with the most overlap
        :return:
        """
        chosen_nt_node = max(self.current_non_terminal_nodes,
                             key=lambda nt_node: self._gen_graph.nodes[nt_node][
                                 'nt'].id)  # choose the non-terminal with max id
        logging.debug(f'Picking nt: {chosen_nt_node}')
        chosen_nt: NonTerminal = self._gen_graph.nodes[chosen_nt_node]['nt']
        rule_idx = chosen_nt.id - 1

        chosen_rule = self.grammar.rule_list[rule_idx]
        assert chosen_rule.lhs_nt.size == chosen_nt.size, 'Non-terminal size mismatch'

        return chosen_nt_node, chosen_rule, chosen_rule.nodes_covered


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
