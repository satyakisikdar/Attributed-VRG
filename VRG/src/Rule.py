from typing import List, Set

import networkx as nx

import VRG.src.MDL as MDL
from VRG.src.LightMultiGraph import LightMultiGraph
from VRG.src.NonTerminal import NonTerminal
from VRG.src.utils import load_pickle, edge_matcher


class BaseRule:
    """
    Base class for Rule
    """
    __slots__ = 'lhs_nt', 'graph', 'level', 'cost', 'frequency', 'id', 'rhs_non_terminals', 'nodes_covered', 'is_attributed'

    def __init__(self, lhs_nt: NonTerminal, graph: LightMultiGraph, is_attributed: bool, level: int = 0, frequency: int = 1):
        self.lhs_nt = lhs_nt  # the left hand side: the number of boundary edges
        self.graph = graph  # the right hand side subgraph
        self.level = level  # level of discovery in the tree (the root is at 0)
        self.frequency = frequency  # frequency of occurence
        self.id = None  # id of the rule
        self.cost = 0
        self.rhs_non_terminals: List[NonTerminal] = []  # list of non-terminals in the RHS graph
        self.nodes_covered: Set[int] = set()  # set of nodes in the original graph that is covered by this rule
        self.is_attributed: bool = is_attributed  # flag for attributed

        for node, data in self.graph.nodes(data=True):
            if 'nt' in data:  # the node in the RHS is a NonTerminal node
                nt = data['nt']
                self.rhs_non_terminals.append(nt)
                # self.nodes_covered[0].update(nt.nodes_covered)
            else:
                self.nodes_covered.add(node)
        return

    def __str__(self) -> str:
        if self.id is not None:
            st = f'({self.id}) '
        else:
            st = ''
        st += f'{self.lhs_nt.size} → (n = {self.graph.order()}, m = {self.graph.size()})'
        # print non-terminals if present

        if len(self.rhs_non_terminals) != 0:  # if it has non-terminals, print the sizes
            st += ' nt: {' + ','.join(map(lambda x: str(x.size), self.rhs_non_terminals)) + '}'

        if self.frequency > 1:  # if freq > 1, show it in square brackets
            st += f' [{self.frequency}x]'
        return st

    def __repr__(self) -> str:
        st = f'<{self.lhs_nt.size} → ({self.graph.order()}, {self.graph.size()})'

        if len(self.rhs_non_terminals) != 0:  # if it has non-terminals, print the sizes
            st += '{' + ','.join(map(lambda nt: str(nt.size), self.rhs_non_terminals)) + '}'
        if self.frequency > 1:  # if freq > 1, show it in square brackets
            st += f' [{self.frequency}x]'
        st += '>'
        return st

    def __eq__(self, other) -> bool:  # two rules are equal if the LHSs match and RHSs are isomorphic
        import networkx.algorithms.isomorphism as iso
        isomorphic = self.lhs_nt.size == other.lhs_nt.size  # first check if the LHS are the same size

        g1 = nx.convert_node_labels_to_integers(self.graph)
        g2 = nx.convert_node_labels_to_integers(other.graph)
        isomorphic = isomorphic and nx.is_isomorphic(g1, g2, node_match=iso.numerical_node_match('b_deg', default=0),
                                                     edge_match=edge_matcher)  # use the node and edge matcher
        return isomorphic

    def __hash__(self):
        g = nx.freeze(self.graph)
        return hash((self.lhs_nt.size, g))

    def __deepcopy__(self, memodict={}):
        return BaseRule(lhs_nt=self.lhs_nt, graph=self.graph, level=self.level, frequency=self.frequency,
                        is_attributed=self.is_attributed)

    def calculate_cost(self) -> float:
        if self.cost != 0:
            return self.cost
        b_deg = nx.get_node_attributes(self.graph, 'b_deg')
        assert len(b_deg) > 0, 'invalid b_deg'
        max_boundary_degree = max(b_deg.values())

        self.cost = MDL.gamma_code(self.lhs_nt.size + 1) + MDL.graph_dl(self.graph, self.is_attributed) + \
                    MDL.gamma_code(self.frequency + 1) + \
                    self.graph.order() * MDL.gamma_code(max_boundary_degree + 1)
        return self.cost

    def contract_rhs(self):
        pass

    def set_id(self, id):
        self.id = id
        return

    def draw(self):
        """
        Returns a graphviz object that can be rendered into a pdf/png
        """
        from graphviz import Graph
        flattened_graph = nx.Graph(self.graph)

        dot = Graph(engine='dot')
        for node, data in self.graph.nodes(data=True):
            if 'nt' in data:
                dot.node(str(node), str(data['nt'].size), shape='square', height='0.20')
            else:
                dot.node(str(node), '', height='0.12', shape='circle')

        for u, v in flattened_graph.edges():
            w = self.graph.number_of_edges(u, v)
            if w > 1:
                dot.edge(str(u), str(v), label=str(w))
            else:
                dot.edge(str(u), str(v))
        return dot

    def generalize_rhs_and_store_correspondence(self):
        """
        Relabels the RHS such that the terminal nodes are Latin characters,
        and non-terminals are marked as 'nt'. Also stores the local node correspondance
        for the terminal nodes
        TODO: what happens when there are multiple occurences of the same rule?
        :param self: RHS subgraph
        :return:
        """
        return
        mapping = {}
        actual_label = {}  # stores the local node correspondance

        internal_node_counter = 'a'

        for n, d in self.graph.nodes(data=True):
            if 'nt' not in d:  # dont relabel non-terminals - can cause loopy relabels
                mapping[n] = internal_node_counter  # terminals are in latin characters
                actual_label[internal_node_counter] = n
                internal_node_counter = chr(ord(internal_node_counter) + 1)
            else:  # non-terminal
                mapping[n] = n
                actual_label[n] = n

        # nx.relabel_nodes(self.graph, mapping=mapping, copy=False)
        nx.set_node_attributes(self.graph, name='actual_label', values=actual_label)
        return

    def update(self, new_rule):
        """
        Update existing rule
            - update frequency
            - update node correspondences - actual_labels

        :param new_rule:
        :return:
        """
        self.frequency += 1  # update frequency
        return


class VRGRule(BaseRule):
    """
    Rule class for vanilla VRGs
    """
    def __init__(self, lhs_nt, graph, level=0, frequency=1):
        super().__init__(lhs_nt=lhs_nt, graph=graph, level=level, frequency=frequency, is_attributed=False)

    def __deepcopy__(self, memodict={}):
        return VRGRule(lhs_nt=self.lhs_nt, graph=self.graph, level=self.level, frequency=self.frequency)


class AVRGRule(BaseRule):
    """
    Rules for attributed VRG
    """
    def __init__(self, lhs_nt: NonTerminal, graph: LightMultiGraph, attr_name: str):
        super().__init__(lhs_nt, graph, is_attributed=True)
        self.attr_name = attr_name
        return

    def __eq__(self, other) -> bool:
        """
        Check for rule isomorphism
        :param other:
        :return:
        """
        import networkx.algorithms.isomorphism as iso
        isomorphic = self.lhs_nt.size == other.lhs_nt.size  # first check if the LHS are the same size

        g1 = nx.convert_node_labels_to_integers(self.graph)
        g2 = nx.convert_node_labels_to_integers(other.graph)
        isomorphic = isomorphic and nx.is_isomorphic(g1, g2, node_match=iso.categorical_node_match(self.attr_name, ''),
                                                     edge_match=edge_matcher)  # use the node and edge matcher

        return isomorphic


class NCERule(BaseRule):
    """
    Rule class for Neighborhood Controlled Embedding
    """
    def __init__(self, lhs_nt, graph, level=0, cost=0, frequency=1):
        super().__init__(lhs_nt=lhs_nt, graph=graph, level=level, frequency=frequency, is_attributed=False)
        self.boundary_nodes: Set = set()  # store the boundary nodes
        self.boundary_edges: Set = set()  # store the boundary edges
        return

    def __deepcopy__(self, memodict={}):
        return NCERule(lhs_nt=self.lhs_nt, graph=self.graph, level=self.level, frequency=self.frequency)

    def update_boundary_nodes_edges(self, boundary_edges: List):
        """
        Update the boundary nodes and edges
        :return:
        """
        for u, v in boundary_edges:
            if self.graph.has_node(u):  # u is the internal node
                self.boundary_nodes.add(v)
            else:
                self.boundary_nodes.add(u)
        # logging.debug(f'Boundary nodes: {self.boundary_nodes} edges: {self.boundary_edges}')
        self.boundary_edges = boundary_edges
        return


if __name__ == '__main__':
    grammar = load_pickle('../dumps/grammars/sample/mu_level_dl_cond_3_0.pkl')
    print(grammar)
    rule1: VRGRule = grammar.rule_dict[0][0]  # rule 0 -> 2=2

    graph1: LightMultiGraph = rule1.graph
    graph2: LightMultiGraph = rule1.graph.copy()
    # graph2.nodes['nt1']['b_deg'] = 1

    print('before mod', edge_matcher(graph1['nt1']['nt2'], graph2['nt1']['nt2']))
    graph2['nt1']['nt2']['weight'] = 3
    print('after mod', edge_matcher(graph1['nt1']['nt2'], graph2['nt1']['nt2']))

    rule2 = VRGRule(lhs_nt=rule1.lhs_nt, graph=graph2)

    print('Rules are isomorphic?', rule1 == rule2)
