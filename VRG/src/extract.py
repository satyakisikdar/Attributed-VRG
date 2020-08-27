"""
VRG extraction
"""
import abc
import logging
from itertools import chain
from typing import List, Tuple, Set

import numpy as np
from tqdm import tqdm

from VRG.src.LightMultiGraph import LightMultiGraph
from VRG.src.NonTerminal import NonTerminal
from VRG.src.Rule import VRGRule, NCERule, AVRGRule
from VRG.src.Tree import TreeNode, get_leaves
from VRG.src.VRG import VRG, NCE, AttributedVRG
from VRG.src.utils import find_boundary_edges, set_boundary_degrees, timer


class BaseExtractor(abc.ABC):
    """
    New base extractor class with Anytree class
    """
    ALLOWED_TYPES = ('mu_random', )

    def __init__(self, g: LightMultiGraph, type: str, root: TreeNode, mu: int, clustering: str):
        assert type in BaseExtractor.ALLOWED_TYPES, f'Invalid mode: {type}'
        self.g: LightMultiGraph = g
        self.type: str = type
        self.root: TreeNode = root
        self.grammar = None
        self.mu: int = mu
        self.clustering = clustering
        return

    def __str__(self) -> str:
        return f'type: {self.type}, mu: {self.mu} #rules: {len(self.grammar.rule_list)}'

    def __repr__(self) -> str:
        return str(self)

    def get_best_tnode(self) -> TreeNode:
        """
        Extract the lowest scoring tnode from the tree
        """
        best_tnode = self.root
        best_score = self.root.score
        for tnode in self.root.descendants:
            if tnode.is_leaf:
                continue
            if tnode.score == 0:
                best_tnode = tnode
                best_score = tnode.score
                break
            if tnode.score < best_score:
                best_score = tnode.score
                best_tnode = tnode

        return best_tnode

    @abc.abstractmethod
    def extract_rule(self, tnode: TreeNode) -> Tuple[Set, VRGRule, List]:
        """
        Extract the rule based on the tnode
        :param tnode:
        :return:
        """
        pass

    def set_tnode_score(self, tnode: TreeNode) -> None:
        score = None
        diff = self.mu - len(tnode.leaves)

        if diff < 0:  # there are more nodes than mu
            mu_score = np.inf
        elif diff >= 0:
            mu_score = diff  # mu is greater

        sg = self.g.subgraph(get_leaves(tnode)).copy()
        if (sg.size() == 0):  # or (not nx.is_connected(sg)):  # check for connectivity here
            mu_score = np.inf

        if self.type == 'mu_random':
            score = mu_score  # |mu - nleaf|
        elif self.type == 'mu_level':
            score = mu_score, tnode.depth  # |mu - nleaf|, depth of the tnode
        elif 'dl' in self.type:  # compute cost only if description length is used for scores
            raise NotImplementedError('DL related methods are not implemented yet')

        assert score is not None, 'score is None'
        tnode.score = score  # set the score
        return

    def update_subtree_scores(self, start_tnode: TreeNode):
        """
        Update the subtree scores rooted at the treenode tnode
        :param start_tnode:
        :return:
        """
        for tnode in chain([start_tnode], start_tnode.descendants):  # start from start_tnode and then the descendants
            tnode: TreeNode
            if tnode.is_leaf:  # skip over the leaves
                continue
            self.set_tnode_score(tnode=tnode)  # update the score of the tree node
            # logging.debug(f'tnode: {tnode.name} score: {tnode.score}')  # active_leaves: {active_leaves}')
        return

    def update_ancestors(self, start_tnode: TreeNode, name: str):
        """
        Update the scores of the ancestors after the start_tnode is updated - also update the Tree data structure
        :param start_tnode:
        :param name: new name of the non-terminal
        :return:
        """
        start_tnode.children = tuple()  # make it a leaf - updates the data structure
        start_tnode.name = name
        start_tnode.score = None  # now it's a leaf

        tnode = start_tnode.parent  # start from the parent
        while tnode is not None:  # we might not have to go up all the way - stop when the score of a node is still infinity
            old_score = tnode.score
            self.set_tnode_score(tnode=tnode)
            if np.isinf(tnode.score):  # dont bother any more TODO: check if this works when score is a tuple
                break
            assert (not np.isinf(tnode.score)) and \
                   tnode.score != old_score, 'Score was not set properly'  # all the ancestor's scores have to change
            tnode = tnode.parent
        return

    def update_tree(self, tnode: TreeNode, name: str):
        """
        Update the tree after each extraction
        :return:
        """
        self.update_ancestors(start_tnode=tnode, name=name)
        return

    def update_graph(self, rule: VRGRule, boundary_edges: List, subtree: Set) -> None:
        """
        Update the graph after extracting the rule
        1. Remove the existing nodes from the graph
        2. Replace it with a new node (nt#x) with the associated NonTerminal object
        3. Connect the new node to the rest of the graph
        :param tnode:
        :param rule:
        :param boundary_edges:
        :return:
        """
        # remove existing nodes
        self.g.remove_nodes_from(subtree)

        # replace it with a new NonTerminal node from the Rule
        nt = rule.lhs_nt
        new_node_label = f'nt{nt.id}'
        self.g.add_node(new_node_label, nt=nt)

        # rewire the new node to the rest using the boundary edges
        for u, v in boundary_edges:
            if u in subtree:  # u is now the new non-terminal
                u = new_node_label
            else:
                v = new_node_label
            self.g.add_edge(u, v)
        return

    @timer
    def extract(self) -> VRG:
        """
        Extracts a RandomVRG
        :return:
        """
        num_nodes = self.g.order()
        tqdm.write(
            f'Extracting grammar name:{self.grammar.name} mu:{self.grammar.mu} type:{self.grammar.type} clustering:{self.grammar.clustering}')

        self.update_subtree_scores(self.root)  # initialize the subtree scores
        with tqdm(total=100, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50) as pbar:
            while self.g.order() > 1:  # the graph has more than 1 node
                tnode = self.get_best_tnode()  # get the best tnode
                subtree, rule, boundary_edges = self.extract_rule(tnode)
                self.grammar.add_rule(rule)
                self.update_graph(boundary_edges=boundary_edges, subtree=subtree, rule=rule)
                self.update_tree(tnode, name=f'nt{rule.lhs_nt.id}')

                percent = (1 - (self.g.order() - 1) / (num_nodes - 1)) * 100
                curr_progress = percent - pbar.n
                pbar.update(curr_progress)

        if hasattr(self.grammar, 'unique_rule_rhs'):
            # sort the unique rule rhs list
            logging.error('Sorting the rule RHS graph based on frequencies')
            self.grammar.unique_rule_rhs.sort(key=lambda item: item[1], reverse=True)

        return self.grammar


class VRGExtractor(BaseExtractor):
    def __init__(self, g: LightMultiGraph, type: str, root: TreeNode, mu: int, clustering: str):
        super().__init__(g=g, type=type, root=root, mu=mu, clustering=clustering)
        self.grammar = VRG(clustering=clustering, mu=mu, type=type, name=g.name)
        return

    def set_tnode_score(self, tnode: TreeNode) -> None:
        score = None
        diff = self.mu - len(tnode.leaves)

        if diff < 0:  # there are more nodes than mu
            mu_score = np.inf
        elif diff >= 0:
            mu_score = diff  # mu is greater

        sg = self.g.subgraph(get_leaves(tnode)).copy()
        if sg.size() == 0:  # or (not nx.is_connected(sg)):  # check for connectivity here
            mu_score = np.inf

        if self.type == 'mu_random':
            score = mu_score  # |mu - nleaf|
        elif self.type == 'mu_level':
            score = mu_score, tnode.depth  # |mu - nleaf|, depth of the tnode
        elif 'dl' in self.type:  # compute cost only if description length is used for scores
            raise NotImplementedError('DL related methods are not implemented yet')

        assert score is not None, 'score is None'
        tnode.score = score  # set the score
        return

    def extract_rule(self, tnode: TreeNode) -> Tuple[Set, VRGRule, List]:
        """
        Extract rule
        :param tnode:
        :return:
        """
        subtree = get_leaves(tnode)
        sg = self.g.subgraph(subtree).copy()
        assert isinstance(sg, LightMultiGraph)
        # assert nx.is_connected(sg), 'Subgraph is not connected'
        boundary_edges = find_boundary_edges(self.g, subtree)

        # nodes_covered = subtree  # for now
        nodes_covered = set(filter(lambda node: isinstance(node, int), subtree))
        nt = NonTerminal(size=len(boundary_edges), nodes_covered=nodes_covered)

        rule = VRGRule(lhs_nt=nt, graph=sg)
        set_boundary_degrees(self.g, rule.graph)
        rule.generalize_rhs_and_store_correspondence()

        return subtree, rule, boundary_edges


class AVRGExtractor(VRGExtractor):
    def __init__(self, g: LightMultiGraph, type: str, root: TreeNode, mu: int, clustering: str, attr_name: str):
        super().__init__(g=g, type=type, root=root, mu=mu, clustering=clustering)
        self.attr_name = attr_name
        self.grammar = AttributedVRG(clustering=clustering, mu=mu, name=g.name, attr_name=attr_name)
        return

    def extract_rule(self, tnode: TreeNode) -> Tuple[Set, AVRGRule, List]:
        """
        Extract AVRG rule
        :param tnode:
        :return:
        """
        subtree = get_leaves(tnode)
        sg = self.g.subgraph(subtree).copy()
        assert isinstance(sg, LightMultiGraph)
        # assert nx.is_connected(sg), 'Subgraph is not connected'
        boundary_edges = find_boundary_edges(self.g, subtree)

        # nodes_covered = subtree  # for now
        nodes_covered = set(filter(lambda node: isinstance(node, int), subtree))
        nt = NonTerminal(size=len(boundary_edges), nodes_covered=nodes_covered)

        rule = AVRGRule(lhs_nt=nt, graph=sg, attr_name=self.attr_name)
        set_boundary_degrees(self.g, rule.graph)
        rule.generalize_rhs_and_store_correspondence()

        return subtree, rule, boundary_edges


class NCEExtractor(BaseExtractor):
    def __init__(self, g: LightMultiGraph, type: str, root: TreeNode, mu: int, clustering: str):
        super().__init__(g=g, type=type, root=root, mu=mu, clustering=clustering)
        self.grammar = NCE(type='mu_random', clustering=clustering, mu=mu, name=g.name)
        return

    def extract_rule(self, tnode: TreeNode) -> Tuple[Set, NCERule, List]:
        """
        Extract rule
        :param tnode:
        :return:
        """
        # raise NotImplementedError('Need to store both boundary nodes and boundary edges for each rule')
        subtree = get_leaves(tnode)
        sg = self.g.subgraph(subtree).copy()
        assert isinstance(sg, LightMultiGraph)
        # assert nx.is_connected(sg), 'Subgraph is not connected'
        boundary_edges = find_boundary_edges(self.g, subtree)

        nodes_covered = set(filter(lambda node: isinstance(node, int), subtree))
        nt = NonTerminal(size=len(boundary_edges), nodes_covered=nodes_covered, id=self.grammar.num_rules+1)

        rule = NCERule(lhs_nt=nt, graph=sg)
        set_boundary_degrees(self.g, rule.graph)
        rule.generalize_rhs_and_store_correspondence()
        rule.update_boundary_nodes_edges(boundary_edges=boundary_edges)

        return subtree, rule, boundary_edges
