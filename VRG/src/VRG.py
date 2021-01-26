'''
refactored VRG
'''
import abc
import logging
from typing import List, Dict

import networkx as nx

from VRG.src.Rule import VRGRule, NCERule
from VRG.src.utils import node_matcher_strict, edge_matcher


class BaseVRG(abc.ABC):
    """
    Base class for VRG
    """
    __slots__ = ('name', 'type', 'clustering', 'mu', 'rule_list', 'rule_dict', 'cost', 'num_rules', 'unique_rule_list',
                 'unique_rule_dict', 'unique_rule_rhs')

    def __init__(self, type: str, clustering: str, name: str, mu: int):
        self.name: str = name  # name of the graph
        self.type: str = type  # type of grammar - mu, local, global, selection strategy - random, dl, level, or dl_levels
        self.clustering: str = clustering  # clustering strategy
        self.mu = mu

        self.rule_list: List[VRGRule] = []  # list of Rule objects
        self.rule_dict: Dict[int, List[VRGRule]] = {}  # dictionary of rules, keyed in by their LHS

        self.unique_rule_list: List[NCERule] = []  # list of unique Rule objects
        self.unique_rule_dict: Dict[int, List[NCERule]] = {}  # dictionary of rules, keyed in by their LHS
        self.unique_rule_rhs: List = []  # list of unique rule RHSs

        self.cost: int = -1  # the MDL of the rules
        self.num_rules: int = 0  # number of active rules

    def copy(self):
        vrg_copy = BaseVRG(type=self.type, clustering=self.clustering, name=self.name, mu=self.mu)
        vrg_copy.rule_list = self.rule_list[:]
        vrg_copy.rule_dict = dict(self.rule_dict)
        vrg_copy.cost = self.cost
        vrg_copy.num_rules = self.num_rules
        return vrg_copy

    def __len__(self):
        return len(self.rule_list)

    def __contains__(self, rule: VRGRule):
        return rule in self.rule_dict[rule.lhs_nt.size]

    def __str__(self):
        if self.cost == 0:
            self.calculate_cost()
        st = f'graph: {self.name!r}, mu: {self.mu}, type: {self.type!r} clustering: {self.clustering!r} rules: {len(self.rule_list):_d}' \
             f'({self.num_rules:_d}) mdl: {round(self.cost, 3):_g} bits'
        return st

    def __repr__(self):
        return f'<{str(self)}>'

    def __getitem__(self, item):
        return self.rule_list[item]

    def get_cost(self) -> float:
        if self.cost == -1:
            self.cost = 0
            self.calculate_cost()
        return self.cost

    def reset(self):
        # reset the grammar
        self.rule_list = []
        self.rule_dict = {}
        self.cost = 0
        self.num_rules = 0

    def calculate_cost(self):
        self.cost = 0
        for rule in self.rule_list:
            rule.calculate_cost()
            self.cost += rule.cost
        return

    @abc.abstractmethod
    def add_rule(self, rule: VRGRule) -> int:
        pass


class VRG(BaseVRG):
    """
    Class for Vertex Replacement Grammars - No node correspondence
    """

    def add_rule(self, rule: VRGRule) -> int:
        """
        adds to the grammar iff it's a new rule - isomorphic rules are stored multiple times
        :param rule:
        :return:
        """
        if rule.lhs_nt.size not in self.rule_dict:
            self.rule_dict[rule.lhs_nt.size] = []

        self.num_rules += 1
        rule.set_id(self.num_rules)
        rule.lhs_nt.id = rule.id

        for old_rule in self.rule_dict[rule.lhs_nt.size]:
            if rule == old_rule:  # check for isomorphism
                logging.debug(f'Duplicate rule found!')
                old_rule.update(rule)
                rule.id = old_rule.id
                return old_rule.id

        # new rule
        self.rule_list.append(rule)
        self.rule_dict[rule.lhs_nt.size].append(rule)

        return rule.id
        #
        # # update the unique rules
        # rule_copy = VRGRule(lhs_nt=rule.lhs_nt, graph=rule.graph, level=rule.level, frequency=1)
        #
        # if rule_copy.lhs_nt.size not in self.unique_rule_dict:
        #     self.unique_rule_dict[rule_copy.lhs_nt.size] = []
        # isomorphic_rule_found = False
        # for old_rule in self.unique_rule_dict[rule_copy.lhs_nt.size]:
        #     # if nx.is_isomorphic(old_rule.graph, rule_copy.graph):
        #     if nx.is_isomorphic(old_rule.graph, rule_copy.graph, node_match=node_matcher_strict,
        #                         edge_match=edge_matcher):
        #         isomorphic_rule_found = True
        #         logging.debug('Isomorphic rule found!')
        #         old_rule.frequency += 1
        #
        # if not isomorphic_rule_found:  # it is a new rule
        #     self.unique_rule_list.append(rule_copy)
        #     self.unique_rule_dict[rule_copy.lhs_nt.size].append(rule_copy)
        #
        #
        # isomorphic_rhs_found = False
        # for i in range(len(self.unique_rule_rhs)):
        #     old_rule_rhs = self.unique_rule_rhs[i][0]
        #     if nx.is_isomorphic(old_rule_rhs, rule_copy.graph):
        #         # if nx.is_isomorphic(old_rule_rhs, rule_copy.graph, node_match=node_matcher, edge_match=edge_matcher):
        #         self.unique_rule_rhs[i][1] += 1  # ugly hack to update the frequencies in place
        #         isomorphic_rhs_found = True
        #
        # if not isomorphic_rhs_found:
        #     self.unique_rule_rhs.append([rule_copy.graph, 1])
        # return rule.id

    def copy(self):
        vrg_copy = VRG(type=self.type, clustering=self.clustering, name=self.name, mu=self.mu)
        vrg_copy.rule_list = self.rule_list[:]
        vrg_copy.rule_dict = dict(self.rule_dict)
        vrg_copy.cost = self.cost
        vrg_copy.num_rules = self.num_rules
        return vrg_copy


class AttributedVRG(VRG):
    def __init__(self, clustering: str, name: str, mu: int, attr_name: str):
        super().__init__(type='A-VRG', clustering=clustering, name=name, mu=mu)
        self.attr_name = attr_name


class NCE:
    """
    Class for Vertex Replacement Grammars - for isomorphic graphs
    """
    __slots__ = 'name', 'type', 'clustering', 'mu', 'rule_list', 'rule_dict',\
                'unique_rule_list', 'unique_rule_dict', 'unique_rule_rhs', 'cost', 'num_rules'

    def __init__(self, type: str, clustering: str, name: str, mu: int):
        self.name: str = name  # name of the graph
        self.type: str = type  
        self.clustering: str = clustering  # clustering strategy
        self.mu = mu

        self.rule_list: List[NCERule] = []  # list of Rule objects
        self.rule_dict: Dict[int, List[NCERule]] = {}  # dictionary of rules, keyed in by their LHS

        self.unique_rule_list: List[NCERule] = []  # list of unique Rule objects
        self.unique_rule_dict: Dict[int, List[NCERule]] = {}  # dictionary of rules, keyed in by their LHS
        self.unique_rule_rhs: List = []  # list of unique rule RHSs

        self.cost: int = -1  # the MDL of the rules
        self.num_rules: int = 0  # number of active rules

    def __len__(self):
        return len(self.rule_list)

    def __contains__(self, rule: VRGRule):
        return rule in self.rule_dict[rule.lhs_nt.size]

    def __str__(self):
        if self.cost == 0:
            self.calculate_cost()
        st = f'graph: {self.name}, mu: {self.mu}, type: {self.type} clustering: {self.clustering} ' \
             f'rules: {len(self.unique_rule_list):_d} ({self.num_rules:_d}) unique RHSs: {len(self.unique_rule_rhs):_d}'
        return st

    def __repr__(self):
        return f'<{str(self)}>'

    def __getitem__(self, item):
        return self.rule_list[item]

    def reset(self):
        # reset the grammar
        self.rule_list = []
        self.rule_dict = {}
        self.cost = 0
        self.num_rules = 0

    def get_cost(self) -> float:
        if self.cost == -1:
            self.calculate_cost()
        return self.cost

    def add_rule(self, rule: VRGRule) -> int:
        # adds to the grammar iff it's a new rule
        if rule.lhs_nt.size not in self.rule_dict:
            self.rule_dict[rule.lhs_nt.size] = []

        self.num_rules += 1
        rule.set_id(self.num_rules)

        # new rule
        self.rule_list.append(rule)
        self.rule_dict[rule.lhs_nt.size].append(rule)

        # update the unique rules
        rule_copy = NCERule(lhs_nt=rule.lhs_nt, graph=rule.graph, level=rule.level, frequency=1)

        if rule_copy.lhs_nt.size not in self.unique_rule_dict:
            self.unique_rule_dict[rule_copy.lhs_nt.size] = []

        isomorphic_rule_found = False
        for old_rule in self.unique_rule_dict[rule_copy.lhs_nt.size]:
            # if nx.is_isomorphic(old_rule.graph, rule_copy.graph):
            if nx.is_isomorphic(old_rule.graph, rule_copy.graph, node_match=node_matcher_strict, edge_match=edge_matcher):
                isomorphic_rule_found = True
                logging.debug('Isomorphic rule found!')
                old_rule.frequency += 1

        if not isomorphic_rule_found:  # it is a new rule
            self.unique_rule_list.append(rule_copy)
            self.unique_rule_dict[rule_copy.lhs_nt.size].append(rule_copy)

        isomorphic_rhs_found = False
        for i in range(len(self.unique_rule_rhs)):
            old_rule_rhs = self.unique_rule_rhs[i][0]
            if nx.is_isomorphic(old_rule_rhs, rule_copy.graph):
            # if nx.is_isomorphic(old_rule_rhs, rule_copy.graph, node_match=node_matcher, edge_match=edge_matcher):
                self.unique_rule_rhs[i][1] += 1  # ugly hack to update the frequencies in place
                isomorphic_rhs_found = True

        if not isomorphic_rhs_found:
            self.unique_rule_rhs.append([rule_copy.graph, 1])
        return rule.id

    def calculate_cost(self):
        self.cost = 0
        for rule in self.rule_list:
            rule.calculate_cost()
            self.cost += rule.cost
        return
