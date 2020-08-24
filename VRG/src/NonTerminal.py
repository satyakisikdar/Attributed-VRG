from typing import Set


class NonTerminal:
    """
    Container for non-terminals
    Each non-terminal has an unique id, size (omega), nodes covered
    """
    __slots__ = 'id', 'size', 'nodes_covered'

    def __init__(self, size: int, nodes_covered: Set[int], id: int=-1) -> None:
        self.id = id  # each non-terminal has a unique id
        self.size = size  # number of broken boundary edges
        self.nodes_covered = nodes_covered  # set of nodes that were compressed to create this non-terminal
        return

    def __str__(self) -> str:
        return f'id: {self.id}, size: {self.size}'  #, nodes: {self.nodes_covered}'

    def __repr__(self) -> str:
        return f'<id: {self.id}; size: {self.size}>'

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return self.size == other.size  # they must have the same size

    def set_id(self, id: int) -> None:
        self.id = id
        return
