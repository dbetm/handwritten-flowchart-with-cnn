import sys

from graphviz import Digraph

sys.path.append('..')
from node import Node
from graph import Graph

class FlowchartGenerator(object):
    """docstring for FlowchartGenerator."""

    def __init__(self, graph):
        super(FlowchartGenerator, self).__init__()
        self.graph = graph


if __name__ == '__main__':
    graph = Graph([], [])
    flow_gen = FlowchartGenerator(graph)
