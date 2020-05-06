import sys

from graphviz import Digraph

sys.path.append('..')
from node import Node
from graph import Graph


class FlowchartGenerator(object):
    """Flowchart Generator allows generate image for directed graph
    of the flowchart.
    """

    def __init__(self, graph, flow, filename):
        super(FlowchartGenerator, self).__init__()
        self.graph_nodes = graph.get_nodes()
        print(self.graph_nodes)
        self.flow = flow
        self.added_nodes = []
        self.dot = Digraph(filename="results/"+filename+"flowchart")
        self.dot.format = 'png'
        self.DICT = {
            'start_end':'ellipse',
            'print':'invhouse',
            'scan':'parallelogram',
            'process':'box',
            'decision':'diamond'
        }

    def generate_flowchart(self):
        """Driver method for generate reconstructed flowchart image."""

        for i, key in enumerate(self.flow):

            node = self.flow[key]
            if(len(node) == 1):
                cls = self.graph_nodes[key].get_class()
                if(self.__is_any_arrow(cls)):
                    continue

                self.__build_node(cls, self.graph_nodes[key].get_text(), key)
                # Find next shape to connect it
                dest_cls, dest_key = self.__find_dest(node[0])
                if(dest_cls != 'decision'):
                    self.__build_node(
                        dest_cls,
                        self.graph_nodes[dest_key].get_text(),
                        dest_key
                    )
                    self.__add_edge(str(key), str(dest_key))
                else:
                    if (dest_key in self.added_nodes):
                        self.__add_edge(str(key), str(dest_key))
                    else:
                        self.__add_subgraph(dest_cls, dest_key)
            elif(len(node) == 2): #decision
                arrow_text = self.graph_nodes[node[0]].get_text().lower()
                if(arrow_text == 'si' or arrow_text == 'yes' or arrow_text == 'sí'):
                    dest_cls, dest_key = self.__find_dest(self.flow[key][0])
                    if(dest_cls != 'decision'):
                        self.__build_node(
                            dest_cls,
                            self.graph_nodes[dest_key].get_text(),
                            dest_key
                        )
                        self.__add_edge(str(key), str(dest_key), text="Sí")
                    else:
                        self.__add_subgraph(
                            dest_cls,
                            dest_key,
                            last_key=key,
                            text_edge="Sí"
                        )

        # Render flowchart image
        self.dot.render(view='true')

    def __is_any_arrow(self, _class):
        return _class.split('_')[0] == "arrow"

    def __build_node(self, _class, text, key):
        if not(key in self.added_nodes):
            type = self.__get_type_node(_class)
            self.dot.node(str(key), label=text, shape=type)
            self.added_nodes.append(key)

    def __add_subgraph(self, _class, key, last_key=None, text_edge=None):
        with self.dot.subgraph() as s:
            s.attr(rank='same')
            text = self.graph_nodes[key].get_text()
            type = self.__get_type_node(_class)
            s.node(str(key), label=text, shape=type)
            if(last_key == None):
                last_key = self.added_nodes[len(self.added_nodes)-1]
            self.added_nodes.append(key)
            self.__add_edge(str(last_key), str(key), text_edge)
            """By default, the denial of the condition is selected to be at the same
            level as the decision shape.
            """
            arrow_text = self.graph_nodes[self.flow[key][0]].get_text().lower()
            if(arrow_text == 'no'):
                dest_cls, dest_key = self.__find_dest(self.flow[key][0])
            else:
                dest_cls, dest_key = self.__find_dest(self.flow[key][1])

            text = self.graph_nodes[dest_key].get_text()
            type = self.__get_type_node(dest_cls)
            s.node(str(dest_key), label=text, shape=type)
            self.added_nodes.append(dest_key)
            self.__add_edge(str(key), str(dest_key), text='No')


    def __get_type_node(self, _class):
        return self.DICT[_class]

    def __find_dest(self, key):
        node = self.flow[key]

        if(len(node) == 1):
            cls = self.graph_nodes[node[0]].get_class()
            if(self.__is_any_arrow(cls)):
                # recursion
                return self.__find_dest(node[0])
            # found destination
            return cls, node[0]
        elif(len(node) == 2):
            pass
        else:
            pass

    def __add_edge(self, origin, dest, text=None):
        if(text == None):
            self.dot.edge(origin, dest)
        else:
            self.dot.edge(origin, dest, label=text)


if __name__ == '__main__':
    # Order (x1, x2, y1, y2)
    # ------ TEST CASE 'Square area' ------
    # Create text nodes
    t0 = Node(coordinate=[689,1012,207,335], text='inicio')
    t1 = Node(coordinate=[607,1284,658,787], text='x=0.0, res=0.0')
    t2 = Node(coordinate=[764,881,1076,1169], text='x')
    t3 = Node(coordinate=[576,1117,1430,1556], text='res=x*x')
    t4 = Node(coordinate=[756,953,1928,2023], text='res')
    t5 = Node(coordinate=[681,912,2448,2561], text='fin')

    # Create shape nodes
    s0 = Node(coordinate=[553,1122,174,397], class_shape='start_end')
    s1 = Node(coordinate=[776,858,376,630], class_shape='arrow_line_down')
    s2 = Node(coordinate=[458,1392,592,846], class_shape='process')
    s3 = Node(coordinate=[787,879,838,1038], class_shape='arrow_line_down')
    s4 = Node(coordinate=[587,1048,1017,1202], class_shape='scan')
    s5 = Node(coordinate=[774,879,1187,1356], class_shape='arrow_line_down')
    s6 = Node(coordinate=[471,1212,1346,1641], class_shape='process')
    s7 = Node(coordinate=[820,935,1597,1882], class_shape='arrow_line_down')
    s8 = Node(coordinate=[610,1202,1853,2166], class_shape='print')
    s9 = Node(coordinate=[792,884,2148,2402], class_shape='arrow_line_down')
    s10 = Node(coordinate=[517,1128,2379,2661], class_shape='start_end')

    filename = 'square_area.dot'
    graph = Graph(
        [t0, t1, t2, t3, t4, t5],
        [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10]
    )

    flow = graph.generate_graph()
    #flow = aux_flow() # use when graph is not good constructed
    print(flow)
    flow_gen = FlowchartGenerator(graph, flow, filename=filename)
    flow_gen.generate_flowchart()
