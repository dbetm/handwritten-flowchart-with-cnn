import sys
import os

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
                        self.__add_subgraph(dest_cls, dest_key, last_key=key)
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
        self.dot.render(view='false')

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


# Unit testing
if __name__ == '__main__':
    # ------ TEST CASE 'Sum 2 numbers' ------
    # Create text nodes
    t0 = Node(coordinate=[569,855,110,242], text='inicio')
    t1 = Node(coordinate=[352,1044,482,589], text='a=0, b=0, res=0')
    t2 = Node(coordinate=[477,905,810,946], text='"Dame a: "')
    t3 = Node(coordinate=[555,691,1267,1374], text='a')
    t4 = Node(coordinate=[412,873,1571,1703], text='"Dame b: "')
    t5 = Node(coordinate=[494,587,2103,2171], text='b')
    t6 = Node(coordinate=[334,791,2424,2532], text='res = a+b')
    t7 = Node(coordinate=[1234,1851,2375,2614], text='"Resultado es: " res')
    t8 = Node(coordinate=[1298,1491,2928,3049], text='fin')
    # Create shape nodes
    s0 = Node(coordinate=[448,1012,57,253], class_shape='start_end')
    s1 = Node(coordinate=[651,755,221,467], class_shape='arrow_line_down')
    s2 = Node(coordinate=[262,1141,378,635], class_shape='process')
    s3 = Node(coordinate=[626,730,603,778], class_shape='arrow_line_down')
    s4 = Node(coordinate=[427,944,753,1089], class_shape='print')
    s5 = Node(coordinate=[580,709,1053,1253], class_shape='arrow_line_down')
    s6 = Node(coordinate=[373,869,1221,1407], class_shape='scan')
    s7 = Node(coordinate=[569,651,1374,1564], class_shape='arrow_line_down')
    s8 = Node(coordinate=[366,969,1510,1882], class_shape='print')
    s9 = Node(coordinate=[494,601,1857,2085], class_shape='arrow_line_down')
    s10 = Node(coordinate=[334,741,2046,2214], class_shape='scan')
    s11 = Node(coordinate=[477,562,2175,2382], class_shape='arrow_line_down')
    s12 = Node(coordinate=[248,873,2353,2574], class_shape='process')
    s13 = Node(coordinate=[826,1216,2392,2507], class_shape='arrow_line_left')
    s14 = Node(coordinate=[1187,1944,2307,2735], class_shape='print')
    s15 = Node(coordinate=[1373,1491,2696,2917], class_shape='arrow_line_down')
    s16 = Node(coordinate=[1169,1619,2878,3082], class_shape='start_end')

    filename = 'sum.dot'
    graph = Graph(
        image_path="",
        text_nodes=[t0, t1, t2, t3, t4, t5, t6, t7, t8],
        shape_nodes=[s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16]
    )

    flow = graph.generate_graph()
    #flow = aux_flow() # use when graph is not good constructed
    print(flow)

    flow_gen = FlowchartGenerator(graph, flow, filename=filename)
    flow_gen.generate_flowchart()
