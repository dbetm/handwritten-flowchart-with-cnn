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
        self.dot = Digraph(filename=filename)
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
                if(arrow_text == 'si' or arrow_text == 'yes'):
                    dest_cls, dest_key = self.__find_dest(self.flow[key][0])
                    if(dest_cls != 'decision'):
                        self.__build_node(
                            dest_cls,
                            self.graph_nodes[dest_key].get_text(),
                            dest_key
                        )
                        self.__add_edge(str(key), str(dest_key), text="Yes")
                    else:
                        self.__add_subgraph(
                            dest_cls,
                            dest_key,
                            last_key=key,
                            text_edge="Yes"
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
    # ------ TEST CASE 'Factorial' ------
    # Create text nodes
    t0 = Node(coordinate=[272,429,25,93], text='inicio')
    t1 = Node(coordinate=[202,494,202,270], text='n=0, cont=2, res=1')
    t2 = Node(coordinate=[281,396,348,397], text='n')
    t3 = Node(coordinate=[264,417,545,602], text='cont <= n')
    t4 = Node(coordinate=[327,426,675,732], text='Yes')
    t5 = Node(coordinate=[251,484,795,848], text='res = res * cont')
    t6 = Node(coordinate=[232,486,925,973], text='cont = cont + 1')
    t7 = Node(coordinate=[464,539,517,613], text='No')
    t8 = Node(coordinate=[619,722,557,628], text='res')
    t9 = Node(coordinate=[767,877,715,787], text='fin')
    # Create shape nodes
    s0 = Node(coordinate=[217,492,14,122], class_shape='start_end')
    s1 = Node(coordinate=[307,372,92,207], class_shape='arrow_line_down')
    s2 = Node(coordinate=[162,544,175,297], class_shape='process')
    s3 = Node(coordinate=[309,362,275,343], class_shape='arrow_line_down')
    s4 = Node(coordinate=[174,526,325,428], class_shape='scan')
    s5 = Node(coordinate=[299,377,408,528], class_shape='arrow_line_down')
    s6 = Node(coordinate=[229,449,502,693], class_shape='decision')
    s7 = Node(coordinate=[292,391,662,790], class_shape='arrow_line_down')
    s8 = Node(coordinate=[211,514,763,865], class_shape='process')
    s9 = Node(coordinate=[317,372,842,920], class_shape='arrow_line_down')
    s10 = Node(coordinate=[206,524,895,1007], class_shape='process')
    s11 = Node(coordinate=[72,246,940,1010], class_shape='arrow_line_left')
    s12 = Node(coordinate=[37,132,567,1000], class_shape='arrow_line_up')
    s13 = Node(coordinate=[69,254,553,617], class_shape='arrow_line_right')
    s14 = Node(coordinate=[431,594,542,608], class_shape='arrow_line_right')
    s15 = Node(coordinate=[587,799,517,657], class_shape='print')
    s16 = Node(
        coordinate=[624,771,635,773],
        class_shape='arrow_rectangle_right',
        #image_path="/home/david/Escritorio/samples_flowcharts/factorial_rect1.jpg"
        image_path="path/file.jpg"
    )
    s17 = Node(coordinate=[749,949,673,792], class_shape='start_end')

    filename = 'factorial.dot'
    graph = Graph(
        [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9],
        [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17]
    )

    flow = graph.generate_graph()
    #flow = aux_flow() # use when graph is not good constructed
    print(flow)
    flow_gen = FlowchartGenerator(graph, flow, filename=filename)
    flow_gen.generate_flowchart()
