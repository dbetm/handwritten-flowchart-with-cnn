import math
from node import Node
class Graph(object):
    def __init__(self,text_nodes,shape_nodes):
        self.text_nodes = text_nodes
        self.shape_nodes = shape_nodes
        self.adj_list = None
    def is_collapse(self,A,B):
        #[xmin,xmax,ymin,ymax]
        """
        This function return if exist a interseccion in two nodes
        """
        coordinateA = A.get_coordinate()
        coordinateB = B.get_coordinate()

        if(coordinateA[0] > coordinateB[1] or coordinateB[0] > coordinateA[1]):
            return False
        if(coordinateA[3] < coordinateB[2] or coordinateB[3] < coordinateA[2]):
            return False
        return True

    def collapse_nodes(self):
        """
        Check the text nodes that are inside of a shape node
        If a text node are inside or near of a shape node the value of the text
        set de value in shape node
        """
        text_nodes = self.text_nodes
        shape_nodes = self.shape_nodes
        nodes_to_delate = []
        for i in range(len(text_nodes)):
            for j in range(len(shape_nodes)):
                if(self.is_collapse(text_nodes[i],shape_nodes[j])):
                    #Set the value of the text of the text_node that are inside of it
                    shape_nodes[j].set_text(text_nodes[i].get_text())
                    nodes_to_delate.append(text_nodes[i])
                    break;

        #Delate all the nodes that are inside a shape_nodes
        for i in nodes_to_delate:
            text_nodes.remove(i)
        #Return the nodes
        shape_nodes.extend(text_nodes)
        return shape_nodes
    def calculate_distance(self,A,B):
        """
        calculate distance between two rectangles
        1)Fist propuest is:
            -Calculate the center point of the two nodes
            -Calculate the distance between the two center points
        """
        coordinateA = A.get_coordinate()
        coordinateB = B.get_coordinate()

        cx1 = int((coordinateA[0] + coordinateA[1]) / 2)
        cy1 = int((coordinateA[2] + coordinateA[3]) / 2)

        cx2 = int((coordinateB[0] + coordinateB[1]) / 2)
        cy2 = int((coordinateB[2] + coordinateB[3]) / 2)

        return math.sqrt(math.pow(cx1 - cx2,2) + math.pow(cy1-cy2,2))

    def generate_graph(self):
        """
        Generate the adyacency list of the nodes starting of the relationship of the nodes
        Desicion shape is a special case, because can have two connectors
        *Check the cases arrow-rectangle/arrow line
        """
        nodes = self.collapse_nodes()
        print("-----------------all-----------------")
        print(nodes)
        print(len(nodes))
        print("---------------------------------------")
        self.adj_list = {key: [] for key in range(len(nodes))}
        visited_list = [0]*len(nodes)
        for i in range(len(nodes)):
            distances = []
            nodes_prompter = []
            #Calculate the distance between the node[i] and the others nodes
            if(i < len(nodes)-1):
                #Check if nodes[i] is a arrow
                """
                point_to_compare = None
                ac = nodes[i].get_coordinate()
                if(nodes[i].get_class() == "arrow_line_down"):
                    point_to_compare = [(ac[0] + ac[1])/2,ac[3]]
                elif(nodes[i].get_class() == "arrow_line_left"):
                    point_to_compare = [(ac[2] + ac[3])/2,ac[0]]
                elif(nodes[i].get_class() == "arrow_line_right"):
                    point_to_compare = [(ac[2] + ac[3])/2,ac[1]]
                elif(nodes[i].get_class() == "arrow_line_up"):
                    point_to_compare = [(ac[0] + ac[1])/2,ac[2]]
                elif(nodes[i].get_class() == "arrow_rectangle_down"):
                    point_to_compare = []
                elif(nodes[i].get_class() == "arrow_rectangle_left"):

                elif(nodes[i].get_class() == "arrow_rectangle_right"):

                elif(nodes[i].get_class() == "arrow_rectangle_up"):
                """

                for j in range(i+1,len(nodes)):
                    dist = self.calculate_distance(nodes[i],nodes[j])
                    if(visited_list[j] == 0):
                        distances.append(dist)
                        nodes_prompter.append(nodes[j])
                node_distance = list(zip(distances,nodes_prompter))
                node_distance = sorted(node_distance)
                relationship = []
                if(nodes[i] == "decision"):
                    relationship = [x[1] for x in node_distance[0:1]]
                else:
                    relationship.append(node_distance[0][1])
                #relationship have the nodes or node that have adyacency
                self.adj_list[i].extend(relationship)
                visited_list[i] = 1
        print("-----------------adj_list-----------------")
        print(self.adj_list)
        print("------------------------------------------")


s1 = Node(coordinate = [302,484,45,124],class_shape = "start_end")
s2 = Node(coordinate = [380,412,127,182],class_shape = "arrow_line_down")
s3 = Node(coordinate = [282,497,177,263],class_shape = "process")
s4 = Node(coordinate = [376,412,249,302],class_shape = "arrow_line_down")
s5 = Node(coordinate = [286,502,303,390],class_shape = "print")
s6 = Node(coordinate = [378,421,392,482],class_shape = "arrow_line_down")
s7 = Node(coordinate = [292,537,479,580],class_shape = "process")
s8 = Node(coordinate = [404,427,576,634],class_shape = "arrow_line_down")
s9 = Node(coordinate = [319,517,635,751],class_shape = "print")
s10 = Node(coordinate = [403,447,708,881],class_shape = "arrow_line_down")
s11 = Node(coordinate = [312,537,881,984],class_shape = "start_end")

t1 = Node(coordinate = [339,435,70,102],text = "inicio")
t2 = Node(coordinate = [336,463,199,232],text = "x=x+1")
t3 = Node(coordinate = [370,404,339,369],text = "x")
t4 = Node(coordinate = [350,477,494,536],text = "x=x/2")
t5 = Node(coordinate = [367,477,650,692],text = "x+5")
t6 = Node(coordinate = [374,452,912,959],text = "fin")

g = Graph([t1,t2,t3,t4,t5,t6],[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11])
g.generate_graph()
