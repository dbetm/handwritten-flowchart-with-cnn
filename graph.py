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
        print("to delete",nodes_to_delate)
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
        """
        nodes = self.collapse_nodes()
        self.adj_list = {key: [] for key in range(len(nodes))}
        print(nodes)
        print(self.adj_list)
        for i in range(len(nodes)):
            distances = []
            nodes_prompter = []
            #Calculate the distance between the node[i] and the others nodes
            if(i < len(nodes)-1):
                for j in range(i+1,len(nodes)):
                    dist = self.calculate_distance(nodes[i],nodes[j])
                    distances.append(dist)
                    nodes_prompter.append(nodes[j])
                node_distance = list(zip(distances,nodes_prompter))
                node_distance = sorted(node_distance)
                relationship = []
                if(nodes[i] == "desicion"):
                    relationship = [x[1] for x in node_distance[0:1]]
                else:
                    relationship.append(node_distance[0][1])
                #relationship have the nodes or node that have adyacency
                self.adj_list[i].extend(relationship)
        print(self.adj_list)


s1 = Node(coordinate = [4,10,2,6],class_shape = "print")
s2 = Node(coordinate = [4,10,2+15,6+15],class_shape = "connector")
s3 = Node(coordinate = [4,10,2+30,6+30],class_shape = "print")

t1 = Node(coordinate = [5,9,3,5],text = "holaaaa")
t2 = Node(coordinate = [5,9,3+15,5+15],text = "holaaaa2")
t3 = Node(coordinate = [5,9,3+30,5+30],text = "holaaaa3")
g = Graph([t1,t2,t3],[s1,s2,s3])
g.generate_graph()
