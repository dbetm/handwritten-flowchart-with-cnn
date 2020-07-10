import math
import cv2
from node import Node

class Graph(object):

    def __init__(self,text_nodes,shape_nodes):
        self.text_nodes = text_nodes
        self.shape_nodes = shape_nodes
        self.nodes = None
        self.adj_list = None
        self.visited_list = None
    def __exist_character(self,cordA,cordB):
        image = self.image
        xmin_A,xmax_A,ymin_A,ymax_A = cordA
        xmin_B,xmax_B,ymin_B,ymax_B = cordB
        for y in range(ymin_A,ymax_A):
            for x in range(xmin_A,xmax_A):
                if((y < ymin_B) and (y > ymax_B) and (x < xmin_B) and (x > xmax_B)):
                    if(image[y,x] == 255):
                        return True
        return False

    def __is_collapse(self,A,B):
        """ This function return if exist a interseccion in two nodes.
        """
        #[xmin,xmax,ymin,ymax]
        if(type(A) is list):
            coordinateA = A
        else:
            coordinateA = A.get_coordinate()
        coordinateB = B.get_coordinate()
        x = max(coordinateA[0], coordinateB[0])
        y = max(coordinateA[2], coordinateB[2])
        w = min(coordinateA[1], coordinateB[1]) - x
        h = min(coordinateA[3], coordinateB[3]) - y
        if w < 0 or h < 0:
        	return False
        return True


    def __collapse_nodes(self):
        """
        Check the text nodes that are inside of a shape node
        If a text node are inside or near of a shape node the value of the text
        set de value in shape node
        *check if exist more than one overlaping text nodes if is the case calculate the distance and the less distance it will be the true text in te shape node.
        """
        text_nodes = self.text_nodes
        shape_nodes = self.shape_nodes
        #check if exist characters or thers text nodes near to collapse

        nodes_to_delate = []
        collapse_list = [None]*len(text_nodes)
        for i in range(len(shape_nodes)):
            for j in range(len(text_nodes)):
                if(self.__is_collapse(shape_nodes[i],text_nodes[j])):
                    #Check if exist more than one
                    #Set the value of the text of the text_node that are inside of it
                    if(collapse_list[j] == None):
                        shape_nodes[i].set_text(text_nodes[j].get_text())
                        nodes_to_delate.append(text_nodes[j])
                        collapse_list[j] = i
                        break;
                    else:
                        if(self.__calculate_distance(shape_nodes[i],text_nodes[j]) < self.__calculate_distance(text_nodes[j],shape_nodes[collapse_list[j]])):
                            shape_nodes[collapse_list[j]].set_text(None)
                            shape_nodes[i].set_text(text_nodes[j].get_text())
                            collapse_list[j] == i
                            break;
        #Delate all the nodes that are inside a shape_nodes
        for i in nodes_to_delate:
            text_nodes.remove(i)
        if(len(text_nodes)>0):
            #Second iteration to collapse nodes to text with arrows
            nodes_to_delate = []
            for i in range(len(text_nodes)):
                min_ditance = float('inf')
                min_node = None
                for j in range(len(shape_nodes)):
                    if(shape_nodes[j].get_class() in ["arrow_line_down","arrow_rectangle_down"]):
                        ac = shape_nodes[j].get_coordinate()
                        if(shape_nodes[j].get_class() == "arrow_line_down"):
                            point_to_compare = [(ac[0] + ac[1])/2,(ac[2] + ac[3])/2]
                        if(shape_nodes[j].get_class() == "arrow_rectangle_down"):
                            point_to_compare = [(ac[0] + ac[1])/2,(ac[2] + ac[3])/2]
                        dist = self.__calculate_distance(point_to_compare,text_nodes[i])
                        if(dist < min_ditance):
                            min_ditance = dist
                            min_node = j
                shape_nodes[min_node].set_text(text_nodes[i].get_text())
                nodes_to_delate.append(text_nodes[i])
            for i in nodes_to_delate:
                text_nodes.remove(i)
        return shape_nodes

    def __calculate_distance(self,A,B):
        """Calculate distance between two rectangles
        1)First propuest is:
            - Calculate the center point of the two nodes.
            - Calculate the distance between the two center points.
        """

        if(type(A) is list):
            cx1 = A[0]
            cy1 = A[1]
        else:
            coordinateA = A.get_coordinate()
            cx1 = int((coordinateA[0] + coordinateA[1]) / 2)
            cy1 = int((coordinateA[2] + coordinateA[3]) / 2)
        coordinateB = B.get_coordinate()
        cx2 = int((coordinateB[0] + coordinateB[1]) / 2)
        cy2 = int((coordinateB[2] + coordinateB[3]) / 2)

        return math.sqrt(math.pow(cx1 - cx2,2) + math.pow(cy1-cy2,2))

    def find_first_state(self):
        for node in self.nodes:
            if(node.get_class() == "start_end" and node.get_text().lower() == "inicio"):
                return self.nodes.index(node)
        return -1

    def __is_any_arrow(self,node):
        return node.get_class().split('_')[0] == "arrow"

    def __is_graph_visited(self):
        return (sum(x == 1 for x in self.visited_list) == len(self.visited_list))

    def __can_visit(self,previous_node,node_index):
        if(self.nodes[node_index].get_class() == "decision"):
            return self.visited_list[node_index] <= 1 and not(previous_node in self.adj_list[node_index])
        elif(self.nodes[node_index].get_class() == "start_end" and self.nodes[node_index].get_text().lower() == "fin"):
            return not(previous_node in self.adj_list[node_index])
        else:
            return self.visited_list[node_index] == 0 and not(previous_node in self.adj_list[node_index])

    def __find_next(self,node_index):
        if(not(self.__is_graph_visited())):
            #calculate the distance with another nodes
            distances = []
            nodes_prompter = []
            min_distance = float('inf')
            min_node = None
            to_compare = None
            #check only with the posibles
            #if is start end:start
            if(self.nodes[node_index].get_class() == "start_end" and self.nodes[node_index].get_text().lower() == "inicio"):
                for i in range(len(self.nodes)):
                    if(node_index != i and self.__can_visit(node_index,i)):
                        distance = self.__calculate_distance(self.nodes[node_index],self.nodes[i])
                        if(distance < min_distance):
                            min_distance = distance
                            min_node = i
                #min_node is the most near node
                if(self.__is_any_arrow(self.nodes[min_node])):
                    #add the adyacency
                    self.adj_list[node_index].append(min_node)
                    self.visited_list[node_index] += 1
                    return self.__find_next(min_node)
                else:
                    return "NV"
            #if is any arrow
            #check the type of arrow to predict better
            elif(self.__is_any_arrow(self.nodes[node_index])):
                #find the point of check the distance between the other node
                point_to_compare = None
                ac = self.nodes[node_index].get_coordinate()
                if(self.nodes[node_index].get_class() == "arrow_line_down"):
                    point_to_compare = [(ac[0] + ac[1])/2,ac[3]]
                elif(self.nodes[node_index].get_class() == "arrow_line_left"):
                    point_to_compare = [ac[0],(ac[2] + ac[3])/2]
                elif(self.nodes[node_index].get_class() == "arrow_line_right"):
                    point_to_compare = [ac[1],(ac[2] + ac[3])/2]
                elif(self.nodes[node_index].get_class() == "arrow_line_up"):
                    point_to_compare = [(ac[0] + ac[1])/2,ac[2]]
                for i in range(len(self.nodes)):
                    if(node_index != i and self.__can_visit(node_index,i)):
                        #calculate the distance
                        distance = self.__calculate_distance(point_to_compare,self.nodes[i])
                        if(distance < min_distance):
                            min_distance = distance
                            min_node = i
                self.adj_list[node_index].append(min_node)
                self.visited_list[node_index] += 1
                return self.__find_next(min_node)
            #if is process,print, or scan
            elif(self.nodes[node_index].get_class() == "print" or self.nodes[node_index].get_class() == "process" or self.nodes[node_index].get_class() == "scan"):
                    for i in range(len(self.nodes)):
                        if(node_index != i and self.__can_visit(node_index,i)):
                            distance = self.__calculate_distance(self.nodes[node_index],self.nodes[i])
                            if(distance < min_distance):
                                min_distance = distance
                                min_node = i
                    #min_node is the most near node
                    if(self.__is_any_arrow(self.nodes[min_node])):
                        #add the adyacency
                        self.adj_list[node_index].append(min_node)
                        self.visited_list[node_index] += 1
                        return self.__find_next(min_node)
                    else:
                        return "NV"
            #if is decision
            elif(self.nodes[node_index].get_class() == "decision"):
                if(self.visited_list[node_index]==0):
                    for i in range(len(self.nodes)):
                        if(node_index != i and self.__can_visit(node_index,i) and self.nodes[i].get_text() != None):
                            distance = self.__calculate_distance(self.nodes[node_index],self.nodes[i])
                            distances.append(distance)
                            nodes_prompter.append(i)
                    node_distance = list(zip(distances,nodes_prompter))
                    node_distance = sorted(node_distance)
                    to_check = node_distance[0:2]
                    if(self.__is_any_arrow(self.nodes[to_check[0][1]]) and self.__is_any_arrow(self.nodes[to_check[1][1]])):
                        self.adj_list[node_index].append(to_check[0][1])
                        self.adj_list[node_index].append(to_check[1][1])
                        self.visited_list[node_index] += 1
                        a = self.__find_next(to_check[0][1])
                        b = self.__find_next(to_check[1][1])
                    else:
                        return str(node_index)+"NV"

    def generate_graph(self):
        """ Generate the adyacency list of the nodes starting of the relationship of
        the nodes. Decision shape is a special case, because can have two connectors
        *Check the cases arrow-rectangle/arrow line.
        """
        self.nodes = self.__collapse_nodes()
        print("nodes",list(enumerate(self.nodes)))
        self.adj_list = {key: [] for key in range(len(self.nodes))}
        self.visited_list = [0]*len(self.nodes)
        first_state = self.find_first_state()
        if(first_state == -1):
            return "Not valid init"
        if(self.__find_next(first_state) == "NV"):
            print(self.adj_list)
            return "Not valid"
        print("adj_list",self.adj_list)
        return self.adj_list

    def get_adyacency_list(self):
        return self.adj_list

    def get_nodes(self):
        return self.nodes
