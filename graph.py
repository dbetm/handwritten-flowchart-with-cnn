import math
from node import Node
import cv2
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
        *check if exist more than one overlaping text nodes if is the case calculate the distance
        and the less distance it will be the true text in te shape node
        """
        text_nodes = self.text_nodes
        shape_nodes = self.shape_nodes
        nodes_to_delate = []
        collapse_list = [None]*len(shape_nodes)
        for i in range(len(text_nodes)):
            for j in range(len(shape_nodes)):
                if(self.is_collapse(text_nodes[i],shape_nodes[j])):
                    #Check if exist more than one
                    #Set the value of the text of the text_node that are inside of it
                    if(collapse_list[j] == None):
                        shape_nodes[j].set_text(text_nodes[i].get_text())
                        nodes_to_delate.append(text_nodes[i])
                        collapse_list[j] = i
                        break;
                    else:
                        if(self.calculate_distance(shape_nodes[j],text_nodes[i]) < self.calculate_distance(shape_nodes[j],text_nodes[collapse_list[j]])):
                            shape_nodes[j].set_text(text_nodes[i].get_text())
                            nodes_to_delate.append(text_nodes[i])
                            nodes_to_delate.remove(text_nodes[collapse_list[j]])
                            collapse_list[j] == i
                            break;


        #Delate all the nodes that are inside a shape_nodes
        for i in nodes_to_delate:
            text_nodes.remove(i)
        print("leftovers",text_nodes)
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
                        dist = self.calculate_distance(point_to_compare,text_nodes[i])
                        if(dist < min_ditance):
                            min_ditance = dist
                            min_node = j
                print("min_node",min_node,text_nodes[i])
                shape_nodes[min_node].set_text(text_nodes[i].get_text())
                nodes_to_delate.append(text_nodes[i])
            for i in nodes_to_delate:
                text_nodes.remove(i)
        #Return the nodes
        #shape_nodes.extend(text_nodes)
        return shape_nodes
    def calculate_distance(self,A,B):
        """
        calculate distance between two rectangles
        1)Fist propuest is:
            -Calculate the center point of the two nodes
            -Calculate the distance between the two center points
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
    def rigth_or_left(self,img_path):
        img = cv2.imread(img_path,0)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        xs,pix_n = 0,0
        xmin = float('inf')
        xmax = float('-inf')
        w,h = img.shape
        for y in range(w):
            for x in range(h):
                if(img[y,x] == 0):
                    xmin = min(x,xmin)
                    xmax = max(x,xmax)
                    xs += x
                    pix_n += 1
        meanX = xs/pix_n
        if(meanX < ((xmax - xmin)/2) + xmin):
            return "left"
        else:
            return "right"
    def generate_graph(self):
        """
        Generate the adyacency list of the nodes starting of the relationship of the nodes
        Desicion shape is a special case, because can have two connectors
        *Check the cases arrow-rectangle/arrow line
        """
        nodes = self.collapse_nodes()
        print("-----------------all-----------------")
        print(list(enumerate(nodes)))
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
                #Arrow rectangles
                elif(nodes[i].get_class() == "arrow_rectangle_down"):
                    if(self.rigth_or_left(nodes[i].get_image_path()) == "left"):
                        point_to_compare = [ac[0],ac[3]]
                    else:
                        point_to_compare = [ac[1],ac[3]]
                elif(nodes[i].get_class() == "arrow_rectangle_left"):
                    if(self.rigth_or_left(nodes[i].get_image_path()) == "left"):
                        point_to_compare = [ac[0],ac[2]]
                    else:
                        point_to_compare = [ac[1],ac[2]]
                elif(nodes[i].get_class() == "arrow_rectangle_right"):
                    if(self.rigth_or_left(nodes[i].get_image_path()) == "left"):
                        point_to_compare = [ac[0],ac[3]]
                    else:
                        point_to_compare = [ac[1],ac[3]]
                elif(nodes[i].get_class() == "arrow_rectangle_up"):
                    if(self.rigth_or_left(nodes[i].get_image_path()) == "left"):
                        point_to_compare = [ac[0],ac[2]]
                    else:
                        point_to_compare = [ac[1],ac[2]]


                for j in range(i+1,len(nodes)):
                    if(nodes[i].get_class().split('_')[0] == "arrow"):
                        dist = self.calculate_distance(point_to_compare,nodes[j])
                    else:
                        dist = self.calculate_distance(nodes[i],nodes[j])
                    if(visited_list[j] == 0):
                        distances.append(dist)
                        nodes_prompter.append(nodes[j])
                node_distance = list(zip(distances,nodes_prompter))
                node_distance = sorted(node_distance)
                relationship = []
                if(nodes[i].get_class() == "decision"):
                    relationship = [nodes.index(x[1]) for x in node_distance[0:2]]
                else:
                    relationship.append(nodes.index(node_distance[0][1]))
                #relationship have the nodes or node that have adyacency
                self.adj_list[i].extend(relationship)
                visited_list[i] = 1
        print("-----------------adj_list-----------------")
        print(self.adj_list)
        print("------------------------------------------")

"""
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

"""
t1 = Node(coordinate = [384,600,123,189],text = "inicio")
t2 = Node(coordinate = [375,597,378,438],text = "x=6")
t3 = Node(coordinate = [429,570,678,750],text = "x>5")
t4 = Node(coordinate = [402,579,960,1008],text = "verdad")
t5 = Node(coordinate = [783,930,948,1008],text = "falso")
t6 = Node(coordinate = [426,546,1341,1410],text = "fin")
t7 = Node(coordinate = [705,825,606,666],text = "si")
t8 = Node(coordinate = [363,423,858,906],text = "no")

s1 = Node(coordinate = [318,675,81,231],class_shape = "start_end")
s2 = Node(coordinate = [456,513,237,354],class_shape = "arrow_line_down")
s3 = Node(coordinate = [306,684,354,489],class_shape = "process")
s4 = Node(coordinate = [462,522,483,594],class_shape = "arrow_line_down")
s5 = Node(coordinate = [345,615,588,858],class_shape = "decision")
s6 = Node(coordinate = [612,921,702,939],class_shape = "arrow_rectangle_down",image_path="graph_images/rect_down.png")
s7 = Node(coordinate = [438,504,849,954],class_shape = "arrow_line_down")
s8 = Node(coordinate = [372,645,948,1125],class_shape = "print")
s9 = Node(coordinate = [741,1059,936,1095],class_shape = "print")
s10 = Node(coordinate = [420,471,1119,1320],class_shape = "arrow_line_down")
s11 = Node(coordinate = [669,849,1092,1410],class_shape = "arrow_rectangle_right",image_path="graph_images/rect_right.png")
s12 = Node(coordinate = [339,669,1317,1455],class_shape = "start_end")


g = Graph([t1,t2,t3,t4,t5,t6,t7,t8],[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12])
g.generate_graph()
