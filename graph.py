import math
import cv2
from node import Node

class Graph(object):

    def __init__(self,image_path,text_nodes,shape_nodes):
        self.image_path = image_path
        self.__set_image()
        self.text_nodes = text_nodes
        self.shape_nodes = shape_nodes
        #print("-----------------------text nodes",self.text_nodes)
        #print("-----------------------shape nodes",self.shape_nodes)
        self.nodes = None
        self.adj_list = None
        self.visited_list = None
    def __set_image(self):
        image = cv2.imread(self.image_path,0)
        blur = cv2.GaussianBlur(image,(5,5),0)
        ret3,image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = (255 - image)
        self.image = image
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

    def is_collapse(self,A,B):
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
        """if(coordinateA[0] > coordinateB[1] or coordinateB[0] > coordinateA[1]):
            return False
        if(coordinateA[3] < coordinateB[2] or coordinateB[3] < coordinateA[2]):
            return False
        return True"""

    def collapse_nodes(self):
        """
        Check the text nodes that are inside of a shape node
        If a text node are inside or near of a shape node the value of the text
        set de value in shape node
        *check if exist more than one overlaping text nodes if is the case calculate the distance and the less distance it will be the true text in te shape node.
        """

        text_nodes = self.text_nodes
        shape_nodes = self.shape_nodes
        #check if exist characters or thers text nodes near to collapse
        print("-----------antes-----------",text_nodes)
        EXPAND_MAX = 10 #pixels
        expand_max = EXPAND_MAX
        for i in text_nodes:
            xmin,xmax,ymin,ymax = i.get_coordinate()
            expand_max = EXPAND_MAX
            print("------------------prueba------------------",text_nodes)
            while(True):
                xmin = xmin - 1
                xmax = xmax + 1
                ymin = ymin - 1
                ymax = ymax + 1
                expand_max -= 1
                if(expand_max == 0):
                    break
                for j in text_nodes:
                    if i != j:
                        if(self.is_collapse([xmin,xmax,ymin,ymax],j)):
                            #Collapse nodes and coordinates
                            print("Colapsaron",i,j)
                            xmin_A,xmax_A,ymin_A,ymax_A = i.get_coordinate()
                            xmin_B,xmax_B,ymin_B,ymax_B = j.get_coordinate()
                            n_xmin = min(xmin_A,xmin_B)
                            n_xmax = max(xmax_A,xmax_B)
                            n_ymin = min(ymin_A,ymin_B)
                            n_ymax = max(ymax_A,ymax_B)
                            i.set_coordinate([n_xmin,n_xmax,n_ymin,n_ymax])
                            n_text = ""
                            if xmin_A < xmin_B:
                                n_text += i.get_text() + " " + j.get_text()
                            else:
                                n_text += j.get_text() + " " + i.get_text()
                            i.set_text(n_text)
                            text_nodes.remove(j)
                            break
            xmin,xmax,ymin,ymax = i.get_coordinate()
            expand_max = EXPAND_MAX

            """while(True):
                xmin = xmin - 1
                xmax = xmax + 1
                ymin = ymin - 1
                ymax = ymax + 1
                expand_max -= 1
                if(expand_max == 0):
                    break
                query = self.__exist_character([xmin,xmax,ymin,ymax],i.get_coordinate())
                if(query == True):
                    while(self.__exist_character([xmin,xmax,ymin,ymax],i.get_coordinate())):
                        xmin = xmin - 1
                        xmax = xmax + 1
                        ymin = ymin - 1
                        ymax = ymax + 1
                        expand_max -= 1
                    i.set_coordinate([xmin,xmax,ymin,ymax])
                    break"""

        print("-----------despues-----------",text_nodes)
        nodes_to_delate = []
        collapse_list = [None]*len(text_nodes)
        for i in range(len(shape_nodes)):
            for j in range(len(text_nodes)):
                if(self.is_collapse(shape_nodes[i],text_nodes[j])):
                    #Check if exist more than one
                    #Set the value of the text of the text_node that are inside of it
                    if(collapse_list[j] == None):
                        shape_nodes[i].set_text(text_nodes[j].get_text())
                        nodes_to_delate.append(text_nodes[j])
                        collapse_list[j] = i
                        break;
                    else:
                        if(self.calculate_distance(shape_nodes[i],text_nodes[j]) < self.calculate_distance(text_nodes[j],shape_nodes[collapse_list[j]])):
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
                        dist = self.calculate_distance(point_to_compare,text_nodes[i])
                        if(dist < min_ditance):
                            min_ditance = dist
                            min_node = j
                shape_nodes[min_node].set_text(text_nodes[i].get_text())
                nodes_to_delate.append(text_nodes[i])
            for i in nodes_to_delate:
                text_nodes.remove(i)
        #Return the nodes
        #shape_nodes.extend(text_nodes)
        return shape_nodes

    def calculate_distance(self,A,B):
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

    def rigth_or_left(self,img_path):
        img = cv2.imread(img_path,0)
        #cv2.imshow("imagen",img)
        #cv2.waitKey(0)
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

    def find_first_state(self):
        for node in self.nodes:
            if(node.get_class() == "start_end" and node.get_text() == "inicio"):
                return self.nodes.index(node)
        return -1

    def is_any_arrow(self,node):
        return node.get_class().split('_')[0] == "arrow"

    def is_graph_visited(self):
        return (sum(x == 1 for x in self.visited_list) == len(self.visited_list))

    def can_visit(self,previous_node,node_index):
        if(self.nodes[node_index].get_class() == "decision"):
            return self.visited_list[node_index] <= 1 and not(previous_node in self.adj_list[node_index])
        elif(self.nodes[node_index].get_class() == "start_end" and self.nodes[node_index].get_text() == "fin"):
            return not(previous_node in self.adj_list[node_index])
        else:
            return self.visited_list[node_index] == 0 and not(previous_node in self.adj_list[node_index])

    def find_next(self,node_index):
        if(not(self.is_graph_visited())):
            #calculate the distance with another nodes
            distances = []
            nodes_prompter = []
            min_distance = float('inf')
            min_node = None
            to_compare = None
            #check only with the posibles
            #if is start end:start
            if(self.nodes[node_index].get_class() == "start_end" and self.nodes[node_index].get_text() == "inicio"):
                for i in range(len(self.nodes)):
                    if(node_index != i and self.can_visit(node_index,i)):
                        distance = self.calculate_distance(self.nodes[node_index],self.nodes[i])
                        if(distance < min_distance):
                            min_distance = distance
                            min_node = i
                #min_node is the most near node
                if(self.is_any_arrow(self.nodes[min_node])):
                    #add the adyacency
                    self.adj_list[node_index].append(min_node)
                    self.visited_list[node_index] += 1
                    return self.find_next(min_node)
                else:
                    return "NV"
            #if is any arrow
            #check the type of arrow to predict better
            elif(self.is_any_arrow(self.nodes[node_index])):
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
                #Arrow rectangles
                elif(self.nodes[node_index].get_class() == "arrow_rectangle_down"):
                    if(self.rigth_or_left(self.nodes[node_index].get_image_path()) == "left"):
                        point_to_compare = [ac[0],ac[3]]
                    else:
                        point_to_compare = [ac[1],ac[3]]
                elif(self.nodes[node_index].get_class() == "arrow_rectangle_left"):
                    if(self.rigth_or_left(self.nodes[node_index].get_image_path()) == "left"):
                        point_to_compare = [ac[0],ac[2]]
                    else:
                        point_to_compare = [ac[1],ac[2]]
                elif(self.nodes[node_index].get_class() == "arrow_rectangle_right"):
                    if(self.rigth_or_left(self.nodes[node_index].get_image_path()) == "left"):
                        point_to_compare = [ac[0],ac[3]]
                    else:
                        point_to_compare = [ac[1],ac[3]]
                elif(self.nodes[node_index].get_class() == "arrow_rectangle_up"):
                    if(self.rigth_or_left(self.nodes[node_index].get_image_path()) == "left"):
                        point_to_compare = [ac[0],ac[2]]
                    else:
                        point_to_compare = [ac[1],ac[2]]
                for i in range(len(self.nodes)):
                    if(node_index != i and self.can_visit(node_index,i)):
                        #calculate the distance
                        distance = self.calculate_distance(point_to_compare,self.nodes[i])
                        if(distance < min_distance):
                            min_distance = distance
                            min_node = i
                self.adj_list[node_index].append(min_node)
                self.visited_list[node_index] += 1
                return self.find_next(min_node)
            #if is process,print, or scan
            elif(self.nodes[node_index].get_class() == "print" or self.nodes[node_index].get_class() == "process" or self.nodes[node_index].get_class() == "scan"):
                    for i in range(len(self.nodes)):
                        if(node_index != i and self.can_visit(node_index,i)):
                            distance = self.calculate_distance(self.nodes[node_index],self.nodes[i])
                            if(distance < min_distance):
                                min_distance = distance
                                min_node = i
                    #min_node is the most near node
                    if(self.is_any_arrow(self.nodes[min_node])):
                        #add the adyacency
                        self.adj_list[node_index].append(min_node)
                        self.visited_list[node_index] += 1
                        return self.find_next(min_node)
                    else:
                        return "NV"
            #if is decision
            elif(self.nodes[node_index].get_class() == "decision"):
                if(self.visited_list[node_index]==0):
                    for i in range(len(self.nodes)):
                        if(node_index != i and self.can_visit(node_index,i)):
                            distance = self.calculate_distance(self.nodes[node_index],self.nodes[i])
                            distances.append(distance)
                            nodes_prompter.append(i)
                    node_distance = list(zip(distances,nodes_prompter))
                    node_distance = sorted(node_distance)
                    to_check = node_distance[0:2]
                    if(self.is_any_arrow(self.nodes[to_check[0][1]]) and self.is_any_arrow(self.nodes[to_check[1][1]])):
                        self.adj_list[node_index].append(to_check[0][1])
                        self.adj_list[node_index].append(to_check[1][1])
                        self.visited_list[node_index] += 1
                        a = self.find_next(to_check[0][1])
                        b = self.find_next(to_check[1][1])
                    else:
                        return str(node_index)+"NV"

    def generate_graph(self):
        """ Generate the adyacency list of the nodes starting of the relationship of
        the nodes. Decision shape is a special case, because can have two connectors
        *Check the cases arrow-rectangle/arrow line.
        """

        self.nodes = self.collapse_nodes()
        #print("-----------------all-----------------")
        print("collapse_nodes",list(enumerate(self.nodes)))
        #print(len(self.nodes))
        #print("---------------------------------------")
        self.adj_list = {key: [] for key in range(len(self.nodes))}
        self.visited_list = [0]*len(self.nodes)
        first_state = self.find_first_state()
        if(first_state == -1):
            return "Not valid init"
        if(self.find_next(first_state) == "NV"):
            print(self.adj_list)
            return "Not valid"
        return self.adj_list

    def get_adyacency_list(self):
        return self.adj_list

    def get_nodes(self):
        return self.nodes

"""
s1 = Node(coordinate = [302,484,45,124],class_shape =self.visited_list "start_end")
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
"""
"""
t1 = Node(coordinate = [504,1344,334,436],text = "ans,n,a=0,b=1,cont=2")
t2 = Node(coordinate = [1068,1318,96,212],text = "inicio")
t3 = Node(coordinate = [772,846,642,694],text = "n")
t4 = Node(coordinate = [170,312,630,716],text = "fin")
t5 = Node(coordinate = [660,896,1006,1090],text = "cont<n")
t6 = Node(coordinate = [450,554,964,1036],text = "no")
t7 = Node(coordinate = [132,270,1028,1086],text = "ans")
t8 = Node(coordinate = [828,888,1250,1326],text = "si")
t9 = Node(coordinate = [696,974,1452,1528],text = "ans=a+b")
t10 = Node(coordinate = [734,878,1760,1824],text = "a=b")
t11 = Node(coordinate = [728,948,1992,2058],text = "b=ans")
t12 = Node(coordinate = [664,1106,2232,2302],text = "cont=cont+1")


s1 = Node(coordinate = [950,1413,80,148],class_shape = "start_end")
s2 = Node(coordinate = [717,957,113,309],class_shape = "arrow_rectangle_down",image_path="graph_images/set9/rectdown.png")
s3 = Node(coordinate = [468,1361,303,453],class_shape = "process")
s4 = Node(coordinate = [798,867,441,611],class_shape = "arrow_line_down")
s5 = Node(coordinate = [572,1020,609,726],class_shape = "scan")
s6 = Node(coordinate = [74,434,614,742],class_shape = "start_end")
s7 = Node(coordinate = [780,852,738,926],class_shape = "arrow_line_down")
s8 = Node(coordinate = [176,262,786,1006],class_shape = "arrow_line_up")
s9 = Node(coordinate = [80,328,998,1172],class_shape = "print")
s10 = Node(coordinate = [334,630,1022,1082],class_shape = "arrow_line_left")
s11 = Node(coordinate = [626,964,914,1230],class_shape = "decision")
s12 = Node(coordinate = [962,1324,1008,1742],class_shape = "arrow_rectangle_left",image_path="graph_images/set9/rectleft.png")
s13 = Node(coordinate = [780,830,1224,1434],class_shape = "arrow_line_down")
s14 = Node(coordinate = [646,1036,1438,1562],class_shape = "process")
s15 = Node(coordinate = [788,838,1570,1726],class_shape = "arrow_line_down")
s16 = Node(coordinate = [646,986,1728,1852],class_shape = "process")
s17 = Node(coordinate = [784,836,1852,1966],class_shape = "arrow_line_down")
s18 = Node(coordinate = [668,1006,1966,2088],class_shape = "process")
s19 = Node(coordinate = [780,836,2086,2216],class_shape = "arrow_line_down")
s20 = Node(coordinate = [602,1140,2222,2328],class_shape = "process")
s21 = Node(coordinate = [1154,1330,1740,2252],class_shape = "arrow_rectangle_up",image_path="graph_images/set9/rectup.png")


g = Graph([t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12],[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21])
print("----------------------------------------------------")
print(g.generate_graph())
"""
"""# Order (x1, x2, y1, y2)
    # ------ TEST CASE 'Fibonacci' ------
    # Create text nodes
t0 = Node(coordinate=[934,1219,136,263], text='inicio')
t1 = Node(coordinate=[407,1296,370,486], text='ans,n,a=0,b=1,cont=2')
t2 = Node(coordinate=[730,873,655,759], text='n')
t3 = Node(coordinate=[630,923,951,1090], text='cont < n')
t4 = Node(coordinate=[780,927,1190,1282], text='si')
t5 = Node(coordinate=[569,900,1386,1490], text='ans = a+b')
t6 = Node(coordinate=[627,854,1632,1751], text='a = b')
t7 = Node(coordinate=[592,884,1886,1997], text='b = ans')
t8 = Node(coordinate=[542,996,2194,2294], text='cont = cont+1')
t9 = Node(coordinate=[477,607,901,1047], text='no')
t10 = Node(coordinate=[115,354,982,1109], text='ans')
t11 = Node(coordinate=[127,323,601,717], text='fin')
# Create shape nodes
s0 = Node(coordinate=[838,1342,55,305], class_shape='start_end')
s1 = Node(
    coordinate=[519,880,105,363],
    class_shape='arrow_rectangle_down',
    image_path="graph_images/prueba2/fib_rect0.jpg"
)
s2 = Node(coordinate=[350,1330,320,520], class_shape='process')
s3 = Node(coordinate=[761,892,447,663], class_shape='arrow_rectangle_down')
s4 = Node(coordinate=[573,1015,617,778], class_shape='scan')
s5 = Node(coordinate=[765,873,770,901], class_shape='arrow_line_down')
s6 = Node(coordinate=[619,961,874,1201], class_shape='decision')
s7 = Node(coordinate=[711,884,1147,1386], class_shape='arrow_line_down')
s8 = Node(coordinate=[527,988,1351,1540], class_shape='process')
s9 = Node(coordinate=[696,796,1494,1636], class_shape='arrow_line_down')
s10 = Node(coordinate=[542,954,1590,1782], class_shape='process')
s11 = Node(coordinate=[692,765,1763,188], class_shape='arrow_line_down')
s12 = Node(coordinate=[523,980,1847,2028], class_shape='process')
s13 = Node(coordinate=[650,757,1982,2167], class_shape='arrow_line_down')
s14 = Node(coordinate=[480,1077,2117,2363], class_shape='process')
s15 = Node(
    coordinate=[1027,1407,1582,2294],
    class_shape='arrow_rectangle_up',
    image_path="graph_images/prueba2/fib_rect1.jpg"
)
s16 = Node(
    coordinate=[919,1438,928,1636],
    class_shape='arrow_rectangle_left',
    image_path="graph_images/prueba2/fib_rect2.jpg"
)
s17 = Node(coordinate=[384,661,947,1094], class_shape='arrow_line_left')
s18 = Node(coordinate=[34,407,951,1244], class_shape='print')
s19 = Node(coordinate=[161,304,717,1005], class_shape='arrow_line_up')
s20 = Node(coordinate=[77,496,570,767], class_shape='start_end')
filename = 'fibonacci.dot'
graph = Graph(
    [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11],
    [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20]
)
flow = graph.generate_graph()
print(flow)"""
