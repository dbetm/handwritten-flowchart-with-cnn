class Graph(object):
    def __init__(self,text_nodes,shape_nodes):
        self.text_nodes = text_nodes
        self.shape_nodes = shape_nodes
        self.V = len(text_nodes) + len(shape_nodes)
        self.ad_list = {key: [] for key in range(self.V)}
    def is_collapse(self,A,B):
        #[xmin,xmax,ymin,ymax]
        """
        This function return if exist a interseccion in two nodes
        """
        coordinateA = A.get_coordinate()
        coordinateB = B.get_coordinate()

        x = max(coordinateA[0], coordinateB[0])
		y = max(coordinateA[1], coordinateB[1])
		w = min(coordinateA[2], coordinateB[2]) - x
		h = min(coordinateA[3], coordinateB[3]) - y
        if w < 0 or h < 0:
			return False
		return w * h > 0
        return False

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
            for j in range(i+1,len(shape_nodes)):
                if(is_collapse(text_nodes[i],shape_nodes[j])):
                    #Set the value of the text of the text_node that are inside of it
                    shape_nodes[j].set_text(text_nodes[i])
                    nodes_to_delate.append(i)
                    break;
        #Delate all the nodes that are inside a shape_nodes
        for i in nodes_to_delate:
            text_nodes.pop(i)
        #Return the nodes
        return shape_nodes.extend(text_nodes)
    def calculate_distance(A,B):
        """
        calculate distance between two rectangles
        1)Fist propuest is:
            -Calculate the center point of the two nodes
            -Calculate the distance between the two center points

        """

    def generate_graph(self):
        """
        Generate the adyacency list of the nodes starting of the relationship of the nodes
        Desicion shape is a special case, because can have two connectors
        """
        nodes = self.collapse_nodes()
