class Node(object):
    def __init__(self,coordinate,text=None,class_shape=None,image_path=None):
        self.coordinate = coordinate
        self.text = text
        self.class_shape = class_shape
        self.image_path = image_path
    def get_coordinate(self):
        return self.coordinate
    def get_text(self):
        return self.text
    def get_class(self):
        return self.class_shape
    def set_coordinate(self,coordinate):
        self.coordinate = coordinate
    def set_text(self,text):
        self.text = text
    def set_class(self,class_shape):
        self.class_shape = class_shape
    #Change in the class diagram
    def get_type(self):
        if(self.class_shape == None):
            return 'text'
        if(self.text == None):
            return 'connector'
        return 'shape'
    def get_image_path(self):
        return self.image_path
    def __str__(self):
        return "Node(coord:"+str(self.coordinate)+",class:"+str(self.class_shape)+",text:"+str(self.text)+")"
    def __repr__(self):
        return "Node(class:"+str(self.class_shape)+",text:"+str(self.text)+")"
        #return "Node(coord:"+str(self.coordinate)+",class:"+str(self.class_shape)+",text:"+str(self.text)+")"
