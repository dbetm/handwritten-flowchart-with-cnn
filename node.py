# -*- coding: utf-8 -*-

class Node(object):
    """Node represents a simple element of the flowchart."""

    def __init__(self,coordinate,text=None,class_shape=None,image_path=None):
        self.coordinate = coordinate
        self.text = text
        self.class_shape = class_shape
        self.image_path = image_path

    def get_coordinate(self):
        """Returns a tuple (x1, x2, y1, y2)."""

        return self.coordinate

    def get_text(self):
        return self.text

    def get_class(self):
        """Return class of shape."""

        return self.class_shape

    def set_coordinate(self,coordinate):
        self.coordinate = coordinate

    def set_text(self,text):
        self.text = text

    def set_class(self,class_shape):
        self.class_shape = class_shape

    #Change in the class diagram
    def get_type(self):
        """Return type of node (None, text, shape or connector)."""

        if(self.class_shape == None and self.text == None):
            return None
        if(self.class_shape == None):
            return 'text'
        if(self.text == None):
            return 'connector'
        return 'shape'

    def get_image_path(self):
        """Return the path of the cropped node (only rectangle arrows)."""

        return self.image_path

    def __str__(self):
        return "Node(coord:"+str(self.coordinate)+",class:"+str(self.class_shape)+",text:"+str(self.text)+")"

    def __repr__(self):
        return "Node(class:"+str(self.class_shape)+",text:"+str(self.text)+")"
        #return "Node(coord:"+str(self.coordinate)+",class:"+str(self.class_shape)+",text:"+str(self.text)+")"
