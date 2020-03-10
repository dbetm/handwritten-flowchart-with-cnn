class Nodo(object):
    def __init__(self,coordinate,text=None,class=None):
        self.coordinate = coordinate
        self.text = text
        self.class = class
    def get_coordinate(self):
        return self.coordinate
    def get_text(self):
        return self.text
    def get_class(self):
        return self.class
    def set_coordinate(self,coordinate):
        self.coordinate = coordinate
    def set_text(self,text):
        self.text = text
    def set_class(self,class):
        self.class = class
    #Change in the class diagram 
    def get_type(self):
        if(self.class = None)return 'text'
        if(self.text = None)return 'connector'
        return 'shape'
