from graph import Graph
from node import Node
import os
class CodeGenerator(object):
    def __init__(self,graph,file_path):
        self.FILE_PATH = "results/"+file_path+"code.c"
        self.adj_list = graph.get_adyacency_list()
        self.nodes = graph.get_nodes()
        self.pos_x = 0
        #ponter list to know how tabs need to white
        self.pointer_x_list = [0]*5
        self.lines_to_write = []
        self.variables = dict()
        self.type_map = {"int":'"%d"',"double":'"%f"',"char":'"%c"'}
        self.__collapse_end_node()
    def __collapse_end_node(self):
        cont = 0
        end_node = None
        nodes_list = []
        for node in range(len(self.nodes)):
            if self.nodes[node].get_class() == "start_end" and self.nodes[node].get_text().lower() == "fin":
                nodes_list.append(node)
                end_node = node
                cont += 1
        nodes_list = nodes_list[:-1]
        if(cont > 1):
            for x in nodes_list:
                for i in self.adj_list.keys():
                    for y in range(len(self.adj_list[i])):
                        if self.adj_list[i][y] == x:
                            self.adj_list[i][y] = end_node

    def __is_any_arrow(self,node):
        return node.get_class().split('_')[0] == "arrow"
    def __generate_tabs(self,pos_x):
        return "    "*pos_x
    def __get_type(self,sentence):
        sentence = sentence.replace(" ", "")
        if sentence in self.variables.keys():
            return self.type_map[self.variables[sentence]]
        tam_data = {"char":1,"int":2,"double":8}
        separated = []
        pos = 0
        i = 0
        while(i < len(sentence)):
            x = sentence[i]
            if(x == '+' or x == '-' or x == '/' or x == '*' or x == '%' or x == '<' or x == '>' or x == '^' or x == '='):
                separated.append(str(sentence[pos:i]))
                pos = i + 1
            i += 1
        separated.append(sentence[pos:len(sentence) + 1])
        for i in range(len(separated)):
            if("(" in separated[i]):
                separated[i] = separated[i].replace('(', '')
            if(")" in separated[i]):
                separated[i] = separated[i].replace(')', '')
        maxi = float('-inf')
        max_data = None
        for i in range(len(separated)):
            if(separated[i][0] == '"' and separated[i][-1] == '"'):
                return '"%s"'
            x = list(filter(lambda x: (x[0] == separated[i]), self.variables))
            if(len(x) > 0):
                if(tam_data[x[0][1]] > maxi):
                    maxi = tam_data[x[0][1]]
                    max_data = x[0][1]
            else:
                if(len(separated[i]) == 1 and separated[i].isalpha()):
                    maxi = tam_data["char"]
                    max_data = "char"
                try:
                    int(separated[i])
                    maxi = tam_data["int"]
                    max_data = "int"
                except ValueError:
                    try:
                        float(separated[i])
                        maxi = tam_data["double"]
                        max_data = "double"
                    except ValueError:
                        return ""

            return self.type_map[max_data]
    def __predict_type(self,sentence):
        sentence = sentence.replace(" ", "")
        def type_variable(s):
            if(len(s) == 1 and s.isalpha()):
                return "char"
            try:
                int(s)
                return "int"
            except ValueError:
                try:
                    float(s)
                    return "float"
                except ValueError:
                    return ""
        if("=" in sentence):
            var = [s.split('=')[0] for s in sentence.split(',')]
            value = [s.split('=')[1] for s in sentence.split(',')]
            pos = 0

            flag = True
            for i in range(len(var)):
                if(not(var[i] in self.variables.keys())):
                    self.variables.update({var[i]:type_variable(value[i])})
                else:
                    flag = False
            if(flag):
                return type_variable(value[0]) +" "+ sentence
            return sentence
        else:
            return sentence
    #Generate the code of the graph
    def generate_code(self,index,end_x):
        #Is is diferent to Not valid
        if(self.adj_list == "Not valid"):
            return False
        #Start to write the code
        self.lines_to_write.append("#include<stdio.h>\n")
        self.lines_to_write.append("int main(){\n")
        self.pos_x += 1
        #Call the function with the next node
        self.generate(self.adj_list[index][0],end_x)
    def generate(self,index,end_x):
        if(end_x != index):
            #Is is a arrow
            if(self.__is_any_arrow(self.nodes[index])):
                #Call the function with the next node
                self.generate(self.adj_list[index][0],end_x)
            #If is a process node
            elif(self.nodes[index].get_class() == "process"):
                self.lines_to_write.append(self.__generate_tabs(self.pos_x)+self.__predict_type(self.nodes[index].get_text())+";\n")
                #Call the function with the next node
                self.generate(self.adj_list[index][0],end_x)
            elif(self.nodes[index].get_class() == "scan"):
                self.lines_to_write.append(self.__generate_tabs(self.pos_x)+'scanf('+self.type_map[self.variables[self.nodes[index].get_text()]]+',&'+self.nodes[index].get_text()+');\n')
                self.generate(self.adj_list[index][0],end_x)
            elif(self.nodes[index].get_class() == "print"):
                #change the form to get tthe type
                self.lines_to_write.append(self.__generate_tabs(self.pos_x)+'printf('+self.__get_type(self.nodes[index].get_text())+','+ self.nodes[index].get_text() +');\n')
                self.generate(self.adj_list[index][0],end_x)
            elif(self.nodes[index].get_class() == "start_end" and self.nodes[index].get_text()):
                self.lines_to_write.append(self.__generate_tabs(self.pos_x)+"return 0;\n");
                self.lines_to_write.append("}\n");

                f = open(self.FILE_PATH, "a+")
                f.writelines(self.lines_to_write)
                f.close()
            elif(self.nodes[index].get_class() == "decision"):
                #find a path tyo the same node
                visited_list = [0]*len(self.nodes)
                def dfs(v,pivote):
                    if(self.adj_list[v] == []):
                        return None
                    next = self.adj_list[v][0]
                    if(next == pivote):
                        return True,v
                    return dfs(next,pivote)

                yes_way = -1
                for i in self.adj_list[index]:
                    if(self.nodes[i].get_text().lower() == "si" or self.nodes[i].get_text().lower() == "yes"):
                        yes_way = self.adj_list[index].index(i)
                ans = dfs(self.adj_list[index][yes_way],index)
                visited_list = [0]*len(self.nodes)
                if(ans == None):
                    #it is a if
                    self.lines_to_write.append(self.__generate_tabs(self.pos_x)+"if("+self.nodes[index].get_text()+"){\n")
                    self.pos_x += 1
                    def dfs2(v,visited):
                        visited[v] = 1
                        for i in self.adj_list[v]:
                            if(visited[i] == 0):
                                dfs2(i,visited)
                        return visited
                    yes_path = -1
                    no_path = -1
                    for i in self.adj_list[index]:
                        if(self.nodes[i].get_text().lower() == "si" or self.nodes[i].get_text().lower() == "yes"):
                            yes_path = self.adj_list[index].index(i)
                    for i in self.adj_list[index]:
                        if(self.nodes[i].get_text() == "no"):
                            no_path = self.adj_list[index].index(i)
                    yes_visited = dfs2(self.adj_list[index][yes_path],[0]*len(self.nodes))
                    no_visited = dfs2(self.adj_list[index][no_path],[0]*len(self.nodes))
                    stop = -1
                    for i in range(len(yes_visited)):
                        if(yes_visited[i] == 1 and no_visited[i] == 1):
                            stop = i
                            break

                    self.generate(self.adj_list[index][yes_path],stop)
                    self.pos_x -= 1
                    self.lines_to_write.append(self.__generate_tabs(self.pos_x)+"}\n")
                    self.lines_to_write.append(self.__generate_tabs(self.pos_x)+"else{\n")
                    self.pos_x += 1
                    self.generate(self.adj_list[index][no_path],stop)
                    self.pos_x -= 1
                    self.lines_to_write.append(self.__generate_tabs(self.pos_x)+"}\n")
                    self.generate(stop,-1)
                elif(ans[0] == True):
                    self.lines_to_write.append(self.__generate_tabs(self.pos_x)+"while("+self.nodes[index].get_text()+"){\n")
                    self.pos_x += 1
                    end = index
                    print("endd",end)
                    start = -1
                    for i in self.adj_list[index]:
                        if(self.nodes[i].get_text().lower() == "si" or self.nodes[i].get_text().lower() == "yes"):
                            start = self.adj_list[index].index(i)
                    self.generate(self.adj_list[index][start],end)
                    self.pos_x -= 1
                    self.lines_to_write.append(self.__generate_tabs(self.pos_x)+"}\n")
                    for i in self.adj_list[index]:
                        if(self.nodes[i].get_text().lower() == "no"):
                            start = self.adj_list[index].index(i)
                    self.generate(self.adj_list[index][start],-1)
