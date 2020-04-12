from graphviz import Digraph

dot = Digraph(filename="fibo.dot")

dot.node('1', label='Inicio', shape='ellipse')
dot.node('2', label='ans, n, a=0, b=1, cont=2', shape='rectangle')
dot.edge('1', '2')
dot.node('3', label='n', shape='parallelogram')
dot.edge('2', '3')

with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('4', label='cont < n', shape='diamond')
    dot.edge('3', '4')
    s.node('5', label='ans', shape='invhouse')
    dot.edge('4', '5', label='No')

dot.node('6', label='Fin', shape='ellipse')
dot.edge('5', '6')

dot.node('7', label='ans=a+b', shape='rectangle')
dot.edge('4', '7', label='Yes')
dot.node('8', label='a=b', shape='rectangle')
dot.edge('7', '8')
dot.node('9', label='b=ans', shape='rectangle')
dot.edge('8', '9')
dot.node('10', label='cont=cont+1', shape='rectangle')
dot.edge('9', '10')

dot.edge('10', '4')

dot.format = 'png'
dot.render(view='true')
