from graphviz import Digraph

dot = Digraph(filename='factorial.dot')
dot.attr('graph', splines='polyline')

# add nodes
dot.node('1', label='Inicio', shape='ellipse')
dot.node('2', label='n, cont=2, res=1', shape='rectangle')
dot.node('3', label='n', shape='parallelogram')

# Create a subgraph
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('4', label='cont <= n', shape='diamond')
    s.node('7', label='res', shape='invhouse')

dot.node('5', label='res = res * cont', shape='rectangle')
dot.node('6', label='cont = cont + 1', shape='rectangle')
dot.node('8', label='Fin', shape='ellipse')
# Add edges
dot.edge('1', '2')
dot.edge('2', '3')
dot.edge('3', '4')
dot.edge('4', '5', label='Yes')
dot.edge('5', '6')
dot.edge('6', '4')
# dot.attr('graph', splines='line', nodesep='1')
dot.edge('4', '7', label='No')
dot.edge('7', '8')

# Display and save
dot.format = 'png'
dot.render(view=True)
