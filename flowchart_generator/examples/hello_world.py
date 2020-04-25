from graphviz import Digraph

dot = Digraph(
    comment='First flowchart',
    name='Hello world',
    filename='hello_world.dot',
)

dot.node('1', 'Inicio')
dot.node('2', '"Hola mundo"', shape='invhouse')
#dot.node('2', '"Hola mundo"', shapefile='assets/print.svg')
dot.node('3', 'Fin')
dot.node('4', 'Fin 2')

dot.edge('1', '2')
#dot.attr('graph', splines='ortho', nodesep='1')
dot.edge('2', '3') # constraint='false')
dot.edge('2', '4')


dot.format = 'png'
dot.render(view=True)
