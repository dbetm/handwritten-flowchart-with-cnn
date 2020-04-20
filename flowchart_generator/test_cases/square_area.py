# Order (x1, x2, y1, y2)
# ------ TEST CASE 'Square area' ------
# Create text nodes
t0 = Node(coordinate=[689,1012,207,335], text='inicio')
t1 = Node(coordinate=[607,1284,658,787], text='x=0.0, res=0.0')
t2 = Node(coordinate=[764,881,1076,1169], text='x')
t3 = Node(coordinate=[576,1117,1430,1556], text='res=x*x')
t4 = Node(coordinate=[756,953,1928,2023], text='res')
t5 = Node(coordinate=[681,912,2448,2561], text='fin')

# Create shape nodes
s0 = Node(coordinate=[553,1122,174,397], class_shape='start_end')
s1 = Node(coordinate=[776,858,376,630], class_shape='arrow_line_down')
s2 = Node(coordinate=[458,1392,592,846], class_shape='process')
s3 = Node(coordinate=[787,879,838,1038], class_shape='arrow_line_down')
s4 = Node(coordinate=[587,1048,1017,1202], class_shape='scan')
s5 = Node(coordinate=[774,879,1187,1356], class_shape='arrow_line_down')
s6 = Node(coordinate=[471,1212,1346,1641], class_shape='process')
s7 = Node(coordinate=[820,935,1597,1882], class_shape='arrow_line_down')
s8 = Node(coordinate=[610,1202,1853,2166], class_shape='print')
s9 = Node(coordinate=[792,884,2148,2402], class_shape='arrow_line_down')
s10 = Node(coordinate=[517,1128,2379,2661], class_shape='start_end')

filename = 'reversed_string.dot'
graph = Graph(
    [t0, t1, t2, t3, t4, t5],
    [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10]
)
