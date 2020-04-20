# Order (x1, x2, y1, y2)
# ------ TEST CASE 'Square root of n' ------
# Create text nodes
t0 = Node(coordinate=[568,927,47,199], text='inicio')
t1 = Node(coordinate=[501,895,491,611], text='n=0.0')
t2 = Node(coordinate=[615,765,897,1002], text='n')
t3 = Node(coordinate=[504,859,1317,1458], text='sqrt(n)')
t4 = Node(coordinate=[504,739,1958,2073], text='fin')

# Create shape nodes
s0 = Node(coordinate=[409,1095,0,264], class_shape='start_end')
s1 = Node(coordinate=[659,762,226,467], class_shape='arrow_line_down')
s2 = Node(coordinate=[430,977,429,641], class_shape='process')
s3 = Node(coordinate=[648,757,620,879], class_shape='arrow_line_down')
s4 = Node(coordinate=[433,936,847,1050], class_shape='scan')
s5 = Node(coordinate=[636,736,1026,1294], class_shape='arrow_line_down')
s6 = Node(coordinate=[439,998,1258,1623], class_shape='print')
s7 = Node(coordinate=[601,721,1582,1914], class_shape='arrow_line_down')
s8 = Node(coordinate=[324,974,1873,2164], class_shape='start_end')

filename = 'sqrt_n.dot'
graph = Graph(
    [t0, t1, t2, t3, t4],
    [s0,s1,s2,s3,s4,s5,s6,s7,s8]
)
