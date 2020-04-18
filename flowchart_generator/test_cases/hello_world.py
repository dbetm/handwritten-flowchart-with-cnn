# ------ TEST CASE 'Hello world' ------
# Create text nodes
t0 = Node(coordinate=[420,847,294,513], text='inicio')
t1 = Node(coordinate=[351,997,1133,1344], text='"Hola mundo"')
t2 = Node(coordinate=[505,997,2029,2217], text="fin")
# Create shape nodes
s0 = Node(coordinate=[209,1105,256,560], class_shape='start_end')
s1 = Node(coordinate=[474,778,594,1056], class_shape='arrow_line_down')
s2 = Node(coordinate=[316,1143,1006,1537], class_shape='print')
s3 = Node(coordinate=[574,824,1460,1940], class_shape='arrow_line_down')
s4 = Node(coordinate=[274,1309,1902,2329], class_shape='start_end')

filename = 'hello_world.dot'
graph = Graph([t2, t1, t0], [s4, s0, s1, s2, s3])
