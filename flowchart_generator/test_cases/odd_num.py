def aux_flow():
    dict = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5,11], 5: [6], 6: [7], 7: [8], 8: [9], 9: [10], 10: [4], 11: [12], 12: []}

    return dict


# Order (x1, x2, y1, y2)
# ------ TEST CASE 'Odd numbers to 100' ------
# Create text nodes
t0 = Node(coordinate=[879,1154,125,236], text='inicio')
t1 = Node(coordinate=[879,1132,463,569], text='c = 1')
t2 = Node(coordinate=[862,1135,872,980], text='c <= 100')
t3 = Node(coordinate=[1004,1137,1136,1241], text='Yes')
t4 = Node(coordinate=[862,990,1350,1444], text='c')
t5 = Node(coordinate=[749,1112,1766,1863], text='c = c + 2')
t6 = Node(coordinate=[635,754,811,902], text='No')
t7 = Node(coordinate=[199,390,886,1002], text='fin')
# Create shape nodes
s0 = Node(coordinate=[743,1299,83,1299], class_shape='start_end')
s1 = Node(coordinate=[946,1062,244,450], class_shape='arrow_line_down')
s2 = Node(coordinate=[782,1235,433,625], class_shape='process')
s3 = Node(coordinate=[965,1043,594,805], class_shape='arrow_line_down')
s4 = Node(coordinate=[815,1179,777,1119], class_shape='decision')
s5 = Node(coordinate=[943,1007,1097,1322], class_shape='arrow_line_down')
s6 = Node(coordinate=[796,1182,1302,1550], class_shape='print')
s7 = Node(coordinate=[901,965,1533,1708], class_shape='arrow_line_down')
s8 = Node(coordinate=[674,1235,1697,1947], class_shape='process')
s9 = Node(
    coordinate=[1190,1640,1402,1858],
    class_shape='arrow_rectangle_up',
    image_path="/home/david/Escritorio/samples_flowcharts/r/odd_rect1.jpg"
)
s10 = Node(
    coordinate=[1174,1657,852,1402],
    class_shape='arrow_rectangle_left',
    image_path="/home/david/Escritorio/samples_flowcharts/r/odd_rect2.jpg"
)
s11 = Node(coordinate=[490,851,880,983], class_shape='arrow_line_left')
s12 = Node(coordinate=[71,518,819,1058], class_shape='start_end')

filename = 'odd_num.dot'
graph = Graph(
    [t0, t1, t2, t3, t4, t5, t6, t7],
    [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12]
)
