def aux_flow():
    dict = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5, 12], 5: [6], 6: [7], 7: [8], 8: [9], 9: [10], 10: [11], 11: [4], 12: [13], 13: []}

    return dict

# Order (x1, x2, y1, y2)
# ------ TEST CASE 'Even numbers to 100' ------
# Create text nodes
t0 = Node(coordinate=[409,691,358,471], text='inicio')
t1 = Node(coordinate=[414,743,676,769], text='cont = 2')
t2 = Node(coordinate=[459,856,1098,1179], text='cont <= 100')
t3 = Node(coordinate=[648,781,1438,1553], text='Sí')
t4 = Node(coordinate=[450,668,1671,1784], text='cont')
t5 = Node(coordinate=[453,1020,2107,2212], text='cont = cont + 2')
t6 = Node(coordinate=[1030,1150,1007,1120], text='No')
t7 = Node(coordinate=[1420,1614,1053,1179], text='fin')
# Create shape nodes
s0 = Node(coordinate=[302,794,317,494], class_shape='start_end')
s1 = Node(coordinate=[514,622,479,653], class_shape='arrow_line_down')
s2 = Node(coordinate=[312,814,638,807], class_shape='process')
s3 = Node(coordinate=[530,655,799,1020], class_shape='arrow_line_down')
s4 = Node(coordinate=[346,930,989,1427], class_shape='decision')
s5 = Node(coordinate=[586,712,1410,1643], class_shape='arrow_line_down')
s6 = Node(coordinate=[381,830,1643,1876], class_shape='print')
s7 = Node(coordinate=[571,668,1815,2082], class_shape='arrow_line_down')
s8 = Node(coordinate=[371,1109,2038,2258], class_shape='process')
s9 = Node(coordinate=[43,443,2141,2233], class_shape='arrow_line_left')
s10 = Node(coordinate=[56,121,1174,2185], class_shape='arrow_line_up')
s11 = Node(coordinate=[55,296,1160,1235], class_shape='arrow_line_right')
s12 = Node(coordinate=[948,1332,1079,1187], class_shape='arrow_line_right')
s13 = Node(coordinate=[1299,1814,966,1241], class_shape='start_end')

filename = 'even_num.dot'
graph = Graph(
    [t0, t1, t2, t3, t4, t5, t6, t7],
    [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13]
)
