def aux_flow():
    flow = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7, 16], 7: [8], 8: [9, 13], 9: [10], 10: [11], 11: [12], 12: [20], 13: [14], 14: [15], 15: [20], 16: [17], 17: [18], 18: [19], 19: [20], 20: []}

    return flow


# Order (x1, x2, y1, y2)
# ------ TEST CASE 'Conditions1' ------
# Create text nodes
t0 = Node(coordinate=[333,622,165,292], text='inicio')
t1 = Node(coordinate=[337,568,500,611], text='n = 0')
t2 = Node(coordinate=[357,476,811,903], text='n')
t3 = Node(coordinate=[218,483,1188,1311], text='n < 10')
t4 = Node(coordinate=[383,503,1461,1565], text='Yes')
t5 = Node(coordinate=[114,460,1788,1926], text='n%2 = 0')
t6 = Node(coordinate=[310,457,2023,2146], text='Yes')
t7 = Node(coordinate=[118,449,2257,2415], text='"Bien"')
t8 = Node(coordinate=[649,810,1711,1842], text='No')
t9 = Node(coordinate=[1080,1372,1788,1973], text='Error')
t10 = Node(coordinate=[587,745,1119,1257], text='No')
t11 = Node(coordinate=[864,1160,1169,1334], text='Error')
t12 = Node(coordinate=[1672,1845,1811,1923], text='Fin')

# Create shape nodes
s0 = Node(coordinate=[214,793,108,329], class_shape='start_end')
s1 = Node(coordinate=[393,531,300,475],class_shape='arrow_line_down')
s2 = Node(coordinate=[252,677,441,650], class_shape='process')
s3 = Node(coordinate=[377,493,616,787], class_shape='arrow_line_down')
s4 = Node(coordinate=[164,660,754,946], class_shape='scan')
s5 = Node(coordinate=[343,464,908,1091], class_shape='arrow_line_down')
s6 = Node(coordinate=[172,552,1066,1462], class_shape='decision')
s7 = Node(coordinate=[302,439,1446,1671], class_shape='arrow_line_down')
s8 = Node(coordinate=[60,539,1654,2058], class_shape='decision')
s9 = Node(coordinate=[222,364,2046,2237], class_shape='arrow_line_down')
s10 = Node(coordinate=[60,552,2216,2521], class_shape='print')
s11 = Node(coordinate=[452,1872,2225,2421],class_shape='arrow_line_right')
s12 = Node(coordinate=[1664,1860,1875,2383],class_shape='arrow_line_up')
s13 = Node(coordinate=[506,1085,1708,1933], class_shape='arrow_line_right')
s14 = Node(coordinate=[1060,1452,1729,2033], class_shape='print')
s15 = Node(coordinate=[1410,1643,1775,1925], class_shape='arrow_line_right')
s16 = Node(coordinate=[514,839,1113,1291],class_shape='arrow_line_right')
s17 = Node(coordinate=[806,1281,1083,1471],class_shape='print')
s18 = Node(coordinate=[1247,1727,1183,1300], class_shape='arrow_line_right')
s19 = Node(coordinate=[1618,1810,1212,1804], class_shape='arrow_line_down')
s20 = Node(coordinate=[1606,1915,1758,1956], class_shape='start_end')

filename = 'conditions1.dot'
graph = Graph(
    [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12],
    [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20]
)
