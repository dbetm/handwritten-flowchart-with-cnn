def aux_flow():
    dict = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: [8], 8: [9], 9: [10], 10: [11], 11: [12], 12: [13], 13: [14], 14: [15], 15: [16], 16: []}

    return dict

# ------ TEST CASE 'Sum 2 numbers' ------
# Create text nodes
t0 = Node(coordinate=[569,855,110,242], text='inicio')
t1 = Node(coordinate=[352,1044,482,589], text='a=0, b=0, res=0')
t2 = Node(coordinate=[477,905,810,946], text='"Dame a: "')
t3 = Node(coordinate=[555,691,1267,1374], text='a')
t4 = Node(coordinate=[412,873,1571,1703], text='"Dame b: "')
t5 = Node(coordinate=[494,587,2103,2171], text='b')
t6 = Node(coordinate=[334,791,2424,2532], text='res = a+b')
t7 = Node(coordinate=[1234,1851,2375,2614], text='"Resultado es: " res')
t8 = Node(coordinate=[1298,1491,2928,3049], text='fin')
# Create shape nodes
s0 = Node(coordinate=[448,1012,57,253], class_shape='start_end')
s1 = Node(coordinate=[651,755,221,467], class_shape='arrow_line_down')
s2 = Node(coordinate=[262,1141,378,635], class_shape='process')
s3 = Node(coordinate=[626,730,603,778], class_shape='arrow_line_down')
s4 = Node(coordinate=[427,944,753,1089], class_shape='print')
s5 = Node(coordinate=[580,709,1053,1253], class_shape='arrow_line_down')
s6 = Node(coordinate=[373,869,1221,1407], class_shape='scan')
s7 = Node(coordinate=[569,651,1374,1564], class_shape='arrow_line_down')
s8 = Node(coordinate=[366,969,1510,1882], class_shape='print')
s9 = Node(coordinate=[494,601,1857,2085], class_shape='arrow_line_down')
s10 = Node(coordinate=[334,741,2046,2214], class_shape='scan')
s11 = Node(coordinate=[477,562,2175,2382], class_shape='arrow_line_down')
s12 = Node(coordinate=[248,873,2353,2574], class_shape='process')
s13 = Node(coordinate=[826,1216,2392,2507], class_shape='arrow_line_left')
s14 = Node(coordinate=[1187,1944,2307,2735], class_shape='print')
s15 = Node(coordinate=[1373,1491,2696,2917], class_shape='arrow_line_down')
s16 = Node(coordinate=[1169,1619,2878,3082], class_shape='start_end')

filename = 'sum.dot'
graph = Graph(
    [t0, t1, t2, t3, t4, t5, t6, t7, t8],
    [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16]
)
