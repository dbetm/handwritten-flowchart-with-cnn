# Order (x1, x2, y1, y2)
# ------ TEST CASE 'Count digits' ------
# Create text nodes
t0 = Node(coordinate=[704,951,211,325], text='inicio')
t1 = Node(coordinate=[551,1140,555,683], text='n=0, cont=0')
t2 = Node(coordinate=[760,885,938,1030], text='n')
t3 = Node(coordinate=[615,879,1358,1455], text='n > 0')
t4 = Node(coordinate=[743,849,1608,1691], text='Sí')
t5 = Node(coordinate=[451,887,1850,1950], text='n = n / 10')
t6 = Node(coordinate=[499,1040,2269,2347], text='cont = cont + 1')
t7 = Node(coordinate=[999,1121,1286,1394], text='No')
t8 = Node(coordinate=[1318,1546,1363,1480], text='cont')
t9 = Node(coordinate=[1304,1507,1877,1986], text='fin')
# Create shape nodes
s0 = Node(coordinate=[585,1185,172,350], class_shape='start_end')
s1 = Node(coordinate=[812,907,336,544], class_shape='arrow_line_down')
s2 = Node(coordinate=[496,1240,502,744], class_shape='process')
s3 = Node(coordinate=[810,885,708,891], class_shape='arrow_line_down')
s4 = Node(coordinate=[515,1060,875,1111], class_shape='scan')
s5 = Node(coordinate=[704,812,1072,1294], class_shape='arrow_line_down')
s6 = Node(coordinate=[537,949,1272,1600], class_shape='decision')
s7 = Node(coordinate=[685,771,1575,1819], class_shape='arrow_line_down')
s8 = Node(coordinate=[401,987,1794,2011], class_shape='process')
s9 = Node(coordinate=[662,746,1972,2219], class_shape='arrow_line_down')
s10 = Node(coordinate=[418,1112,2194,2413], class_shape='process')
s11 = Node(coordinate=[110,451,2244,2350], class_shape='arrow_line_left')
s12 = Node(coordinate=[96,179,1397,2308], class_shape='arrow_line_up')
s13 = Node(coordinate=[121,557,1361,1480], class_shape='arrow_line_right')
s14 = Node(coordinate=[929,1293,1352,1469], class_shape='arrow_line_right')
s15 = Node(coordinate=[1271,1699,1302,1591], class_shape='print')
s16 = Node(coordinate=[1371,1479,1575,1833], class_shape='arrow_line_down')
s17 = Node(coordinate=[1185,1679,1791,2038], class_shape='start_end')

filename = 'count_digts.dot'
graph = Graph(
    [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9],
    [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17]
)
