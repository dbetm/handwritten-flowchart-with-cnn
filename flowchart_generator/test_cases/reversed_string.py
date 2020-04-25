def aux_flow():
    dict = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: [8], 8: [9], 9: [10], 10: [11], 11: [12], 12: [13, 20], 13: [14], 14: [15], 15: [16], 16: [17], 17: [18], 18: [19], 19: [12], 20: [21], 21: []}


    return dict

# Order (x1, x2, y1, y2)
# ------ TEST CASE 'Reversed string' ------
# Create text nodes
t0 = Node(coordinate=[740,947,95,182], text='inicio')
t1 = Node(coordinate=[735,1077,370,435], text='n=0, i=0')
t2 = Node(coordinate=[737,840,615,687], text='n')
t3 = Node(coordinate=[605,1012,907,1007], text='str[n] = ""')
t4 = Node(coordinate=[665,825,1267,1350], text='str')
t5 = Node(coordinate=[587,887,1527,1615], text='i = n-1')
t6 = Node(coordinate=[630,875,1792,1902], text='i >= 0')
t7 = Node(coordinate=[777,877,1977,2052], text='Sí')
t8 = Node(coordinate=[625,855,2162,2262], text='str[i]')
t9 = Node(coordinate=[587,842,2572,2650], text='i=i-1')
t10 = Node(coordinate=[307,422,1750,1822], text='No')
t11 = Node(coordinate=[92,287,2117,2232], text='fin')

# Create shape nodes
s0 = Node(coordinate=[635,1032,47,217], class_shape='start_end')
s1 = Node(coordinate=[777,855,187,347], class_shape='arrow_line_down')
s2 = Node(coordinate=[657,1125,312,462], class_shape='process')
s3 = Node(coordinate=[780,837,445,600], class_shape='arrow_line_down')
s4 = Node(coordinate=[595,1025,577,730], class_shape='scan')
s5 = Node(coordinate=[752,847,697,907], class_shape='arrow_line_down')
s6 = Node(coordinate=[562,1102,872,1075], class_shape='process')
s7 = Node(coordinate=[730,812,1040,1250], class_shape='arrow_line_down')
s8 = Node(coordinate=[510,975,1230,1375], class_shape='scan')
s9 = Node(coordinate=[715,792,1352,1512], class_shape='arrow_line_down')
s10 = Node(coordinate=[507,972,1500,1647], class_shape='process')
s11 = Node(coordinate=[710,775,1632,1760], class_shape='arrow_line_down')
s12 = Node(coordinate=[567,932,1747,2005], class_shape='decision')
s13 = Node(coordinate=[727,792,1992,2135], class_shape='arrow_line_down')
s14 = Node(coordinate=[547,972,2120,2402], class_shape='print')
s15 = Node(coordinate=[692,777,2370,2557], class_shape='arrow_line_down')
s16 = Node(coordinate=[545,907,2540,2707], class_shape='process')
s17 = Node(
    coordinate=[870,1372,2242,2625],
    class_shape='arrow_rectangle_up',
    image_path="/home/david/Escritorio/samples_flowcharts/r/rev_str_r1.jpg"
)
s18 = Node(coordinate=[1250,1342,1835,2242], class_shape='arrow_line_up')
s19 = Node(coordinate=[915,1300,1790,1887], class_shape='arrow_line_left')
s20 = Node(
    coordinate=[151,580,1718,2065],
    class_shape='arrow_rectangle_down',
    image_path="/home/david/Escritorio/samples_flowcharts/r/rev_str_r2.jpg"
)
s21 = Node(coordinate=[0,432,2060,2265], class_shape='start_end')

filename = 'reversed_string.dot'
graph = Graph(
    [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11],
    [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21]
)
