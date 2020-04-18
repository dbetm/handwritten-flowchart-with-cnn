def aux_flow():
    flow = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7, 17], 7: [8], 8: [9], 9: [10], 10: [11], 11: [12], 12: [13], 13: [14], 14: [15], 15: [16], 16: [6], 17: [18], 18: [19], 19: [20], 20: []}

    return flow

# Order (x1, x2, y1, y2)
# ------ TEST CASE 'Fibonacci' ------
# Create text nodes
t0 = Node(coordinate=[934,1219,136,263], text='inicio')
t1 = Node(coordinate=[407,1296,370,486], text='ans,n,a=0,b=1,cont=2')
t2 = Node(coordinate=[730,873,655,759], text='n')
t3 = Node(coordinate=[630,923,951,1090], text='cont < n')
t4 = Node(coordinate=[780,927,1190,1282], text='Yes')
t5 = Node(coordinate=[569,900,1386,1490], text='ans = a+b')
t6 = Node(coordinate=[627,854,1632,1751], text='a = b')
t7 = Node(coordinate=[592,884,1886,1997], text='b = ans')
t8 = Node(coordinate=[542,996,2194,2294], text='cont = cont+1')
t9 = Node(coordinate=[477,607,901,1047], text='no')
t10 = Node(coordinate=[115,354,982,1109], text='ans')
t11 = Node(coordinate=[127,323,601,717], text='fin')
# Create shape nodes
s0 = Node(coordinate=[838,1342,55,305], class_shape='start_end')
s1 = Node(
    coordinate=[519,880,105,363],
    class_shape='arrow_rectangle_down',
    image_path="/home/david/Escritorio/samples_flowcharts/fib_rect0.jpg"
)
s2 = Node(coordinate=[350,1330,320,520], class_shape='process')
s3 = Node(coordinate=[761,892,447,663], class_shape='arrow_line_down')
s4 = Node(coordinate=[573,1015,617,778], class_shape='scan')
s5 = Node(coordinate=[765,873,770,901], class_shape='arrow_line_down')
s6 = Node(coordinate=[619,961,874,1201], class_shape='decision')
s7 = Node(coordinate=[711,884,1147,1386], class_shape='arrow_line_down')
s8 = Node(coordinate=[527,988,1351,1540], class_shape='process')
s9 = Node(coordinate=[696,796,1494,1636], class_shape='arrow_line_down')
s10 = Node(coordinate=[542,954,1590,1782], class_shape='process')
s11 = Node(coordinate=[692,765,1763,188], class_shape='arrow_line_down')
s12 = Node(coordinate=[523,980,1847,2028], class_shape='process')
s13 = Node(coordinate=[650,757,1982,2167], class_shape='arrow_line_down')
s14 = Node(coordinate=[480,1077,2117,2363], class_shape='process')
s15 = Node(
    coordinate=[1027,1407,1582,2294],
    class_shape='arrow_rectangle_up',
    image_path="/home/david/Escritorio/samples_flowcharts/fib_rect1.jpg"
)
s16 = Node(
    coordinate=[919,1438,928,1636],
    class_shape='arrow_rectangle_left',
    image_path="/home/david/Escritorio/samples_flowcharts/fib_rect2.jpg"
)
s17 = Node(coordinate=[384,661,947,1094], class_shape='arrow_line_left')
s18 = Node(coordinate=[34,407,951,1244], class_shape='print')
s19 = Node(coordinate=[161,304,717,1005], class_shape='arrow_line_up')
s20 = Node(coordinate=[77,496,570,767], class_shape='start_end')

filename = 'fibonacci.dot'
graph = Graph(
    [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11],
    [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20]
)
