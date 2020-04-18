    # ------ TEST CASE 'Factorial' ------
    # Create text nodes
    t0 = Node(coordinate=[272,429,25,93], text='inicio')
    t1 = Node(coordinate=[202,494,202,270], text='n=0, cont=2, res=1')
    t2 = Node(coordinate=[281,396,348,397], text='n')
    t3 = Node(coordinate=[264,417,545,602], text='cont <= n')
    t4 = Node(coordinate=[327,426,675,732], text='Yes')
    t5 = Node(coordinate=[251,484,795,848], text='res = res * cont')
    t6 = Node(coordinate=[232,486,925,973], text='cont = cont + 1')
    t7 = Node(coordinate=[464,539,517,613], text='No')
    t8 = Node(coordinate=[619,722,557,628], text='res')
    t9 = Node(coordinate=[767,877,715,787], text='fin')
    # Create shape nodes
    s0 = Node(coordinate=[217,492,14,122], class_shape='start_end')
    s1 = Node(coordinate=[307,372,92,207], class_shape='arrow_line_down')
    s2 = Node(coordinate=[162,544,175,297], class_shape='process')
    s3 = Node(coordinate=[309,362,275,343], class_shape='arrow_line_down')
    s4 = Node(coordinate=[174,526,325,428], class_shape='scan')
    s5 = Node(coordinate=[299,377,408,528], class_shape='arrow_line_down')
    s6 = Node(coordinate=[229,449,502,693], class_shape='decision')
    s7 = Node(coordinate=[292,391,662,790], class_shape='arrow_line_down')
    s8 = Node(coordinate=[211,514,763,865], class_shape='process')
    s9 = Node(coordinate=[317,372,842,920], class_shape='arrow_line_down')
    s10 = Node(coordinate=[206,524,895,1007], class_shape='process')
    s11 = Node(coordinate=[72,246,940,1010], class_shape='arrow_line_left')
    s12 = Node(coordinate=[37,132,567,1000], class_shape='arrow_line_up')
    s13 = Node(coordinate=[69,254,553,617], class_shape='arrow_line_right')
    s14 = Node(coordinate=[431,594,542,608], class_shape='arrow_line_right')
    s15 = Node(coordinate=[587,799,517,657], class_shape='print')
    s16 = Node(
        coordinate=[624,771,635,773],
        class_shape='arrow_rectangle_right',
        image_path="/home/david/Escritorio/samples_flowcharts/factorial_rect1.jpg"
    )
    s17 = Node(coordinate=[749,949,673,792], class_shape='start_end')

    filename = 'factorial.dot'
    graph = Graph(
        [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9],
        [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17]
    )
