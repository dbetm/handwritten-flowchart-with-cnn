def resize_new_data(image,input_size):
    def get_max_min(image):
        h,w = image.shape
        argmin = float("inf")
        argmax = -float("inf")
        for i in range(h):
            for j in range(w):
                if(image[i,j] == 0):
                    argmax = max(i,argmax)
                    argmin = min(i,argmin)
        return argmax,argmin
    def image_resize(image,height = None,inter = cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape
        r = height / float(h)
        dim = (int(w * r),height)
        resized = cv2.resize(image,dim,interpolation = inter)
        return resized
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image,(3,3),0)
    ret,image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    argmax,argmin = get_max_min(image)
    image = image[argmin:argmax]
    h,w = image.shape
    wt,ht = input_size
    if argmax - argmin > input_size[1] // 2:
        image = image_resize(image,height = (input_size[1] // 2))
    h,w = image.shape
    print(image.shape)
    target = np.ones((ht , wt))*255
    target[0:h,0:w] = image
    image = cv2.transpose(target)
    return image
