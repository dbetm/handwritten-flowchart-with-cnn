import cv2
def mass_center(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    xs,pix_n = 0,0
    xmin = float('inf')
    xmax = float('-inf')
    w,h = img.shape
    for y in range(w):
        for x in range(h):
            if(img[y,x] == 0):
                xmin = min(x,xmin)
                xmax = max(x,xmax)
                xs += x
                pix_n += 1
    meanX = xs/pix_n
    print(meanX,((xmax - xmin)/2) + xmin)
    if(meanX < ((xmax - xmin)/2) + xmin):
        return "left"
    else:
        return "right"
img = cv2.imread("images/prueba2.png",0)
cv2.imshow(mass_center(img),img)
cv2.imshow(mass_center(cv2.flip(img,1)),cv2.flip(img,1))
cv2.waitKey(0)
cv2.destroyAllWindows()
