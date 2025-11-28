import matplotlib.pyplot as plt 
import skimage
import skimage.color

def homography_apply(H, x1, x2):
    L = [x1,x2,1]
    x2 = 0
    y2 = 0
    numx = 0
    numy = 0
    deno = 0
    
    for i in range (0,2):
        numx = numx + H[0][i]*L[i]
        numy = numy + H[1][i]*L[i]
        deno = deno + H[2][i]*L[i]
    x2 = numx/deno
    x1 = numy/deno           
    return (x2,y2)
    