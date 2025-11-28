import numpy as np
import skimage

def Homography_estimate(x1, y1, x2, y2):
    n1 = len(x1)
    if n1 < 4:
        raise ValueError("Il faut au moins 4 points.")

    A = []
    B = []
    
    for k in range(n1):
        sx, sy = x1[k], y1[k]
        dx, dy = x2[k], y2[k]
        
        # correspond à h11*sx + h12*sy + h13 - h31*sx*dx - h32*sy*dx = dx
        ligne_1 = [sx, sy, 1, 0, 0, 0, -sx*dx, -sy*dx]
        A.append(ligne_1)
        B.append(dx)
        
        # correspond à h21*sx + h22*sy + h23 - h31*sx*dy - h32*sy*dy = dy
        ligne_2 = [0, 0, 0, sx, sy, 1, -sx*dy, -sy*dy]
        A.append(ligne_2)
        B.append(dy)

    A = np.array(A)
    B = np.array(B)

    # Résolution
    X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    # Reconstruction de la matrice 3x3
    H_flat = np.append(X, 1)
    H = H_flat.reshape((3, 3))
    
    return H

def homography_apply(H, x1, y1):
    x1=np.array(x1)
    y1=np.array(y1)
    numx = 0
    numy = 0
    deno = 0
    
    # numx = h11*x + h12*y + h13
    numx = H[0,0]*x1 + H[0,1]*y1 + H[0,2]
    
    # numy = h21*x + h22*y + h23
    numy = H[1,0]*x1 + H[1,1]*y1 + H[1,2]
    
    # deno = h31*x + h32*y + h33
    deno = H[2,0]*x1 + H[2,1]*y1 + H[2,2]
    
    x2 = numx/deno
    x1 = numy/deno           
    return (x2,y2)
    

#(Source: Carré, Destination: Carré décalé de 5,5)
x1, x2, y1, y2 = [0, 10, 10, 0], [5, 15, 15, 5], [0, 0, 10, 10], [5, 5, 15, 15]

H = Homography_estimate(x1, y1, x2, y2)

x2_calc, y2_calc = homography_apply(H, x1, y1)

print("x2 calculés:", x2_calc)
print("x2 théoriques:", x2)
print("y2 calculés:",y2_calc)
print("y2 théorique:",y2)