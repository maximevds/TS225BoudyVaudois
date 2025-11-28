import numpy as np
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