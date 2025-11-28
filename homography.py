import numpy as np
import matplotlib.pyplot as plt


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
    x1 = np.array(x1)
    y1 = np.array(y1)
    
    # numx = h11*x + h12*y + h13
    numx = H[0,0]*x1 + H[0,1]*y1 + H[0,2]
    
    # numy = h21*x + h22*y + h23
    numy = H[1,0]*x1 + H[1,1]*y1 + H[1,2]
    
    # deno = h31*x + h32*y + h33
    deno = H[2,0]*x1 + H[2,1]*y1 + H[2,2]
    
    # Ajout d'une petite sécurité pour éviter la division par 0
    deno[deno == 0] = 1e-10
    
    x2 = numx / deno
    y2 = numy / deno
    
    return x2, y2


def homography_projection(I1, I2, x, y):
    """
    Projette l'image rectangulaire I1 dans la zone quadrangulaire de I2 
    en parcourant TOUTE l'image I2.
    """
    # Définition des coins de l'image Source (I1)
    h_src, w_src = I1.shape[0], I1.shape[1]
    
    # Coordonnées théoriques du rectangle source
    # Ordre: Haut-Gauche, Haut-Droit, Bas-Droit, Bas-Gauche
    x_src = [0, w_src, w_src, 0]
    y_src = [0, 0, h_src, h_src]

    # Estimation de l'homographie "Inverse" (Destination -> Source)
    H_inv = Homography_estimate(x, y, x_src, y_src)
    
    I_result = I2.copy()
    
    # Dimensions de l'image destination
    h_dest, w_dest = I2.shape[0], I2.shape[1]

    # 4. Balayage de TOUTE l'image I2
    # On utilise h_dest et w_dest, pas les vecteurs x et y
    for v_dest in range(h_dest):
        for u_dest in range(w_dest):
            
            # Application de l'homographie inverse pour un seul pixel
            # On passe des listes [u_dest] car homography_apply attend des tableaux
            u_src_arr, v_src_arr = homography_apply(H_inv, [u_dest], [v_dest])
            u_src, v_src = u_src_arr[0], v_src_arr[0]
        
            # On vérifie si le point projeté tombe bien dans le rectangle de I1
            if 0 <= u_src < w_src and 0 <= v_src < h_src:
                
                # Interpolation "Plus proche voisin"
                # On copie le pixel de I1 vers I2
                I_result[v_dest, u_dest] = I1[int(v_src), int(u_src)]

    return I_result

I1_full = plt.imread('./images/mon_ecran.png')
I2_full = plt.imread('./images/ecran_P.jpg')


I2_rot = np.rot90(I2_full, k=1, axes=(1,0))

# Si I1 est en float (entre 0 et 1), on le passe en 0-255
if I1_full.dtype == np.float32 or I1_full.dtype == np.float64:
    if I1_full.max() <= 1.0:
        print("Conversion de I1 (float 0-1) vers uint8 (0-255)")
        I1_full = (I1_full * 255).astype(np.uint8)

# Si I2 est en float (ça arrive parfois selon les versions), on le passe en 0-255 aussi
if I2_full.dtype == np.float32 or I2_full.dtype == np.float64:
    if I2_full.max() <= 1.0:
        print("Conversion de I2 (float 0-1) vers uint8 (0-255)")
        I2_full = (I2_full * 255).astype(np.uint8)

#Sous-échantillonnage car image trop lourde
facteur = 10

#[::step] -> prend 1 pixel tous les 'facteur' pixels
I2_small = I2_rot[::facteur, ::facteur] 

# Pour I1, on réduit aussi pour que ce soit cohérent
I1_small = I1_full[::(facteur//3), ::(facteur//3)] 

#Si I1 est un PNG (4 canaux RGBA) et I2 un JPG (3 canaux RGB),
# on supprime le canal Alpha de I1 pour éviter les erreurs dans la fonction.
if I1_small.shape[2] == 4:
    I1_small = I1_small[:, :, :3]
if I2_small.shape[2] == 4:
    I2_small = I2_small[:, :, :3]

# 4. Adaptation des coordonnées
# Si l'image est 10x plus petite, les coordonnées doivent être divisées par 10
x_orig = [220, 1700, 1870, 463]
y_orig = [1527, 1432, 2267, 2715]

x_small = [val / facteur for val in x_orig]
y_small = [val / facteur for val in y_orig]

print(f"Taille originale I2 : {I2_rot.shape}")
print(f"Nouvelle taille I2  : {I2_small.shape}")
print("Calcul en cours...")

Iresult = homography_projection(I1_small, I2_small, x_small, y_small)
# --- AFFICHAGE ---
plt.figure()
plt.imshow(I1_small)
plt.figure()
plt.imshow(Iresult)
plt.title(f"Projection (Image réduite par {facteur})")
plt.axis('off')
plt.show()