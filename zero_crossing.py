import numpy as np
import scipy.ndimage as im
import scipy.misc as misc
import matplotlib.pyplot as plot

def Gaussian_laplacian(x, y, sigma):
    G = (-1*(x*x + y*y - 2*sigma^2)/sigma^4) * np.exp(-1*(x^2 + y^2)/2*sigma^2)
    return G*(-1/(np.pi * sigma^2))

Lena = im.imread('lena_grey.png')
deviate = np.std(Lena)
print(deviate)
