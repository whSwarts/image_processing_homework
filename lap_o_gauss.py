import numpy as np
import scipy.ndimage as im
import scipy.misc as misc
import matplotlib.pyplot as plot

def Gaussian_laplacian(x, y, sigma):
    G = (-1*(x*x + y*y - 2*sigma^2)/sigma^4) * np.exp(-1*(x^2 + y^2)/2*sigma^2)
    return G*(-1/(np.pi * sigma^2))


def generate_mask(length, width):
    """""/* create empty arrays for coordinates *\""""
    empty_X = np.empty((length, width), dtype=float)
    empty_y = np.empty((length, width), dtype=float)

    """populate the arrays with coordinates where (width, lenght)/2 the origin"""
    oriX = int((width/2)-1)
    oriY = int((length/2)-1)
    empty_X[oriX, oriY] = 0
    empty_y[oriX, oriY] = 0


generate_mask(8, 8)

Lena = im.imread('lena_grey.png')
deviate = np.std(Lena)
print(deviate)
