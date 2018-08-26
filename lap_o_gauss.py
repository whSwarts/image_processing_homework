import numpy as np
import scipy.ndimage as im
import scipy.misc as misc
import matplotlib.pyplot as plot
import scipy.signal as signal

def LaplacianOfGaussian(x, y, sigma):
    power = ((np.power(x,2) + np.power(y,2))/(2* np.power(sigma, 2)))
    G = np.exp(-1*power)
    LaplaceTerm = -1 * (1/(np.pi * np.power(sigma, 4))) * (1-power)
    return LaplaceTerm * G 


def generate_mask_gauss(length, width, sigma):
    """generates two coordinate matrices for masks"""
    XCordinate = np.zeros((width, length), dtype=int)
    Ycoordinate = np.zeros((width, length), dtype= int)
    gaussianMask = np.zeros((width, length), dtype= int)

    center = int((width-1)/2)
    # print(center)

    for i in range(width):
        for j in range(length):
            if( j == center and i == center):
                Ycoordinate[center, center] = 0
            else:
                Ycoordinate[i, j] = center - j
                XCordinate[i, j] = i - center

    XCordinate = np.transpose(XCordinate)
    Ycoordinate = np.transpose(Ycoordinate)

    """apply the gaussian filter"""
    for i in range(width):
        for j in range(width):
            gaussianMask[j, i] = LaplacianOfGaussian(XCordinate[j, i], Ycoordinate[j, i], sigma)

    return gaussianMask


Lena = im.imread('./input/lena_grey.png')
deviate = np.std(Lena)
