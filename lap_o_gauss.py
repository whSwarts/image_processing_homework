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

    def ZeroCrossing(image):
        (M, N) = image.shape
        # detect zero crossing by checking values across 8-neighbors on a 3x3 grid
        temp = np.zeros((M + 2, N + 2))
        temp[1:-1, 1:-1] = image
        img = np.zeros((M, N))
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if temp[i, j] < 0:
                    # Checking over 8 neighbor grid for change in polarity of the gradient
                    for x, y in (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1):
                        if temp[i + x, j + y] > 0:
                            img[i - 1, j - 1] = 1

Lena = im.imread('./input/lena_grey.png')
deviate = np.std(Lena)
Lena_mask_7 = generate_mask_gauss(7,7,deviate)
Lena_mask_13 = generate_mask_gauss(13,13,deviate)
Lena_mask_25 = generate_mask_gauss(25,25,deviate)

Lena_7 = signal.convolve2d(Lena, Lena_mask_7)
img_plot7 = plot.imshow(Lena_7, cmap='gray')
plot.show()

