import numpy as np
import scipy.ndimage as im
import scipy.misc as misc
import matplotlib.pyplot as plot
import scipy.signal as signal

def LaplacianOfGaussian(x, y, sigma):
    power = -1*((np.power(x,2) + np.power(y,2))/(2* np.power(sigma, 2)))
    G = np.exp(-1*power)
    LaplaceTerm = (np.power(x, 1) + np.power(y, 2) -(2 * np.power(sigma, 2))) /np.power(sigma, 4)
    return LaplaceTerm * G


def generate_mask_gauss(length, width, sigma):
    """generates two coordinate matrices for masks"""
    XCordinate = np.zeros((width, length))
    Ycoordinate = np.zeros((width, length))
    gaussianMask = np.zeros((width, length))

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
    print(XCordinate)
    """apply the gaussian filter"""
    for i in range(width):
        for j in range(width):
            gaussianMask[j, i] = LaplacianOfGaussian(XCordinate[j, i], Ycoordinate[j, i], sigma)
    print(gaussianMask)
    return gaussianMask

def ZeroCrossing(image):
    M = image.shape[0]
    N = image.shape[1]
    temp = np.zeros((M + 2, N + 2))
    temp[1:-1, 1:-1] = image
    img = np.zeros((M, N))
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if temp[i, j] < 0:
                # Checking over 8 neighbor grid for change in sign
                for x, y in (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1):
                    if temp[i + x, j + y] > 0:
                        img[i - 1, j - 1] = 1
    print(img)
    return img

Lena = im.imread('./input/lena_grey.png')
deviate = im.standard_deviation(Lena)
# deviate = 1
Lena_mask_7 = generate_mask_gauss(7, 7, deviate)
Lena_mask_13 = generate_mask_gauss(13, 13, deviate)
Lena_mask_25 = generate_mask_gauss(25, 25, deviate)

Lena_7 = signal.convolve2d(Lena, Lena_mask_7)
Lena_13 = signal.convolve2d(Lena, Lena_mask_13)
Lena_25 = signal.convolve2d(Lena, Lena_mask_25)
Lena_7 = ZeroCrossing(Lena_7)
print(deviate)
img_plot7 = plot.imshow(Lena_7, cmap='gray')
plot.show()
img_plot13 = plot.imshow(Lena_13, cmap='gray')
plot.show()
img_plot25 = plot.imshow(Lena_25, cmap='gray')
plot.show()