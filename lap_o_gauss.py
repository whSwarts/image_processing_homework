import numpy as np
import scipy.ndimage as im
import scipy.misc as misc
import matplotlib.pyplot as plot
import scipy.signal as signal


"""Ek het nie tyd gehad om op te spoor hoekom my Gaussian nie werk nie, so ek het maar n library call gebruik"""
np.set_printoptions(precision=1)
def LaplacianOfGaussian(x, y, sigma):
    power = -1*((np.power(x,2) + np.power(y,2))/(2* np.power(sigma, 2)))
    G = np.exp(power)
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
    # print(XCordinate)
    """apply the gaussian filter"""
    for i in range(0,width-1,1):
        for j in range(0, width-1, 1):
            gaussianMask[i,j] = LaplacianOfGaussian(XCordinate[i, j], Ycoordinate[i,j], sigma)
    print(gaussianMask)
    return gaussianMask

def ZeroCrossing(image, sigma):
    w = image.shape[0]
    h = image.shape[1]
    img = np.zeros(image.shape)
    thres = 2* np.sqrt(2)*sigma
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = image[y - 1:y + 2, x - 1:x + 2]
            p = image[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thres) and zeroCross:
                img[y, x] = 1
    # print(img)
    return img

Lena = im.imread('./input/lena_gray_256.tif')
# deviate = im.standard_deviation(Lena)
# deviate = 1
Lena_mask_7 = generate_mask_gauss(7, 7, 1)
Lena_mask_13 = generate_mask_gauss(13, 13, 2)
Lena_mask_25 = generate_mask_gauss(25, 25, 4)

# Lena_7 = signal.convolve2d(Lena, Lena_mask_7)
Lena_13 = im.gaussian_laplace(Lena, 2)
Lena_7 = im.gaussian_laplace(Lena, 1)
Lena_25 = signal.convolve2d(Lena, Lena_mask_25)
Lena_7_zero = ZeroCrossing(Lena_7, 1)
Lena_13_zero = ZeroCrossing(Lena_13, 2)
Lena_25_zero = ZeroCrossing(Lena, 4)
print(Lena_7_zero)
img_plot7 = plot.imshow(Lena_7, cmap='gray')
plot.title("Masker van 7x7 -- Lena")
plot.savefig("./output_hw2/LenaMask7.png")
plot.show()
img_plot13 = plot.imshow(Lena_13, cmap='gray')
plot.title("Masker van 13x13 -- Lena")
plot.savefig("./output_hw2/LenaMask13.png")
plot.show()
"""Zero crossings"""
img_plot7_0 = plot.imshow(Lena_7_zero, cmap='gray')
plot.title("Zero crossing van 7x7 -- Lena")
plot.savefig("./output_hw2/LenaZero7.png")
plot.show()
img_plot25 = plot.imshow(Lena_25, cmap='gray')
plot.title("Masker van 25*25 -- Lena")
plot.savefig("./output_hw2/LenaMask25.png")
plot.show()
img_plot25 = plot.imshow(Lena_13_zero, cmap='gray')
plot.title("Zero crossing van 13*13 -- Lena")
plot.savefig("./output_hw2/LenaZero13.png")
plot.show()
img_plot25 = plot.imshow(Lena_25_zero, cmap='gray')
plot.title("Zero crossing van 25*25 -- Lena")
plot.savefig("./output_hw2/LenaZero7.png")
plot.show()