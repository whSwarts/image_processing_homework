"""author: Willem Swarts 24223468"""
import numpy as np
import scipy.ndimage as im
import scipy.misc as misc
import matplotlib.pyplot as plot
import scipy.signal as signal

np.set_printoptions(precision=1)
def LaplacianOfGaussian(x, y, sigma):
    """Calculates the laplacian of the gaussian via a single equation"""
    LoG = ((np.power(x, 2) + np.power(y, 2) - (2 * np.power(sigma, 2))) / np.power(sigma, 4))\
        * np.exp(-1 * ((np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2))))
    return LoG

def Gaussian(x, y, sigma):
    """Calculates the gaussian filter"""
    guass = np.exp(-1 * ((np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2))))
    return guass

def Laplacian():
    """Generates a 3x3 Gaussian mask and returns the mask"""
    laplacian_mask = np.zeros((3, 3))
    counter = 0
    for i in range(0, laplacian_mask.shape[0]):
        for j in range(0, laplacian_mask.shape[1]):
            if (i == 1 and j == 1):
                laplacian_mask[i, j] = -8
                counter += 8
            else:
                laplacian_mask[i, j] = 1
                counter += 1

    return laplacian_mask

def GaussFirst(imageName, filterSize, sigma, imageTitle):
    """Calculates the LoG by first applying the Gaussian and then the laplacian:
    :parameter imageName must include the file extension"""

    imageDirectory = "./input/" + imageName
    chosenImage = im.imread(imageDirectory)

    #     Calls Gaussian to generate the filter to be applied
    mask = np.zeros((filterSize, filterSize))
    halfway = int((filterSize -1)/2)
    for x in range(-halfway, halfway):
        for y in range(-halfway, halfway):
            mask[x + halfway+1, y + halfway + 1] = Gaussian(x, y , sigma)

    #     applies the filter to the image
    image_G = signal.convolve2d(chosenImage, mask)
    image_LoG = signal.convolve2d(image_G, Laplacian())

    plot.imshow(image_LoG, cmap='gray')
    titleOfPlot = str(imageName) + "LoG(G first) mask with dimension of " + str(filterSize) + " and sigma =" + str(sigma)
    plot.title(titleOfPlot)
    plot.show()
    figureOutPathFilter = "./output_hw2/gaussianFirst/" + imageTitle + ".png"
    misc.imsave(figureOutPathFilter, image_LoG)

    #     apply the zerocrossing to obtain the image's edges
    imageZeroX = ZeroCrossing(image_LoG, sigma)
    plot.imshow(imageZeroX, cmap='gray')
    # X in filename indicates zero crossing
    titleOfPlotX = imageName + "LoG (G_first) and zero crossing mask with dimension of " + str(filterSize) + " and sigma=" + str(
        sigma)
    plot.title(titleOfPlot)
    plot.show()
    figureOutPathFilterX = "./output_hw2/gaussianFirst/" + imageTitle + "_zerocross.png"
    misc.imsave(name=figureOutPathFilterX, arr=imageZeroX)



def generateMaskGaussian(mask_dimension, sigma):
    """applies the LoG formula to a mask with:
    :parameter x, y: the x and y coordinate for the mask
    :parameter sigma: the standard deviation for the image, depending on the mask size"""
    mask = np.zeros((mask_dimension, mask_dimension))
    for i in range(mask_dimension):
        for j in range(mask_dimension):
            x = i - int((mask_dimension - 1) / 2)
            y = j - int((mask_dimension - 1) / 2)
            mask[i][j] = LaplacianOfGaussian(x, y, sigma)
    return mask

def ZeroCrossing(image, sigma):
    """Calculates the zerocrossing of the image by checking the neighbors' signs
        if it is different and greater than the absolute value of the differences, then it is 1, else it is 0"""
    w = image.shape[0]
    h = image.shape[1]
    zeroX = np.zeros(image.shape)
    thres = 0.3 * np.max(image)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            neighbors = image[i - 1:i + 2, j - 1:j + 2]
            p = image[i, j]
            neighborOne = neighbors.max()
            neighborTwo = neighbors.min()
            if (p > 0):
                zeroCross = 1 if neighborTwo < 0 else 0
            else:
                zeroCross = 1 if neighborOne > 0 else 0
            if (np.abs(neighborOne - neighborTwo) > thres) and zeroCross:
                zeroX[i, j] = 1

    return zeroX


def filterImage(imageName, filterSize, sigma, imageTitle):
    """Opens the chosen image, applies the LoG filter and saves the result in output"""
    """:parameter imageName must include file extension"""

    imageDirectory = "./input/" + imageName
    chosenImage = im.imread(imageDirectory)

#     Calls generateMaskGaussian to generate the filter to be applied
    LoG_mask = generateMaskGaussian(filterSize, sigma)

#     applies the filter to the image
    image_LoG = signal.convolve2d(chosenImage, LoG_mask)

    plot.imshow(image_LoG, cmap = 'gray')
    titleOfPlot = str(imageName) + "LoG mask with dimension of " + str(filterSize) + " and sigma =" + str(sigma)
    plot.title(titleOfPlot)
    plot.show()
    figureOutPathFilter = "./output_hw2/" + imageTitle + ".png"
    misc.imsave(figureOutPathFilter, image_LoG)


#     apply the zerocrossing to obtain the image's edges
    imageZeroX = ZeroCrossing(image_LoG, sigma)
    plot.imshow(imageZeroX, cmap='gray')
    # X in filename indicates zero crossing
    titleOfPlotX = imageName + "LoG and zero crossing mask with dimension of " + str(filterSize) + " and sigma=" + str(sigma)
    plot.title(titleOfPlot)
    plot.show()
    figureOutPathFilterX = "./output_hw2/" + imageTitle + "_zerocross.png"
    misc.imsave(name=figureOutPathFilterX, arr=imageZeroX)

#     plots the histogram



# # calculate the LoG for Lena
filterImage("lena_gray_256.tif", 7, 1,"lena7")
filterImage("lena_gray_256.tif", 13, 2, "lena13")
filterImage("lena_gray_256.tif", 25, 4, "lena25")
#
# calculate LoG for Camera Dude
filterImage("cameraman.tif", 7, 1, "cameraman7")
filterImage("cameraman.tif", 13, 2, "cameraman13")
filterImage("cameraman.tif", 25, 4, "cameraman25")

# calculate the LoG for the pictures, where the Gaussian and Laplacian is applied separately
GaussFirst("lena_gray_256.tif", 7, 1, "lena7G")
GaussFirst("lena_gray_256.tif", 13, 2, "lena13G")
GaussFirst("lena_gray_256.tif", 25, 4, "lena25G")

GaussFirst("cameraman.tif", 7, 1, "cameraman7G")
GaussFirst("cameraman.tif", 13, 2, "cameraman13G")
GaussFirst("cameraman.tif", 25, 4, "cameraman25G")
