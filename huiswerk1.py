import numpy as np
import scipy as sp
import scipy.ndimage as im
import scipy.misc as misc
import matplotlib.pyplot as plot

mask = np.array(([0, 1, 0], [1,0,1], [0,1,0]))
image = im.imread('lena.png', flatten=True).astype(np.uint8)
misc.imsave('lena_grey.png', image)
plot.imshow(image)

dilation = im.grey_dilation(image, structure=mask)
misc.imsave('dilation.png', dilation)
plot.imshow(dilation)

erosion = im.grey_erosion(image, structure=mask)
misc.imsave('erosie.png', erosion)
plot.imshow(erosion)

open = im.grey_opening(image, structure=mask)
misc.imsave('open.png',open)

close = im.grey_closing(image, structure=mask)
misc.imsave('close.png', close)