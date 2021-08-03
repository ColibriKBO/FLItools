import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.nddata import CCDData
import ccdproc

bias_file = ".\\testimages\\bias.fits"
image_file = ".\\testimages\\image.fits"

# bias_list = fits.open(bias_file)
# image_list = fits.open(image_file)

bias_data = fits.getdata(bias_file)
image_data = fits.getdata(image_file)

bias = CCDData(bias_data, unit=u.adu)
data = CCDData(image_data, unit=u.adu)

bias_subtracted = ccdproc.subtract_bias(data, bias)

plt.imshow(bias,vmin=100,vmax=120)
plt.show()
plt.imshow(bias_subtracted, vmin=-5, vmax=5)
plt.show()