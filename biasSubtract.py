from astropy.io import fits
# import matplotlib.pyplot as plt
# import numpy as np
import sys, os

imdir = sys.argv[1]

biasname = 'bias-high.fits'
bias_file = os.path.join(imdir,'bias-high.fits')

for filename in os.listdir(imdir):
	if filename != biasname:
		image_file = os.path.join(imdir,filename)
		image_data = fits.getdata(image_file)
		bias_data = fits.getdata(bias_file)
		subimg = image_data - bias_data
		outfile = os.path.join(imdir,'b_' + filename)
		hdu = fits.PrimaryHDU(subimg)
		hdu.writeto(outfile, overwrite=True)
		print('Wrote %s' % outfile)