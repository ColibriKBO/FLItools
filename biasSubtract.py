from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import sys, os

# imdir = sys.argv[1]
imdir = 'C:\\Users\\Mike\\Pictures\\test'

biasname = 'bias-high.fits'
bias_file = os.path.join(imdir,'bias-high.fits')
#bias_file = 'C:\\Users\\Mike\\Pictures\\bias-high.fits'

#os.mkdir(imdir + '\\biascorrected')

for filename in os.listdir(imdir):
	if filename != biasname:
		# print(os.path.join(imdir,filename))
		# print(filename)
		image_file = os.path.join(imdir,filename)

		image_data = fits.getdata(image_file)
		bias_data = fits.getdata(bias_file)

		subimg = image_data - bias_data
		# addimg = image_data + bias_data

		outfile = os.path.join(imdir,'b_' + filename)
		# print(outfile)

		hdu = fits.PrimaryHDU(subimg)
		hdu.writeto(outfile, overwrite=True)

		# print(type(image_data))
		# print(image_data.shape)


		# print('Min:', np.min(image_data))
		# print('Max:', np.max(image_data))
		# print('Mean:', np.mean(image_data))
		# print('Stdev:', np.std(image_data))

