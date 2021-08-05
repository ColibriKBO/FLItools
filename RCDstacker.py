import binascii, os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
import imageio

from astropy.io import fits
from sys import platform

def readxbytes(fid,numbytes):
	for i in range(1):
		data = fid.read(numbytes)
		if not data:
			break
	return data

@nb.njit(nb.uint16[::1](nb.uint8[::1]),fastmath=True,parallel=True)
def nb_read_data(data_chunk):
	"""data_chunk is a contigous 1D array of uint8 data)
	eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)"""
	#ensure that the data_chunk has the right length

	assert np.mod(data_chunk.shape[0],3)==0

	out=np.empty(data_chunk.shape[0]//3*2,dtype=np.uint16)
	image1 = np.empty((2048,2048),dtype=np.uint16)
	image2 = np.empty((2048,2048),dtype=np.uint16)

	for i in nb.prange(data_chunk.shape[0]//3):
		fst_uint8=np.uint16(data_chunk[i*3])
		mid_uint8=np.uint16(data_chunk[i*3+1])
		lst_uint8=np.uint16(data_chunk[i*3+2])

		out[i*2] =   (fst_uint8 << 4) + (mid_uint8 >> 4)
		out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8

	return out

def split_images(data,pix_h,pix_v,gain):
	interimg = np.reshape(data, [2*pix_v,pix_h])

	if gain == 'low':
		image = interimg[::2]
	else:
		image = interimg[1::2]

	return image

def file_write(imagelist, fileformat, file):
	if fileformat == 'fits':
		# latitude, longitude = computelatlong(lat,lon)
		hdu = fits.PrimaryHDU(imagelist)
		hdr = hdu.header
		# hdr.set('exptime', int(binascii.hexlify(exptime), 16) * 10.32 / 1000000)
		# hdr.set('DATE-OBS', str(timestamp, 'utf-8'))
		# hdr.set('SITELAT', latitude)
		# hdr.set('SITELONG', longitude)
		# hdr.set('CCD-TEMP', int(binascii.hexlify(sensorcoldtemp), 16))
		hdu.writeto(file, overwrite=True)

def computelatlong(lat,lon): # Calculate Latitude and Longitude
	degdivisor = 600000.0
	degmask = int(0x7fffffff)
	dirmask = int(0x80000000)

	latraw = int(binascii.hexlify(lat),16)
	lonraw = int(binascii.hexlify(lon),16)

	if (latraw & dirmask) != 0:
		latitude = (latraw & degmask) / degdivisor
	else:
		latitude = -1*(latraw & degmask) / degdivisor

	if (lonraw & dirmask) != 0:
		longitude = (lonraw & degmask) / degdivisor
	else:
		longitude = -1*(lonraw & degmask) / degdivisor

	return latitude, longitude

def stackBlats(impath,hiclips,loclips):
	stackArray = np.zeros([2048,2048])
	hiArray = np.zeros([2048,2048,hiclips+1])
	loArray = np.zeros([2048,2048,loclips+1])
	tempArray = np.zeros([2048,2048])
	
	stackcount = 0

	hnumpix = 2048
	vnumpix = 2048

	for filename in glob.glob(impath, recursive=True):
		print(filename)
		inputfile = os.path.splitext(filename)[0]

		fid = open(filename, 'rb')

		# Load data portion of file
		fid.seek(384,0)

		table = np.fromfile(fid, dtype=np.uint8, count=12582912)
		testimages = nb_read_data(table)

		image = split_images(testimages, hnumpix, vnumpix, imgain)

		###
		# Section to clip data for bias
		###
		if hiclips > 0:
			tempArray1 = np.zeros([2048,2048])
			tempArray2 = np.zeros([2048,2048])
			tempArray3 = np.zeros([2048,2048])
			# hiArray = -np.sort(-hiArray,axis=2)

			np.copyto(hiArray[:,:,-1],image)
			hiArray = -np.sort(-hiArray,axis=2)

			np.copyto(tempArray,hiArray[:,:,-1])
			# np.copyto(hiArray[:,:,-1],image,where=image>hiArray[:,:,-1])
			# np.copyto(tempArray1,image,where=image<hiArray[:,:,-1])
			# np.copyto(tempArray2,hiArray[:,:,-1],where=hiArray[:,:,-1]<image)
			# tempArray3 = np.add(tempArray1,tempArray2)
			# plt.imshow(tempArray2)
			# plt.show()
			# print(hiArray[1,0,:])
			# print(tempArray1[1,0])

			# for i in range(2048):
			# 	for j in range(2048):
			# 		if image[i,j] > hiArray[i,j,-1]:
			# 			tempArray[i,j] = hiArray[i,j,-1]
			# 			hiArray[i,j,-1] = image[i,j]

			stackArray = np.add(stackArray,tempArray)
		else:
			stackArray = np.add(stackArray,image)

		stackcount += 1

	biasimage = stackArray/(stackcount-hiclips)

	return biasimage

def stackImages(impath, bias):

	stackArray = np.zeros([2048,2048])
	stackcount = 0

	for filename in glob.glob(impath, recursive=True):
		inputfile = os.path.splitext(filename)[0]

		hnumpix = 2048
		vnumpix = 2048

		fid = open(filename, 'rb')
		fid.seek(384,0)

		table = np.fromfile(fid, dtype=np.uint8, count=12582912)
		testimages = nb_read_data(table)

		image = split_images(testimages, hnumpix, vnumpix, imgain)
		image = np.subtract(image,bias)

		stackArray = np.add(stackArray,image)
		stackcount += 1
		# imageio.imwrite(inputfile + '.png', image)

	imagestack = stackArray/stackcount

	return imagestack

# Start main program
if __name__ == '__main__':

	if len(sys.argv) > 1:
		inputdir = sys.argv[1]

	if len(sys.argv) > 2:
		imgain = sys.argv[2]
	else:
		imgain = 'low'	# Which image/s to work with. Options: low, high, both (still to implement)

	globpath = inputdir + '**\\*.rcd'
	biaspath = inputdir + '\\bias\\' + '**\\*.rcd'
	fitsfile = inputdir + '\\stack.fts'

	start_time = time.time()

	biasImage = stackBlats(biaspath,3,0)
	stackImage = stackImages(globpath,biasImage)
	file_write(stackImage, 'fits', fitsfile)
	file_write(biasImage, 'fits', 'C:\\Users\\Mike\\Downloads\\RCD\\bias.fits')

	print("--- %s seconds ---" % (time.time() - start_time))

	plt.imshow(biasImage, vmin=80, vmax=120)
	plt.show()

	plt.imshow(stackImage+5, vmin=0, vmax=30)
	plt.show()
