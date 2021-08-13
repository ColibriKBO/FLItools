import binascii, os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
import imageio

from astropy.io import fits
from sys import platform

def readxbytes(fid, numbytes):
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
		latitude, longitude = computelatlong(hdict['lat'],hdict['lon'])
		hdu = fits.PrimaryHDU(imagelist)
		hdr = hdu.header
		hdr.set('exptime', int(binascii.hexlify(hdict['exptime']), 16) * 10.32 / 1000000)
		hdr.set('DATE-OBS', str(hdict['timestamp'], 'utf-8'))
		hdr.set('SITELAT', latitude)
		hdr.set('SITELONG', longitude)
		hdr.set('CCD-TEMP', int(binascii.hexlify(hdict['sensorcoldtemp']), 16))
		hdr.set('SERIAL', str(hdict['serialnum'], 'utf-8'))
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


def readRCD(filename):

	hdict = {}

	fid = open(filename, 'rb')
	fid.seek(0,0)
	
	# magicnum = readxbytes(4) # 4 bytes ('Meta')
	# fid.seek(81,0)
	# hpixels = readxbytes(2) # Number of horizontal pixels
	# fid.seek(83,0)
	# vpixels = readxbytes(2) # Number of vertical pixels

	fid.seek(63,0)
	hdict['serialnum'] = readxbytes(fid, 9) # Serial number of camera
	fid.seek(85,0)
	hdict['exptime'] = readxbytes(fid, 4) # Exposure time in 10.32us periods
	fid.seek(89,0)
	hdict['sensorcoldtemp'] = readxbytes(fid, 2)
	fid.seek(91,0)
	hdict['sensortemp'] = readxbytes(fid, 2)

	# fid.seek(99,0)
	# hbinning = readxbytes(1)
	# fid.seek(100,0)
	# vbinning = readxbytes(1)

	fid.seek(141,0)
	hdict['basetemp'] = readxbytes(fid, 2) # Sensor base temperature
	fid.seek(152,0)
	hdict['timestamp'] = readxbytes(fid, 29)
	fid.seek(182,0)
	hdict['lat'] = readxbytes(fid, 4)
	fid.seek(186,0)
	hdict['lon'] = readxbytes(fid, 4)

	# hbin = int(binascii.hexlify(hbinning),16)
	# vbin = int(binascii.hexlify(vbinning),16)
	# hpix = int(binascii.hexlify(hpixels),16)
	# vpix = int(binascii.hexlify(vpixels),16)
	# hnumpix = int(hpix / hbin)
	# vnumpix = int(vpix / vbin)

	# Load data portion of file
	fid.seek(384,0)

	table = np.fromfile(fid, dtype=np.uint8, count=12582912)

	return table, hdict

# Start main program
if __name__ == '__main__':
	if len(sys.argv) > 1:
		inputdir = sys.argv[1]

	if len(sys.argv) > 2:
		imgain = sys.argv[2]
	else:
		imgain = 'low'	# Which image/s to work with. Options: low, high, both (still to implement)

	globpath = inputdir + '**\\*.rcd'
	print(globpath)

	hnumpix = 2048
	vnumpix = 2048

	start_time = time.time()

	for filename in glob.glob(globpath, recursive=True):
		inputfile = os.path.splitext(filename)[0]
		fitsfile = inputfile + '.fits'

		table, hdict = readRCD(filename)

		testimages = nb_read_data(table)

		image = split_images(testimages, hnumpix, vnumpix, imgain)

		# if imgain == 'both':
		# 	image1 = split_images(testimages, hnumpix, vnumpix, 'low')
		# 	image2 = split_images(testimages, hnumpix, vnumpix, 'high')

		file_write(image, 'fits', fitsfile)

	print("--- %s seconds ---" % (time.time() - start_time))