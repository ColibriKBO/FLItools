import binascii, os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
import imageio
import multiprocessing

from astropy.io import fits
from sys import platform
from joblib import Parallel, delayed

def readxbytes(numbytes, fid):
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

def file_write(imagelist, fileformat, file, lat, lon, exptime, sensorcoldtemp, timestamp):
	if fileformat == 'fits':
		latitude, longitude = computelatlong(lat,lon)
		hdu = fits.PrimaryHDU(imagelist)
		hdr = hdu.header
		hdr.set('exptime', int(binascii.hexlify(exptime), 16) * 10.32 / 1000000)
		hdr.set('DATE-OBS', str(timestamp, 'utf-8'))
		hdr.set('SITELAT', latitude)
		hdr.set('SITELONG', longitude)
		hdr.set('CCD-TEMP', int(binascii.hexlify(sensorcoldtemp), 16))
		hdu.writeto(file)

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

# def mkFits(filename):
# 	inputfile = os.path.splitext(filename)[0]
# 	fitsfile = inputfile + '.fits'
# 	fid = open(filename, 'rb')
# 	fid.seek(0,0)
# 	fid.seek(85,0)
# 	exptime = readxbytes(4) # Exposure time in 10.32us periods
# 	fid.seek(89,0)
# 	sensorcoldtemp = readxbytes(2)
# 	fid.seek(91,0)
# 	sensortemp = readxbytes(2)
# 	fid.seek(141,0)
# 	basetemp = readxbytes(2) # Sensor base temperature
# 	fid.seek(152,0)
# 	timestamp = readxbytes(29)
# 	fid.seek(182,0)
# 	lat = readxbytes(4)
# 	fid.seek(186,0)
# 	lon = readxbytes(4)
# 	hnumpix = 2048
# 	vnumpix = 2048

# 	# Load data portion of file
# 	fid.seek(246,0)

# 	table = np.fromfile(fid, dtype=np.uint8, count=12582912)
# 	testimages = nb_read_data(table, fid)
# 	image = split_images(testimages, hnumpix, vnumpix, imgain)
# 	image = split_images(testimages, hnumpix, vnumpix, imgain)
# 	file_write(image, 'fits', fitsfile)

def mkFits(filename):
	inputfile = os.path.splitext(filename)[0]
	fitsfile = inputfile + '.fits'
	fid = open(filename, 'rb')
	fid.seek(0,0)
	fid.seek(85,0)
	exptime = readxbytes(4, fid) # Exposure time in 10.32us periods
	fid.seek(89,0)
	sensorcoldtemp = readxbytes(2, fid)
	fid.seek(91,0)
	sensortemp = readxbytes(2, fid)
	fid.seek(141,0)
	basetemp = readxbytes(2, fid) # Sensor base temperature
	fid.seek(152,0)
	timestamp = readxbytes(29, fid)
	fid.seek(182,0)
	lat = readxbytes(4, fid)
	fid.seek(186,0)
	lon = readxbytes(4, fid)
	hnumpix = 2048
	vnumpix = 2048

	# Load data portion of file
	fid.seek(246,0)

	table = np.fromfile(fid, dtype=np.uint8, count=12582912)
	testimages = nb_read_data(table)
	image = split_images(testimages, hnumpix, vnumpix, imgain)
	image = split_images(testimages, hnumpix, vnumpix, imgain)
	file_write(image, 'fits', fitsfile, lat, lon, exptime, sensorcoldtemp, timestamp)

# Start main program
if __name__ == '__main__':

	if len(sys.argv) > 1:
		inputdir = sys.argv[1]

	if len(sys.argv) > 2:
		imgain = sys.argv[2]
	else:
		imgain = 'low'	# Which image/s to work with. Options: low, high, both (still to implement)

	globpath = inputdir + '*.rcd'

	# num_cores = multiprocessing.cpu_count()
	# print(num_cores)

	start_time = time.time()

	for filename in glob.glob(globpath):
		mkFits(filename)

	# Parallel(n_jobs=8)(delayed(mkFits)(i) for i in glob.glob(globpath))


	print("--- %s seconds ---" % (time.time() - start_time))

# Vid file header...
# uint32  magic   four byte "magic number"
#                 should always be 809789782
# uint32  seqlen  total byte length of frame+header
# uint32  headlen byte length of the header
# uint32  flags   if (flags & 64) then frame has a problem
#                 "problem" is poorly defined
# uint32  seq     sequence number - count of frames since
#                 a recording run started, begins at 0.
#                 Should always increase by 1 - anything else
#                 indicates a frame may have been dropped.
# int32   ts      seconds since the UNIX epoch
# int32   tu      microseconds elapsed since last second (ts) began
# int16   num     station identifier number
# int16   wid     frame width, in pixels
# int16   ht      frame height, in pixels
# int16   depth   bit-depth of image
# uint16  hx      not used in this system - ignore
# uint16  ht      bit used in this system - ignore
# uint16  str     stream ID for sites w/ multiple cameras
# uint16  reserved0 (unused)
# uint32  expose  exposure time in milliseconds (unused)
# uint32  reserved2 (unused)
# char[64] text   string containing a short description of the
#                 .vid file contents (ex "Elgin_SN09149652_EM100")