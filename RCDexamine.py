import binascii, os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
import cv2

from astropy.io import fits
from sys import platform

def readxbytes(numbytes):
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
	print(data_chunk.shape)
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

# Start main program

# if platform == 'linux' or platform == 'linux2':
# 	inputfile = "./first1.rcd"
# elif platform == 'win32':
# 	inputfile = ".\\25ms_0000014.rcd"

inputfile = sys.argv[1]

start_time = time.time()

if len(sys.argv) > 2:
	imgain = sys.argv[2]
else:
	imgain = 'low'	# Which image/s to work with. Options: low, high, both (still to implement)

fid = open(inputfile, 'rb')
fid.seek(0,0)
magicnum = readxbytes(4) # 4 bytes ('Meta')
fid.seek(81,0)
hpixels = readxbytes(2) # Number of horizontal pixels
fid.seek(83,0)
vpixels = readxbytes(2) # Number of vertical pixels
fid.seek(85,0)
exptime = readxbytes(4) # Exposure time in 10.32us periods
fid.seek(89,0)
sensorcoldtemp = readxbytes(2)
fid.seek(91,0)
sensortemp = readxbytes(2)
fid.seek(99,0)
hbinning = readxbytes(1)
fid.seek(100,0)
vbinning = readxbytes(1)
fid.seek(141,0)
basetemp = readxbytes(2) # Sensor base temperature
fid.seek(152,0)
timestamp = readxbytes(30)
fid.seek(182,0)
lat = readxbytes(4)
fid.seek(186,0)
lon = readxbytes(4)

hbin = int(binascii.hexlify(hbinning),16)
vbin = int(binascii.hexlify(vbinning),16)
hpix = int(binascii.hexlify(hpixels),16)
vpix = int(binascii.hexlify(vpixels),16)
hnumpix = int(hpix / hbin)
vnumpix = int(vpix / vbin)

# Load data portion of file
fid.seek(256,0)

table = np.fromfile(fid, dtype=np.uint8)
testimages = nb_read_data(table)

image = split_images(testimages, hnumpix, vnumpix, imgain)

if imgain == 'both':
	image1 = split_images(testimages, hnumpix, vnumpix, 'low')
	image2 = split_images(testimages, hnumpix, vnumpix, 'high')

image = split_images(testimages, hnumpix, vnumpix, imgain)

latitude, longitude = computelatlong(lat,lon)

print("Observation lat/long: " + str(latitude) + "N / " + str(longitude) + "W")

print("--- %s seconds ---" % (time.time() - start_time))

if imgain == 'both':
	fig, ax = plt.subplots(2, figsize=(6,12))
	ax[0].imshow(image1, vmin=np.min(image1), vmax=np.mean(image1)*1.5)
	ax[0].text(0,-15, 'Low gain image...')
	ax[1].imshow(image2, vmin=np.min(image2), vmax=np.mean(image2)*1.5)
	ax[1].text(0,-15, 'High gain image...')
else:
	plt.figure(figsize=(10,10))
	plt.imshow(image, vmin=np.min(image), vmax=np.mean(image)*1.5)

	plt.text(0,-170, 'This is a ' + imgain + ' gain image...')
	plt.text(0,-130, timestamp)
	plt.text(0,-90, 'Temp: ' + str(int(binascii.hexlify(sensorcoldtemp), 16)) + 'C')
	plt.text(0, -50, 'Exposure time: ' + str(int(binascii.hexlify(exptime), 16) * 10.32 / 1000000) + ' seconds')
	plt.text(0,-10,"lat/long: " + str(latitude) + "N / " + str(longitude) + "W")

	plt.colorbar()

plt.tight_layout()
plt.show()

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