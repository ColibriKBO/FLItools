import binascii
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
#import cv2

from astropy.io import fits
from sys import platform

@nb.njit(nb.uint16[::1](nb.uint8[::1]),fastmath=True,parallel=True)
def nb_read_uint12(data_chunk):
  """data_chunk is a contigous 1D array of uint8 data)
  eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)"""

  #ensure that the data_chunk has the right length
  assert np.mod(data_chunk.shape[0],3)==0

  out=np.empty(data_chunk.shape[0]//3*2,dtype=np.uint16)

  for i in nb.prange(data_chunk.shape[0]//3):
    fst_uint8=np.uint16(data_chunk[i*3])
    mid_uint8=np.uint16(data_chunk[i*3+1])
    lst_uint8=np.uint16(data_chunk[i*3+2])

    out[i*2] =   (fst_uint8 << 4) + (mid_uint8 >> 4)
    out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8


  return out

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

def read_uint12(data_chunk): # From https://stackoverflow.com/questions/44735756/python-reading-12-bit-binary-files?rq=1
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])

def file_write(imagelist, fileformat, filename):
	if fileformat == 'fits':
		hdu = fits.PrimaryHDU(imagelist)
		hdu.writeto(filename)

def split_images(data,pix_h,pix_v):

	interimg = np.reshape(data, [2*pix_v,pix_h])

	# if imgain == 'logain':
	# 	print('logain')
	# 	image1 = interimg[::2]
	# 	return image1
	# elif imgain == 'higain':
	# 	image2 = interimg[1::2]
	# 	print('higain')
	# 	return image2
	# else:
	# 	print('both')
	# 	image1 = interimg[::2]
	# 	image2 = interimg[1::2]
	# 	return image1, image2

	image1 = interimg[::2]
	image2 = interimg[1::2]
	return image1, image2
		


def readxbytes(numbytes):
	for i in range(1):
		data = fid.read(numbytes)
		if not data:
			break
	return data

start_time = time.time()

if platform == 'linux' or platform == 'linux2':
	inputfile = "./first1.rcd"
	print('Linux')
elif platform == 'win32':
	inputfile = ".\\first1.rcd"
	print('Windows')

imgain = 'logain'	# Which image/s to work with. Options: logain, higain, both
imgfmt = 'fits'	# Output file format. Options: fits, vid

# inputfile = "C:\\Users\\Mike\\Pictures\\Testing\\Stream\\test.rcd"


filesize = os.path.getsize(inputfile)

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
fid.seek(93,0)
globalgain = readxbytes(1)
# fid.seek(94,0)
# ldrgain = readxbytes(1)
# fid.seek(95,0)
# hdrgain = readxbytes(1)
fid.seek(99,0)
hbinning = readxbytes(1)
fid.seek(100,0)
vbinning = readxbytes(1)
fid.seek(141,0)
basetemp = readxbytes(2) # Sensor base temperature
fid.seek(143,0)
fpgatemp = readxbytes(2)
fid.seek(145,0)
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
expt = int(binascii.hexlify(exptime), 16) * 10.32 / 1000
cldt = int(binascii.hexlify(sensorcoldtemp), 16)
hnumpix = int(hpix / hbin)
vnumpix = int(vpix / vbin)

print(hnumpix)
# Load data portion of file
data_size = hnumpix * vnumpix
fid.seek(246,0)
table = np.fromfile(fid, dtype=np.uint8)
# testimages = nb_read_uint12(table)
testimages = nb_read_data(table)

# if imgain == 'logain':
# 	image1 = split_images(testimages, hnumpix, vnumpix)
# elif imgain == 'higain':
# 	image2 = split_images(testimages, hnumpix, vnumpix)
# else:
# 	image1, image2 = split_images(testimages, hnumpix, vnumpix)

image1, image2 = split_images(testimages, hnumpix, vnumpix)
imsmall = image1[0:2,0:2]
imsmall[0,0] = 1
imsmall[0,1] = 2
imsmall[1,0] = 3
imsmall[1,1] = 4

lst = []
for i in range(5):
	lst.append(imsmall)

# mm = cv2.createMergeMertens()
# merge = mm.process((image1,image2))
# print(np.max(merge))

filestk = np.stack((lst))

file_write(filestk, imgfmt, 'test.fits')

print("# of pixels (width * height): " + str(hpix) + " * " + str(vpix))
print("Binning factor: " + str(hbin) + " x " + str(vbin))

degdivisor = 600000.0
degmask = int(0x7fffffff)
dirmask = int(0x80000000)

latraw = int(binascii.hexlify(lat),16)
lonraw = int(binascii.hexlify(lon),16)

# Calculate Latitude and Longitude
if (latraw & dirmask) != 0:
	latdec = (latraw & degmask) / degdivisor
else:
	latdec = -1*(latraw & degmask) / degdivisor

if (lonraw & dirmask) != 0:	
	londec = (lonraw & degmask) / degdivisor
else:
	londec = -1*(lonraw & degmask) / degdivisor

print("Observation lat/long: " + str(latdec) + "N / " + str(londec) + "W")
# print int(binascii.hexlify(hpixels), 16), int(binascii.hexlify(vpixels), 16)

print("--- %s seconds ---" % (time.time() - start_time))

plt.imshow(image1, vmin=np.min(image1), vmax=np.max(image1)*0.25)
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