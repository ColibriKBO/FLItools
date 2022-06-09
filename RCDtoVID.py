import binascii, os, sys, glob, array, random, re, math
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
import imageio
import cv2
from datetime import datetime

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

def stackBlats(impath,hiclips,loclips):
	stackArray = np.zeros([2048,2048])
	hiArray = np.zeros([2048,2048,hiclips+1])
	loArray = np.zeros([2048,2048,loclips+1])
	hiTempArray = np.zeros([2048,2048])
	loTempArray = np.zeros([2048,2048])
	hiLoTempArray = np.zeros([2048,2048])
	
	stackcount = 0

	hnumpix = 2048
	vnumpix = 2048

	for filename in glob.glob(impath, recursive=True):
		inputfile = os.path.splitext(filename)[0]

		fid = open(filename, 'rb')

		# Load data portion of file
		fid.seek(384,0)

		table = np.fromfile(fid, dtype=np.uint8, count=12582912)
		testimages = nb_read_data(table)

		image = split_images(testimages, hnumpix, vnumpix, 'low')

		###
		# Section to clip data for bias
		###
		if (hiclips > 0) and (loclips == 0):

			np.copyto(hiArray[:,:,-1],image)
			hiArray = -np.sort(-hiArray,axis=2)
			np.copyto(hiTempArray,hiArray[:,:,-1])
			stackArray = np.add(stackArray,hiTempArray)

		if (hiclips == 0) and (loclips > 0):
			np.copyto(loArray[:,:,-1],image)
			loArray = -np.sort(-loArray,axis=2)
			np.copyto(loTempArray,loArray[:,:,0])
			if stackcount > loclips:
				stackArray = np.add(stackArray,loTempArray)

		if (hiclips == 0) and (loclips == 0):
			stackArray = np.add(stackArray,image)

		stackcount += 1

	biasimage = stackArray/(stackcount-hiclips-loclips)

	return biasimage

def vidWriter(indir, bias, outfile):
	n = 0
	# Take input directory and loop through images and write them to a vid file
	# From each file extract the time and convert to unix time
	for (path, dirs, files) in os.walk(indir):
		for file in files:
			# if n == 100:
			# 	return
			if file.endswith('.rcd'):

				width = 2048
				height = 2048

				magic = 809789782 # 4 bytes
				seqlen = width*height*2 # 4 bytes
				headlen = 116 # 4 bytes
				flags = 999 # 4 bytes
				seq = n # 4 bytes
				# seq = 0 # 4 bytes
				# ts = time_sec # 4 bytes
				# tu = time_ms*1000+time_us # 4 bytes
				num = 1 # 2 bytes
				wid = width # 2 bytes
				ht = height # 2 bytes
				depth = 16 # 2 bytes
				hx = 0 # 2 bytes
				ht = 0 # 2 bytes
				cam = 15 # 2 bytes
				reserved0 = 000 # 2 bytes
				expose = 25 # 4 bytes
				reserved2 = 000 # 4 bytes
				text = "Colibri-50cm" # 64 bytes
				# print(path)
				# print(file)
				print(seq)
				table, hdict = readRCD(path + '\\' + file)
				timestamp1 = hdict['timestamp'].decode('utf-8')
				# print(timestamp1[11:26])
				# print(datetime.fromisoformat(str(timestamp1[:23])))
				timeparts = re.split('[-T:]',timestamp1[:26])
				# print(timeparts)

				if int(timeparts[3]) > 24:
					actualtime = re.split('[_.]',path)
					timestamp3 = '2021-08-13T' + actualtime[1] + ':' + timestamp1[14:23]
					# print(timestamp1)
					timestamp4 = datetime.strptime(timestamp3[:23],'%Y-%m-%dT%H:%M:%S.%f')
					unixtime = datetime.timestamp(timestamp4)
					print('Time corrected to %s' % timestamp4)
				else:
					timestamp2 = datetime.strptime(timestamp1[:23],'%Y-%m-%dT%H:%M:%S.%f')
					unixtime = datetime.timestamp(timestamp2)

				ts = math.floor(unixtime)
				tu = round((float(timeparts[5])%1*1000000))

				testimages = nb_read_data(table)

				image = split_images(testimages, width, height, 'low')
				image = image+100
				image = np.subtract(image, bias)
				image = image.astype('int16')

				# plt.figure()
				# plt.imshow(image3)
				# plt.show()


				# Write to file in big endian order
				# if sys.byteorder == "little":
				#     a.byteswap()
				with open(outfile, "ab") as f:
					f.write((magic).to_bytes(4, byteorder='little'))
					f.write((seqlen).to_bytes(4, byteorder='little'))
					f.write((headlen).to_bytes(4, byteorder='little'))
					f.write((flags).to_bytes(4, byteorder='little'))
					f.write((seq).to_bytes(4, byteorder='little'))
					f.write((ts).to_bytes(4, byteorder='little'))
					f.write((tu).to_bytes(4, byteorder='little'))
					f.write((num).to_bytes(2, byteorder='little'))
					f.write((width).to_bytes(2, byteorder='little'))
					f.write((height).to_bytes(2, byteorder='little'))
					f.write((depth).to_bytes(2, byteorder='little'))
					f.write((hx).to_bytes(2, byteorder='little'))
					f.write((ht).to_bytes(2, byteorder='little'))
					f.write((cam).to_bytes(2, byteorder='little'))
					f.write((reserved0).to_bytes(2, byteorder='little'))
					f.write((expose).to_bytes(4, byteorder='little'))
					f.write((reserved2).to_bytes(4, byteorder='little'))
					f.write(text.encode())
					for i in range(52):
						f.write((0).to_bytes(1,byteorder='little'))

					flatimage = image.flatten()[58:]
					# flatimage.byteswap(inplace=True)
					flatimage.tofile(f)

					# print(text.encode())
				n += 1

def mp4Writer(indir, bias, outfile):
	n = 0
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(outfile, fourcc, 40.0, (2048,2048),0)
	# Take input directory and loop through images and write them to a vid file
	# From each file extract the time and convert to unix time
	for (path, dirs, files) in os.walk(indir):
		for file in files:
			# if n == 100:
			# 	return
			if file.endswith('.rcd'):
				if n%25 == 0:
					print(n)

				width = 2048
				height = 2048

				magic = 809789782 # 4 bytes
				seqlen = width*height*2 # 4 bytes
				headlen = 116 # 4 bytes
				flags = 999 # 4 bytes
				seq = n # 4 bytes
				# seq = 0 # 4 bytes
				# ts = time_sec # 4 bytes
				# tu = time_ms*1000+time_us # 4 bytes
				num = 1 # 2 bytes
				wid = width # 2 bytes
				ht = height # 2 bytes
				depth = 16 # 2 bytes
				hx = 0 # 2 bytes
				ht = 0 # 2 bytes
				cam = 15 # 2 bytes
				reserved0 = 000 # 2 bytes
				expose = 25 # 4 bytes
				reserved2 = 000 # 4 bytes
				text = "Colibri-50cm" # 64 bytes
				# print(path)
				# print(file)
				# print(seq)
				table, hdict = readRCD(path + '\\' + file)
				timestamp1 = hdict['timestamp'].decode('utf-8')
				# print(timestamp1[11:26])
				# print(datetime.fromisoformat(str(timestamp1[:23])))
				timeparts = re.split('[-T:]',timestamp1[:26])
				# print(timeparts)

				if int(timeparts[3]) > 24:
					actualtime = re.split('[_.]',path)
					timestamp3 = '2021-08-13T' + actualtime[1] + ':' + timestamp1[14:23]
					# print(timestamp1)
					timestamp4 = datetime.strptime(timestamp3[:23],'%Y-%m-%dT%H:%M:%S.%f')
					unixtime = datetime.timestamp(timestamp4)
					print('Time corrected to %s' % timestamp4)
				else:
					timestamp2 = datetime.strptime(timestamp1[:23],'%Y-%m-%dT%H:%M:%S.%f')
					unixtime = datetime.timestamp(timestamp2)

				ts = math.floor(unixtime)
				tu = round((float(timeparts[5])%1*1000000))

				testimages = nb_read_data(table)

				image = split_images(testimages, width, height, 'low')
				# image = image+100
				# image = image*100
				image = np.subtract(image, bias)
				image = image.astype('int8')

				# plt.figure()
				# plt.imshow(image)
				# plt.show()

				out.write(image)

				# cv2.imshow('image', image)
				# c = cv2.waitKey(1)
				# if c & 0xFF == ord('q'):
				# 	break

				# Write to file in big endian order
				# if sys.byteorder == "little":
				#     a.byteswap()
				# with open(outfile, "ab") as f:
				# 	f.write((magic).to_bytes(4, byteorder='little'))
				# 	f.write((seqlen).to_bytes(4, byteorder='little'))
				# 	f.write((headlen).to_bytes(4, byteorder='little'))
				# 	f.write((flags).to_bytes(4, byteorder='little'))
				# 	f.write((seq).to_bytes(4, byteorder='little'))
				# 	f.write((ts).to_bytes(4, byteorder='little'))
				# 	f.write((tu).to_bytes(4, byteorder='little'))
				# 	f.write((num).to_bytes(2, byteorder='little'))
				# 	f.write((width).to_bytes(2, byteorder='little'))
				# 	f.write((height).to_bytes(2, byteorder='little'))
				# 	f.write((depth).to_bytes(2, byteorder='little'))
				# 	f.write((hx).to_bytes(2, byteorder='little'))
				# 	f.write((ht).to_bytes(2, byteorder='little'))
				# 	f.write((cam).to_bytes(2, byteorder='little'))
				# 	f.write((reserved0).to_bytes(2, byteorder='little'))
				# 	f.write((expose).to_bytes(4, byteorder='little'))
				# 	f.write((reserved2).to_bytes(4, byteorder='little'))
				# 	f.write(text.encode())
				# 	for i in range(52):
				# 		f.write((0).to_bytes(1,byteorder='little'))

				# 	flatimage = image.flatten()[58:]
				# 	# flatimage.byteswap(inplace=True)
				# 	flatimage.tofile(f)

					# print(text.encode())
				n += 1
	out.release()
	cv2.destroyAllWindows()

def readRCD(filename):

	hdict = {}

	fid = open(filename, 'rb')
	fid.seek(0,0)
	
	# magicnum = readxbytes(4) # 4 bytes ('Meta')
	fid.seek(81,0)
	hdict['hpixels'] = readxbytes(fid,2) # Number of horizontal pixels
	fid.seek(83,0)
	hdict['vpixels'] = readxbytes(fid,2) # Number of vertical pixels

	fid.seek(63,0)
	hdict['serialnum'] = readxbytes(fid, 9) # Serial number of camera
	fid.seek(85,0)
	hdict['exptime'] = readxbytes(fid, 4) # Exposure time in 10.32us periods

	fid.seek(99,0)
	hdict['binning'] = readxbytes(fid,1)

	fid.seek(152,0)
	hdict['timestamp'] = readxbytes(fid, 29)

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
	# if len(sys.argv) > 1:
	# 	inputdir = sys.argv[1]

	# if len(sys.argv) > 2:
	# 	imgain = sys.argv[2]
	# else:
	# 	imgain = 'low'	# Which image/s to work with. Options: low, high, both (still to implement)

	# globpath = inputdir + '**\\*.rcd'
	# print(globpath)

	# hnumpix = 2048
	# vnumpix = 2048

	start_time = time.time()

	# for filename in glob.glob(globpath, recursive=True):
	# 	inputfile = os.path.splitext(filename)[0]


	# 	table, hdict = readRCD(filename)

	# 	testimages = nb_read_data(table)

	# D:\ColibriData\20220531\20220531_02.51.48.882
	# D:\ColibriData\20220531\Bias\20220531_02.12.39.974

	# 	image = split_images(testimages, hnumpix, vnumpix, imgain)
	biaspath = 'D:\\ColibriData\\20220531\\Bias\\20220531_02.12.39.974\\*.rcd'
	biasimage = stackBlats(biaspath,0,0)

	for (path, dirs, files) in os.walk('D:\\ColibriData\\20220531'):
		for directory in dirs:
			# print(path + ' ' + directory)
			if directory != 'Bias':
				if not os.path.isfile('D:\\TauHerculids\\' + directory + '.vid'):
					print('Creating %s.mp4' % directory)
					# print(path + '\\' + directory)
					vidWriter(path + '\\' + directory + '\\', biasimage, 'D:\\TauHerculids\\' + directory + '.vid')
					# mp4Writer(path + '\\' + directory + '\\', biasimage, 'D:\\TauHerculids\\' + directory + '.avi')
	print("--- %s seconds ---" % (time.time() - start_time))