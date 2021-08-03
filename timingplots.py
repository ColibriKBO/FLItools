#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:46:28 2021

@author: rbrown

makes plot of time of all images (from rcd)

add filepath to parent (night) directory with / as argument to run
saves plots of each minute's timings, as well as any anomalous 
events in directory 'filename' at end of script
"""

import binascii, os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
from astropy.time import Time

from astropy.io import fits
from sys import platform

def readxbytes(numbytes):
	for i in range(1):
		data = fid.read(numbytes)
		if not data:
			break
	return data

# Start main program

inputdir = sys.argv[1]
if len(sys.argv) > 1:
	dirlist = sys.argv[1]

if len(sys.argv) > 2:
	imgain = sys.argv[2]
    
globpath1 = glob.glob(dirlist+'*/')

for inputdir in globpath1:
    print(inputdir)
    globpath = inputdir + '*.rcd'
#    print(globpath)

  #  start_time = time.time()

    fileList = []
    timeList = []
    listr = glob.glob(globpath)

    for filename in glob.glob(globpath):
        inputfile = os.path.splitext(filename)[0]   

        fid = open(filename, 'rb')
        fid.seek(152,0)
        timestamp = readxbytes(30)
  
        hour = str(timestamp).split('T')[1].split(':')[0]
        fileMinute = str(timestamp).split(':')[1]
        dirMinute = inputdir.split('_')[1].split('.')[1]
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:

            print('Bad time: ', hour)
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            newLocalHour = int(inputdir.split('_')[1].split('.')[0])
        
            if int(fileMinute) < int(dirMinute):
                newUTCHour = newLocalHour + 4 + 1     #add 1 if hour changed over during minute
            else:
                newUTCHour = newLocalHour + 4
        
            #replace bad hour in timestamp string with correct hour
            newUTCHour = str(newUTCHour)
            newUTCHour = newUTCHour.zfill(2)
        
            replaced = str(timestamp).replace('T' + hour, 'T' + newUTCHour).strip('b').strip(' \' ')
        
            #encode into bytes
            #newTimestamp = replaced.encode('utf-8')
            timestamp = replaced
        
        

        time = float(str(timestamp).split(':')[2].strip('\'').strip('Z'))
        fileList.append(inputfile.split('_')[3].strip('.rcd').lstrip('0'))
        #timeList.append(float(str(timestamp).split(':')[2].strip('\'').strip('Z')))
        timeList.append(str(timestamp).strip('b').strip('\'').strip('Z'))

    zero = Time(timeList[0], precision = 9).unix
    for i in range(0, len(timeList)):
        # print('before: ', timeList[i])
        timeList[i] = Time(timeList[i], precision = 9).unix
    
        timeList[i] = timeList[i] - zero
        # print('after: ', timeList[i])
        #if timeList[i] < timeList[i-1]:
            #    timeList[i] = timeList[i] + 60.
            
    for i in range(1, len(timeList)):
        if timeList[i] < timeList[i-1]:
            print('GREAT SCOTT! ', fileList[i], fileList[i-1] )
            
            fig, ax = plt.subplots()
            
            ax.plot(fileList[i-50:i+50], timeList[i-50:i+50])
            
            ax.set_xticks(ax.get_xticks()[::10])
            ax.set_xlabel('frame number')
            ax.set_ylabel('seconds since beginning of minute')
            ax.set_title(inputdir)
            #labels = fileList[::200]
            #ax.set_xticklabels(labels, rotation = 45)
            
            #plt.show()
            filename = './ColibriArchive/timingplots/timeTravel_' + fileList[i]+ '_' + inputdir.split('_')[1].strip('\\') + '.png'
 #        print(filename)
            plt.savefig(filename)
            plt.close()
            

    fig, ax = plt.subplots()

    ax.plot(fileList, timeList)

    ax.set_xticks(ax.get_xticks()[::300])
    ax.set_xlabel('frame number')
    ax.set_ylabel('seconds since beginning of minute')
    ax.set_title(inputdir.split('_')[1])
    #labels = fileList[::200]
    #ax.set_xticklabels(labels, rotation = 45)

    #plt.show()
    filename = './ColibriArchive/timingplots/timing_' + inputdir.split('_')[1].strip('\\') + '.png'
 #   print(filename)
    plt.savefig(filename)
    plt.close()

