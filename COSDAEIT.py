#An algorithm to identify local coronal intensity peaks in the limb using both unprocessed and MGN processed SOHO EIT images. Based on OCCULT-2 produced by Dr. Markus Aschwanden & Dr Peter Hardi and Multi-Gaussian-Normalization produced by Dr. Huw Morgan and Dr Miloslav Druckmuller
import numpy as np
#import numpy.ma as ma
import sunpy.map
#import sunpy.io
#import sunpy.image
# from astropy.coordinates import SkyCoord
# from sunpy.coordinates import frames
import astropy.units as u
import copy
#from scipy.interpolate import CubicSpline
# from scipy.interpolate import UnivariateSpline
#from scipy import signal
#import scipy.ndimage
from scipy.optimize import curve_fit as cf
import math
import matplotlib.pyplot as plt
import os,sys
from multiprocessing import Pool
import tqdm
#import aiaprep
import mgncv
import cv2 as cv
from numba import njit
import glob
import warnings
# import time
# from numpy import unravel_index
# import pandas as pd
warnings.simplefilter('ignore')

#INIT ARRAY & VARIABLES

def widthmeasure(file1):
	#print(file1)
	baseang = []
	brights = []
	widths = []
	uncertainty = []
	# aiamgn = (np.zeros((4096,1800)))
	# Single File Prep
	f = file1
	im = file1[0]
	# print(im)

#print(im)
# aia = sunpy.map.Map(f'{path}{f}')
# aiad = (aiaprep.aiaprep(aia).data)[1148:2948,0:4096]
# aia = map2data(f)
# aiad = aia.data
# aiamgn = mgncv.mgn(np.float32(aiad),truncate = 5)
# Triple File Prep
	# print(im)
	for img in im:
		# print(img)
		year = img[7:11]
		mon = img[11:13]
		day = img[13:15]
		date1 = f'{year}_{mon}_{day}'
		# print(year,mon,day)
		# print(len(img))
		# date = img[14:24]
	#print(img)
	# aia = sunpy.map.Map(f'{path}{img}')
	# x0,y0 =85,545
		# aia  = aiaprep.aiaprep(f'{path}{img}')
		aia = sunpy.map.Map(f'{path}{img}')
		# plt.imshow(aia)
		# plt.show()
		aiad = aia.data
		aiamgn = mgncv.mgn(np.float32(aiad),truncate = 2)
		# plt.imshow(aiamgn)
		# plt.show()

	#print(date)
	# aiamgn /= len(im)
	# aiamgn = cv.medianBlur(np.float32(aiamgn),3)
	# demomgn = copy.deepcopy(aiamgn)
	# plt.imshow(aiamgn,cmap='sdoaia171')
	# plt.plot((900,x0),(2048,y0))
	# plt.show()
	# demomgn = copy.deepcopy(aiamgn) #Used for demonstration of mgn - for plotting loop trace on
	fwhm = 3
	loops = []
	cuts = []
	# plt.imshow(aiamgn)
	#plt.show()
	#BACKGROUND SUPPRESSION & SMOOTHING / CROPPING
	# header = {}
	x, y = np.meshgrid(*[np.arange(v.value) for v in aia.dimensions]) * u.pixel
	hpc_coords = aia.pixel_to_world(x, y)
	r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / aia.rsun_obs
	# plt.imshow(r)
	# plt.show()
	# r = r[0:4096,1148:2948]
	lowerr = 1.020
	rdiff  = 0.003
	inmask = np.where(r < lowerr)
	outmask = np.where(r >=lowerr + 0.1)
	kwid = 100
	lw1 = int(2*kwid + 0.5)
	gkern = cv.getGaussianKernel(lw1,kwid)
	
	mgnorig = copy.deepcopy(aiamgn)
	aiamgn = cv.medianBlur(np.float32(aiamgn),3)
	aiamgn= (aiamgn - np.min(aiamgn))*255/(np.max(aiamgn)-np.min(aiamgn))
	aiamgn -= cv.sepFilter2D(np.float32(aiamgn),ddepth=-1,kernelY=gkern,kernelX=gkern,borderType=1)
	aiamgn -= cv.sepFilter2D(np.float32(aiamgn),ddepth=-1,kernelY=gkern,kernelX=gkern,borderType=1)
	
	demomgn = copy.deepcopy(aiamgn)
	

	# aiamgn[np.where(aiamgn < 0)] /= -aiamgn.min()
	# demomgn[np.where(demomgn < 0)] = -1

	# plt.imshow(mgnorig)
	# plt.show()

		
	aiamgn[inmask] = -2
	aiamgn[outmask] = -2
	demomgn[outmask] = -2
	demomgn[inmask] = -2
	# plt.imshow(demomgn/np.max(demomgn))
	# plt.show()
	# aiamgn[0:10] = -3
	# aiamgn[1810:1820] = -3
	aiad[inmask] = -2
	aiad[outmask] = -2
	aiamgn[0:10] = -2
	aiamgn[-1:-10] = -2
	circle = np.where(np.logical_and(r > lowerr,r <= (lowerr +rdiff)))
	circalib = copy.deepcopy(aiamgn)
	circords = np.where(r > (lowerr + rdiff))
	circalib[circords] = -1
	circalib /= circalib.max()
	# brights = np.histogram(aiamgn[circle])
	demomgn[np.where(demomgn > 0)] /= demomgn[circle].max()
	demomgn[np.where(demomgn < 0)] = -1
	aiamgn[np.where(aiamgn > 0)] /= aiamgn[circle].max()
	aiamgn[aiamgn > 1] = 1
	try:
		mins = np.min(circalib)
	except ValueError:
		print("Min array issue. Possible bad frame, quitting run.")
		return
	# demomgn2 = copy.deepcopy(aiamgn)

	# plt.imshow(demomgn)
	# plt.show()
	# print(circle)
	# aiamgn = aiamgn[1148:2948,0:4096]
	# aiad = aiad[1148:2948,0:4096]
	# aiamgn[circle] *= 10
	# plt.imshow(aiamgn)
	# plt.show()
	#INIT LOOP TRACE (RIDGE TRACING)
	# sk = np.arange(0,100,1)
	# upline = np.arange(0,80,1)
	# rmin = 30
	loops = []
	loopsx,loopsy=[],[]
	# avoidx,avoidy = [],[]
	#flux = aiamgn[circle]

	zeroflux = np.max(circalib)
	# print(zeroflux)
	# plt.imshow(circalib)
	# plt.show()
	# brights = np.histogram(circalib)
	# plt.hist(circalib[circalib>0],bins = 50)
	# plt.show()
	# posxy = np.where(circalib > 0)
	# angls = np.vectorize(angcalc)
	# radxy = angls(posxy[1],posxy[0])
	# ncords = np.where(radxy > 0)
	# scords = np.where(radxy > 0)
	# d = {'rads':radxy,'intensity':circalib[posxy]}
	# df = pd.DataFrame(data=d)
	# df = df.sort_values('rads')
	# df1 = df[df.rads > 0]
	# df2 = df[df.rads <= 0]
	# plt.plot(np.array(df1.rads),np.array(df1.intensity))
	# plt.plot(np.array(df2.rads),np.array(df2.intensity))
	# plt.scatter(radxy[norcords],(circalib[posxy])[norcords],s=3)
	# plt.plot(radxy[scords],(circalib[posxy])[scords])
	# plt.plot(radxy[ncords],(circalib[posxy])[ncords])

	# plt.show()
	coords = np.where(circalib == zeroflux)	
	if coords[1].size <= 0:
		return
	# print(coords)
	x0 = coords[1][0]
	y0 = coords[0][0]
	# break1=break2
	# plt.imshow(aiamgn)
	# break1=break4
	# if len(x0) > 1:
	# 	x0 = coords[0][0]
	# if len(y0) > 1:
	# 	y0 = coords[1][0]
	loop = 0
	# base = np.average(fparam)
	base = zeroflux*0.55
	#print(x0,y0)
	# plt.imshow(aiamgn)
	# plt.plot(y0,x0)
	# plt.show()
	points = []

	while (zeroflux > base):
		loop += 1
		# ang,alpha2 = stemfitter(x0,y0,aiamgn)
		alpha2 = np.arctan2(y0-512,x0-512) - np.pi/2 #ARCTAN2 TAKES ARGUMENTS OF Y, X
		if alpha2 < 0:
			alpha2 += 2*np.pi
		# print(np.degrees(alpha2))
		# plt.imshow(aiamgn)
		# plt.scatter(x0,y0,marker = 'X')
		# plt.show()
		# if aiamgn[endcord] == -2:
		xgridl = np.arange(-10,+10,1)
		ygridl = np.arange(-10,+10,1)
		x, y = np.meshgrid(xgridl,ygridl)
		fill = []
		xadj = x + x0
		yadj = y + y0
		# if len(xstep) > 25:		
			# if aiamgn[endcord] == -2:
		# for point in range(len(xstep)):
		r2 = np.sqrt(x**2 + y**2)
		fill = np.where(r2 <= 10)
		#print(fill)
		x3 = xadj[fill]
		y3 = yadj[fill]
		# x3[x3>=1800] = 1799 ###
		aiamgn[(y3,x3)] = -1
		circalib[(y3,x3)] = -1
		# aiamgn[(y0,x0)] = -1
		# circalib[(y0,x0)] = -1
		flux = circalib[circle]
		coords = (np.nonzero(circalib == np.max(flux)))
		zeroflux = np.max(flux)
		# plt.scatter(coords)
		brights.append(aiad[y0,x0])
		points.append([x0,y0])
		baseang.append(alpha2)
		x0 = coords[1][0]
		y0 = coords[0][0]

	# widths = np.array(widths)

	# fig,ax = plt.subplots(facecolor='black')
	# print(len(widths))
	# print(loops)
	aiamgn[aiamgn<= 0 ] = -2
	# plt.imshow(mgnorig,cmap = 'sdoaia171')
	# print(loopsx)
	# print(loopsy)
	# print(points)
	# for x in range(len(points)):
		# print(points[x])
		# plt.scatter(points[x][0],points[x][1],linewidth = 1, color = 'r',marker='X')
	# for p in range(len(cuts)):
	# 	plt.plot(cuts[p][0],cuts[p][1],linewidth = 2, color = 'red',linestyle = '--')
	# plt.show()
	# plt.savefig(f'{len(im)}-{kwid}.png')
	# print(im[:24])
	# print(widths)
	# print(im[:24])
	#print(f'{len(widths)} - {img}')
	del aiamgn
	del aiad
	del aia
	# print(len(baseang))
	# plt.hist(np.degrees(baseang))
	# plt.show()
	# print(baseang)
	if len(baseang) > 0:
		# print(len(widths), print(date1))
		return (np.array(date1), np.degrees(baseang), brights)

angstroms = ["195","171","284","304"]
#angstroms = ["304"]

for angstrom in angstroms:
	path = f"./{angstrom}/lev1/"
	files = os.listdir(path)
	# files = glob.glob('./171eitsample/*.fits')
	# print(files)
	no = 0
	dates = []
	files.sort()
	# print(files)
	for d in files:
		date = d[8:16]
		# print(date)
		dates.append(date)
	dates = np.unique(dates)
	combi1 = []
	for img in dates:
		p1 = []
		for f in files:
			if (f[8:16] == img):
				p1.append(f)
		# while len(p1) < 3:
		# p1.append(p1[0])
		no +=1
		combi1.append((p1,no))
	# combi = combi1[1:]
	#files = files[1:]
	# print(combi)
	if __name__ == '__main__':
		with Pool(8) as p:
			widths1 = list(tqdm.tqdm(p.imap(widthmeasure,combi1),total = len(combi1)))
		f=open(f'EIT{angstrom}TestdistI.txt','w+')
		f.write('date,angle,flux' + '\n')
		for w in widths1:
			try:
				d,an,fl = w[0],w[1],w[2]
				for g in range(len(an)):
					if fl[g] > 0:
						f.write(f'{d},{an[g]},{fl[g]}' + '\n')
			except TypeError:
				print("Type Error, bad frame")				
			
		print('Done!')
		# widths1.reshape(-1,1)
		# print(widths1)
		
# 	#return (widths, np.array(date), np.degrees(basepoints), uncertainty)
# else:
# 	print("No width values, quitting run.")
#combi1 = combi1[1190:]
#angstroms["171","193","211","304"]
#angstroms = ["171"]

#angstroms = ["211","304"]
#angstroms = ["304"]

	#combi1 = combi1[550:]
		#widths1 = list(tqdm.tqdm(p.imap(widthmeasure,files),total = len(files)))
		
#loo.append(w)


