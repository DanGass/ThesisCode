#An algorithm to trace coronal loops in the limb region using both unprocessed and MGN processed SOHO EIT images. Based on OCCULT-2 produced by Dr. Markus Aschwanden & Dr Peter Hardi and Multi-Gaussian-Normalization produced by Dr. Huw Morgan and Dr Miloslav Druckmuller
import numpy as np
#import numpy.ma as ma
import sunpy.map
#import sunpy.io
#import sunpy.image
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
import astropy.units as u
import copy
#from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
#from scipy import signal
#import scipy.ndimage
from scipy.optimize import curve_fit as cf
import math
import matplotlib.pyplot as plt
import os,sys
from multiprocessing import Pool
import tqdm
import aiaprep
import mgncv
import cv2 as cv
from numba import njit,jit
import warnings
import urllib.error as er
import time
warnings.simplefilter('ignore')

def cubic(t,a,b,c,d):
	return a*pow(t,3) + b*pow(t,2) + c*t + d
def gauss(t,a,b,c):
	return a*np.exp(-np.power(t-b,2.)/(2*np.power(c,2)))
@njit
def neighbours(im,x,y):
	return[im[x-1:x+1,y-1:y+1]]

def map2data(imagename):
	return sunpy.map.Map(f'{path}{f}')

@jit
def stemfitter(x0,y0,aiamgn):
	ang = 0
	# xa = []
	# ya = []
	maxflux = 0.0
	upline = np.arange(0,80,1)
	alpha2 = np.arctan2(x0-900, y0-2048) - 0.5*math.pi
	# alpha3 = np.float64(alpha2)
	for angle in range(360):
		flux = 0.0
		angle -=180
		an = np.radians(angle/3)
		a = an - alpha2
		for i in upline:
			xa=min(1799,(x0 + np.cos(a)*i))
			ya=min((y0 + np.sin(a)*i),4097)
			flux+=aiamgn[int(xa),int(ya)]
		if (flux > maxflux):
			ang = a
			maxflux = flux
	return ang,alpha2
@jit
def lparamfit(ang,x0,y0,aiamgn,loop):
	# sk = np.arange(0,100,1)
	dire2,beta1,xm1,ym1,r2,r3 = 1,0,0,0,20,20
	rmin = 20
	maxflux = 0	
	for u2 in range(2):
		dire = 1
		if u2 == 1:
			dire = -1
		for g in range(2):
			beta = ang + math.pi * 0.5
			if g == 1:
				beta = ang - 0.5*math.pi
			for ra in range(1200): ### (?)
				# print(flux,maxflux)
				r2 = ra + rmin
				xc = x0 + rmin*math.cos(beta)
				yc = y0 + rmin*math.sin(beta) #y/xm should be a scalar here
				xm = x0 + (xc - x0)*(r2)/rmin 
				ym = y0 + (yc - y0)*(r2)/rmin
				flux = 0
				for sk in range(0,100):
					betam = beta + dire *(sk/r2) 
					xkm = xm - r2*math.cos(betam)
					ykm = ym - r2*math.sin(betam) #x/ykm should be array, due to betam
			# xkm = min(int(xkm),1799)
			# xkm = xkm.astype(int)
			# ykm = ykm.astype(int)
				# for c in range(len(xkm)):
					flux+=aiamgn[int(min(xkm,1799)),int(min(ykm,4095))]
	# print(xkm.astype(int),ykm.astype(int))
				if flux > maxflux:
					dire2 = dire
					beta1 = beta
					ym1 = ym #ym1 and xm1 should be scalar
					xm1 = xm
					maxflux = flux
	# print(maxflux,xm1,ym1)
					r3 = r2
	# print(dire2,beta1,xm1,ym1,r2,r3)
	return(dire2,beta1,xm1,ym1,r2,r3)

	# return(maxflux)
@jit
def looptrace(rmin,x0,y0,beta1,r2,dire2,aiamgn):
	rmin = 20.0
	xstep = [x0]
	ystep = [y0]
	step = 0
	negatives = 0
	total = 0
	while ((negatives < 8) and (total < 16)):
		xc = x0 + rmin*np.cos(beta1) #xc/yc is an array because of beta1
		yc = y0 + rmin*np.sin(beta1)
		xm = x0 + (xc - x0)*(r2)/rmin 
		ym = y0 + (yc - y0)*(r2)/rmin
		beta2 = beta1 + dire2 *(step/r2)
		# print(beta2)
		xkm = xm - r2*np.cos(beta2)
		ykm = ym - r2*np.sin(beta2) #This should be an array.
		xkm = min(xkm, 1799)
		ykm = min(ykm, 4095)
		flux = aiamgn[int(xkm),int(ykm)]
		if flux > 0:
			negatives = 0
		else:
			negatives += 1
			total += 1
		step = step + 1	
		xstep.append(xkm)
		ystep.append(ykm)
	return(xstep,ystep)
# @njit
# def bgsub(xstep,ystep,xm1,ym1,aiad):
# 	width = np.arange(0,15,1) - 7
# 	trims = np.zeros(15)
# 	#fig,ax = plt.subplots(facecolor='black')
# 	cutsalt = np.empty((8,2,15))
# 	for leng in range(8):
# 		fluxw = np.empty(15)
# 		xmid = round((leng+1)*0.1*len(xstep))
# 		ymid = round((leng+1)*0.1*len(ystep))
# 		xw1 = width + xstep[xmid]
# 		grad = (ym1 - ystep[ymid])/(xm1 - xstep[xmid])
# 		b = (xm1*ystep[ymid] - xstep[xmid]*ym1)/(xm1 - xstep[xmid])
# 		yw1 = xw1*grad + b
# 		xw1[xw1 >= 1800] = 1799 ###
# 		yw1[yw1 >= 4096] = 4095 ###
# 		yw1[yw1 <= 0] = 1
# 		# fluxw = 0
# 		absdiff = np.sqrt((np.max(yw1) - np.min(yw1))**2 + (np.max(xw1) - np.min(xw1))**2)
# 		if absdiff > 14:
# 			rat = absdiff/14
# 			yw1 = ystep[ymid] + (yw1-ystep[ymid])/rat
# 		for xw in range(len(xw1)):
# 			fluxw[xw] = aiad[int(xw1[xw]),int(yw1[xw])]
# 			cutsalt[leng][0][xw] = (xw1[xw])
# 			cutsalt[leng][1][xw] = (yw1[xw])
			
# 		trims = fluxw + trims
# 	# print(cutsalt)
# 	return (trims,cutsalt)
		
		
# 	trims = trims/8
# 	trims2 = np.copy(trims)
# 	newwidth = np.arange(0,15,0.1) - 7
# 	newdist = np.interp(newwidth,width,trims)
# 	# troughs = []
# 	troughs = (np.where(newwidth == newwidth[0])[0])
# 	for a in range(len(newdist)-2):
# 		b = a + 1
# 		if ((newdist[b] < newdist[b+1]) and (newdist[b] < newdist[b-1])):
# 			plce = np.where(newdist == newdist[b])[0]
# 			troughs = np.append(troughs,plce)
# 	plce2 =  np.where(newwidth == newwidth)
# 	troughs = np.append(troughs,(np.where(newwidth == newwidth[-1])[0]))
# 	troughs=np.array(troughs)
# 	troughs = np.reshape(troughs, (-1,1))
# 	if troughs[0].shape == (1,):
# 		troughs2 = []
# 		troughs = np.reshape(troughs, (-1,1))
# 		for div in troughs:
# 			for point in div:
# 				troughs2.append(point)
# 		troughs2 = np.reshape(troughs2, (1,-1))
# 		troughs = troughs2[0]
# 	troughs = np.hstack(troughs)
# 	bg = np.interp(newwidth,newwidth[troughs],newdist[troughs])
# 	newdist = newdist - bg
# 	return(trims2,cutsalt,newdist)
#INIT ARRAY & VARIABLES

def widthmeasure(file1):
	#print(file1)
	basepoints = []
	brights = []
	widths = []
	uncertainty = []
	aiamgn = (np.zeros((1800,4096)))
	# Single File Prep
	f = file1
	im = file1[0]
	
	#print(im)
	# aia = sunpy.map.Map(f'{path}{f}')
	# aiad = (aiaprep.aiaprep(aia).data)[1148:2948,0:4096]
	# aia = map2data(f)
	# aiad = aia.data
	# aiamgn = mgncv.mgn(np.float32(aiad),truncate = 5)
	# Triple File Prep
	
	for img in im:
		#print(img)
		date = img[14:24]
		#print(img)
		aia = sunpy.map.Map(f'{path}{img}')
		try:
			aiad = (aiaprep.aiaprep(aia).data)[1148:2948,0:4096]
		except ValueError:
			print("Error loading map, exiting run.")
			return
		except er.URLError:
			print('URL Error - waiting ...')
			time.sleep(10)
			aiad = (aiaprep.aiaprep(aia).data)[1148:2948,0:4096]
		except TypeError:
			print("Type Error, check map for irregularities")
			print(aia)
			return
		except OSError:
			print("AAAH")
			print(aia)
			return
		
		aiamgn += mgncv.mgn(np.float32(aiad),truncate = 5)
	#print(date)
	aiamgn /= len(im)
	aiamgn = cv.medianBlur(np.float32(aiamgn),3)
	# demomgn = copy.deepcopy(aiamgn) #Used for demonstration of mgn - for plotting loop trace on
	fwhm = 0
	loops = []
	cuts = []

	#BACKGROUND SUPPRESSION & SMOOTHING / CROPPING
	# header = {}
	nsm = 3.0
	x, y = np.meshgrid(*[np.arange(v.value) for v in aia.dimensions]) * u.pixel
	hpc_coords = aia.pixel_to_world(x, y)
	r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / aia.rsun_obs
	r = r[1148:2948,0:4096]
	r1 = (r - 1.05) **2
	# print(r[0])
	innerx = np.where(r1 == np.min(r1))
	innerr = np.sqrt((innerx[0] - 900)**2 + (innerx[1]-2048)**2)

	if (len(innerr) > 1):
		innerr = innerr[0]
	# print(innerr) 
	inmask = np.where(r < 1.05)
	outmask = np.where(r >=1.10)
	kwid = 80
	lw1 = int(2*kwid + 0.5)
	gkern = cv.getGaussianKernel(lw1,kwid)
	# aiamgn= (aiamgn - np.min(aiamgn))*255/(np.max(aiamgn)-np.min(aiamgn))
	aiamgn -= cv.sepFilter2D(np.float32(aiamgn),ddepth=-1,kernelY=gkern,kernelX=gkern,borderType=1)
	# aiamgn = cv.medianBlur(np.float32(aiamgn),3) + 0.1
	
	aiamgn[np.where(aiamgn < 0)] /= -aiamgn.min()
	# plt.imshow(aiamgn)
	# plt.show()
	# try:
	# 	mins = np.min(params)
	# except ValueError:
	# 	print("Min array issue. Possible bad frame, quitting run.")
	# 	return
		
	aiamgn[inmask] = -2
	aiamgn[outmask] = -2
	# aiamgn[0:10] = -3
	# aiamgn[1810:1820] = -3
	aiad[inmask] = -2
	aiad[outmask] = -2
	aiamgn[0:10] = -2
	aiamgn[-1:-10] = -2
	
	circle = np.where(np.logical_and(r > 1.05,r <= 1.053))
	aiamgn[np.where(aiamgn > 0)] /= aiamgn[circle].max()
	aiamgn[aiamgn > 1] = 1
	
	# print(circle)
	# aiamgn = aiamgn[1148:2948,0:4096]
	# aiad = aiad[1148:2948,0:4096]
	# aiamgn[circle] *= 10
	# plt.imshow(aiamgn)
	# plt.show()
	#INIT LOOP TRACE (RIDGE TRACING)
	# sk = np.arange(0,100,1)
	# upline = np.arange(0,80,1)
	rmin = 20
	loops = []
	avoidx,avoidy = [],[]
	flux = aiamgn[circle]
	zeroflux = max(flux)
	coords = np.where(aiamgn == zeroflux)
	# print(coords)
	x0 = coords[0]
	y0 = coords[1]
	if len(x0) > 1:
		x0 = coords[0][0]
	if len(y0) > 1:
		y0 = coords[1][0]
	loop = 0
	# base = np.average(fparam)
	base = zeroflux*0.333
	#print(x0,y0)
	# plt.imshow(aiamgn)
	# plt.plot(y0,x0)
	# plt.show()
	
	while (zeroflux > base): #This runs until optimal initial loop segment determined.
	#	print(loop,zeroflux,base)
		# print(x0,y0)
		loop += 1
		ang,alpha2 = stemfitter(x0,y0,aiamgn)
		# if loop >= 286:
		# 	print(ang,alpha2)
		#INITIAL LOOP FIT 
		dire2,beta1,xm1,ym1,r2,r3= lparamfit(ang,x0,y0,aiamgn,loop)
		# if loop >= 286:
			# print(dire2,beta1,xm1,ym1,r2,r3)

		#TRACING OUTWARDS

		xstep,ystep = looptrace(rmin,x0,y0,beta1,r2,dire2,aiamgn)
		#endcord = (int(xstep[-1]),int(ystep[-1]))
		# break1 = break2
		if len(xstep) > 25:
			endcord = (int(xstep[-1]),int(ystep[-1]))
			# print("!!!")
			if aiamgn[endcord] == -2:

				# loops.append((xstep,ystep))
				# avoidx = np.append(avoidx,xstep) 
				# avoidy = np.append(avoidy,ystep)
				# width = np.arange(0,15,1) - 7
				# #WIDTH COLLECTION - THIS SECTION IS A MESS, SHOULD REWRITE AT SOME POINT
				width = np.arange(0,15,1) - 7
				trims = np.zeros(15)
				#fig,ax = plt.subplots(facecolor='black')
				for leng in range(8):
					xmid = round((leng+1)*0.1*len(xstep))
					ymid = round((leng+1)*0.1*len(ystep))
					xw1 = width + xstep[xmid]
					grad = (ym1 - ystep[ymid])/(xm1 - xstep[xmid])
					b = (xm1*ystep[ymid] - xstep[xmid]*ym1)/(xm1 - xstep[xmid])
					yw1 = xw1*grad + b
					xw1[xw1 >= 1800] = 1799 ###
					yw1[yw1 >= 4096] = 4095 ###
					yw1[yw1 <= 0] = 1
					absdiff = np.sqrt((np.max(yw1) - np.min(yw1))**2 + (np.max(xw1) - np.min(xw1))**2)
					if absdiff > 14:
						rat = absdiff/14
						yw1 = ystep[ymid] + (yw1-ystep[ymid])/rat
					fluxw = aiad[(xw1.astype(int),yw1.astype(int))]
					trims = fluxw + trims
					cuts.append((xw1,yw1))
				# print(cuts)
				
				# print(cuts)
				# trims,cutsalt = bgsub(xstep,ystep,xm1,ym1,aiad)
				# print(cutsalt)
				# cuts = cuts.append(cutsalt[1:].astype(int))
				trims = trims/8
				trims2 = copy.deepcopy(trims)
				newwidth = np.arange(0,15,0.1) - 7
				newdist = np.interp(newwidth,width,trims)
				troughs = []
				troughs.append(np.where(newwidth == newwidth[0])[0])
				for a in range(len(newdist)-2):
					b = a + 1
					if ((newdist[b] < newdist[b+1]) and (newdist[b] < newdist[b-1])):
						troughs.append(np.where(newdist == newdist[b])[0])
				troughs.append(np.where(newwidth == newwidth[-1])[0])
				troughs=np.array(troughs)
				troughs = np.reshape(troughs, (-1,1))
				if troughs[0].shape == (1,):
					troughs2 = []
					troughs = np.reshape(troughs, (-1,1))
					for div in troughs:
						for point in div:
							troughs2.append(point)
					troughs2 = np.reshape(troughs2, (1,-1))
					troughs = troughs2[0]
				troughs = np.hstack(troughs)
				bg = np.interp(newwidth,newwidth[troughs],newdist[troughs])
				newdist = newdist - bg
				#plt.plot(newwidth[troughs],newdist[troughs],marker = 'o',linestyle = 'None',color = 'blue')
				# plt.plot(newwidth,fit_cubic,'--',color = 'red')
				#plt.plot(newwidth,newdist,marker = 'o',linestyle = '--', color = 'orange')
				#plt.plot(newwidth,bg,'--')
				#plt.show()
				# trims2,cutsalt,newdist = bgsub(xstep,ystep,xm1,ym1,aiad)
				# cuts = cuts.append(cutsalt[1:].astype(int))
				# newdist = newdist - fit_cubic
				troughs = []
				peaks = []
				
				for a in range(len(newdist)-2):
					b = a + 1
					if ((newdist[b] < newdist[b+1]) and (newdist[b] < newdist[b-1])):
						troughs.append(b)
					if ((newdist[b] > newdist[b+1]) and (newdist[b] > newdist[b-1])):
						peaks.append(b)
				troughs = np.reshape(troughs, (-1,1))	
				troughs = troughs - 70
				postroughs = troughs[troughs > 0]
				negtroughs = troughs[troughs <= 0]
				fwhm = 0
				if ((len(negtroughs) != 0) and (len(postroughs) != 0)):
					if (len(negtroughs) == 0):
						negtrough = np.sort(postroughs)[0]
						postrough = np.sort(postroughs)[1]
					if (len(postroughs) == 0):
						negtrough = np.sort(negtroughs)[-1]
						postrough = np.sort(negtroughs)[-2]
					if ((len(negtroughs) > 0) and (len(postroughs)>0)):
						postrough = np.sort(postroughs)[0]
						negtrough = np.sort(negtroughs)[-1]
						postrough += 70
						negtrough += 70
					for b in peaks:
						b -= 70
						postrough -= 70
						negtrough -=70 
						if ((b > negtrough) and (b < postrough)):
							b += 70
							peak = b
						postrough += 70
						negtrough += 70
					# vals = max(newdist[postrough],newdist[negtrough])
					#ax.set_facecolor('black')
					
					adjdist = newdist[np.array((width+7)*10)]

					
					peakrange = np.arange(negtrough,postrough+10)
					peakrange = (peakrange/10).astype(int)
					peakrange = np.unique(peakrange)
					# peakrange2 = np.arange(peakrange[0],peakrange[-1],0.1)
					#plt.plot(width,adjdist, linestyle="None", marker = 'o',color = 'orange')
					#plt.plot(width[peakrange],adjdist[peakrange],marker='o',color='blue')
					#plt.show()
					#popt2,pcov = cf(cubic,width[peakrange]+7,adjdist[peakrange],method = 'lm')
					hg =np.max(adjdist[peakrange])
					hg = float(hg)
					hg = abs(hg)
					pg = np.mean(width[peakrange])
					wg = np.std(width[peakrange])
					wg = abs(wg)
					#print(hg,pg,wg)
					params = ("nan","nan","nan")
					errors = ("nan","nan","nan")

					#Trust Region Fits - Bounded
					popt3 = (0,0,0)
					try:
						popt3,pcov2 = cf(gauss,width[peakrange],adjdist[peakrange],p0 = [hg,pg,wg],bounds = [[0,-np.inf,0.2],[hg*1.2,np.inf,np.inf]])
					except RuntimeError:
						#print("Runtime Error. Fit not found.")
						pass
					except ValueError:
						#print("Value Error. Something wrong with bounds.")
						pass

					#Levenberg-Marquardt Fits - No Bounds
					#popt3,pcov2 = cf(gauss,newwidth[((peakrange2)).astype(int)],newdist[(peakrange2).astype(int)],p0 = [hg,pg,wg])
					if (popt3[0] != 0):
						# fit_gauss = gauss(peakrange2-7,*popt3)
						params = popt3
						errors = np.sqrt(np.diag(pcov2))
						widths.append(params[2])
						# print(params[2])
						uncertainty.append(errors[2])
						basepoints = np.append(basepoints, alpha2)
						brights = np.append(brights, max(trims2))
					
					#	newdist[newdist < 0] = 0
					#plt.plot((width+7),adjdist, marker = 'o', linestyle = '--')
					#plt.plot(newdist,fit_cubic,linestyle = '--')
					#plt.plot(peakrange2,fit_gauss,linestyle = '--')

				# plt.plot(newwidth[negtrough],newdist[negtrough],marker = 'o',linestyle = 'None')
				# plt.plot(newwidth[peaks],newdist[peaks],marker = 'o',linestyle = 'None')
				#plt.show()
		#LOOP EXCLUSION
		# if len(xstep) > 1:
		xran = ((max(xstep)+10) - (min(xstep)-10))
		# print(xran)
		yran = ((max(ystep)+2) - (min(ystep)-10)) 
		xgridl = np.arange(int(xran))
		ygridl = np.arange(int(yran))
		x, y = np.meshgrid(xgridl,ygridl)
		fill = []
		aiamgn[(x0,y0)] = -1
		xadj = x + int(min(xstep)-10)
		yadj = y + int(min(ystep)-10)
		if len(xstep) > 25:		
			if aiamgn[endcord] == -2:
				for point in range(len(xstep)):
					x2 = (xadj - xstep[point])
					y2 = (yadj - ystep[point])
					r2 = np.sqrt(x2**2 + y2**2)
					fill = np.where(r2 <= fwhm+1)
					#print(fill)
					x3 = xadj[fill]
					y3 = yadj[fill]
					x3[x3>=1800] = 1799 ###
					aiamgn[(x3,y3)] = -1
		flux = aiamgn[circle]
		coords = (np.nonzero(aiamgn == np.max(flux)))
		zeroflux = np.max(flux)
		x0 = coords[0][0]
		y0 = coords[1][0]

	widths = np.array(widths)
	#fig,ax = plt.subplots(facecolor='black')
	# plt.imshow(aiamgn,cmap = 'sdoaia171')
	# for x in range(len(loops)):
	# 	# fig.plot_coord(loops[x][1],loops[x][0],linewidth = 3, color = 'white')
	# 	plt.plot(loops[x][1],loops[x][0],linewidth = 3, color = 'white')
	# for p in range(len(cuts)):
	# 	# fig.plot_coord(cuts[p][1],cuts[p][0],linewidth = 2, color = 'red',linestyle = '--')
	# 	plt.plot(cuts[p][1],cuts[p][0],linewidth = 2, color = 'red',linestyle = '--')
	# plt.show()
	# print(im[:24])
	# print(widths)
	# print(im[:24])
	#print(f'{len(widths)} - {img}')
	del aiamgn
	del aiad
	del aia
	if len(widths) > 0:
		#print(len(widths))
		return (widths, np.array(date), np.degrees(basepoints), brights, uncertainty)
		#return (widths, np.array(date), np.degrees(basepoints), uncertainty)
	else:
		print("No width values, quitting run.")
#combi1 = combi1[1190:]
#angstroms["171","193","211","304"]
#angstroms = ["171"]

#angstroms = ["211","304"]
#angstroms = ["304"]
print("Wavelength =")
angstroms = input()
angstroms = [str(angstroms)]
for angstrom in angstroms:
	path = f"./{angstrom}/"
	files = os.listdir(path)
	filelist =[] 
	for name in files:
		end = name[-6:-1]
		start = name[9:12]
		year = name[14:18]
		#print(end,start,year)
		if ((str(end) == "1.fit") and (str(start) == angstrom) and (int(year) < 2021)): 
			filelist.append(name)
	filelist.sort()
	combi = []
	dates = []
	no = 0
	for d in filelist:
		date = d[14:24]
		dates.append(date)
	dates = np.unique(dates)
	#dates = dates.sort()
	combi1 = []
	for img in dates:
		p1 = []
		for f in filelist:
			if (f[14:24] == img) and (len(p1) < 3):
				p1.append(f)
		while len(p1) < 3:
			p1.append(p1[0])	
		no +=1
		combi1.append((p1,no))
	#combi1 = combi1[1100:]
	if __name__ == '__main__':
		with Pool(24) as p:
			widths1 = list(tqdm.tqdm(p.imap(widthmeasure,combi1),total = len(combi1)))
			#widths1 = list(tqdm.tqdm(p.imap(widthmeasure,files),total = len(files)))
	widths1 = np.array(widths1)
	#widths1 = widths1.reshape(-1)
	f = open(f'{angstrom}IVcloser.txt',"w+")
	f.write('width,date,angle,flux,error' + '\n')
	loo = []
	print(widths1[0])
	n=0
	for line in widths1:
		#print(line)
		n+=1
		if line is not None:
	 		#print(line.size)
	# 		#print(n)
	# 		print(line[1],line[0])
			wid = line[0]
			date = line[1]
			base = line[2]
			intensity = line[3]
			errors = line[4]
			#errors = line[3]
			
			for w in range(len(line[0])):
				e = 0
				try:
					f.write(str(2*wid[w])+',' + str(date) +  ',' + str(base[w]) + ',' + str(intensity[w])+ ',' + str(2*errors[w]) + '\n')
					#f.write(str(2*wid[w])+',' + str(date) +  ',' + str(base[w]) + ',' + str(2*errors[w]) + '\n')
				except IndexError:
					print(line)
					print(f'Format issue in results #{e}')
				e+=1	 		
#loo.append(w)


