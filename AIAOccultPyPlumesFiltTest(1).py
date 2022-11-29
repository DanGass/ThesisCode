#An algorithm to trace coronal loops in the limb region using both unprocessed and MGN processed SOHO EIT images. Based on OCCULT-2 produced by Dr. Markus Aschwanden & Dr Peter Hardi and Multi-Gaussian-Normalization produced by Dr. Huw Morgan and Dr Miloslav Druckmuller
import numpy as np
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
import astropy.units as u
import copy
#from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
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
import time
from numpy import unravel_index
import pandas as pd

def cubic(t,a,b,c,d):
	return a*pow(t,3) + b*pow(t,2) + c*t + d
def gauss(t,a,b,c):
	return a*np.exp(-np.power(t-b,2.)/(2*np.power(c,2)))

def angcalc(x,y):
	alph = np.arctan2(y-2048,x-900)
	alph = np.rad2deg(alph)
	#alph = (alph*180)/math.pi
	# alph = np.arctan2(y-2048,x-900)
	# if (alph >= 0):
	# 	alph = abs(alph-90)
	# if (alph <= -90):
	# 	alph += 90
	# if (alph <= 0):
	# 	alph = abs(alph - 90)
	return alph
	# return alph

@njit
def stemfitter(x0,y0,aiamgn):
	# ang = 0.0
	# xa = []
	# ya = []
	
	# flux = 0.0
	upline1 = np.arange(0,80,1) - 1
	upline2 = np.arange(0,80,1) 
	upline3 = np.arange(0,80,1) + 1
	# upliney = np.zeros_like(uplinex)
	# print(x0,y0)
	avg = 0.0
	alpha2 = np.arctan2(y0-2048, x0-900) - np.pi/2
	# print(alpha2)
#	if alpha2 < 0:
#		alpha2 += 2*np.pi
	rng = np.arange(-20,20,0.5)
	for uplinex in (upline1,upline2,upline3):
		maxflux = 0.0
		for angle in rng:
			flux2 = 0.0
			ang2 = (alpha2 - np.radians(angle))
			# print(ang2)
			cosalpha = math.cos(ang2)
			sinalpha = math.sin(ang2)
			newy = y0 + (uplinex)*cosalpha
			newx = x0 - (uplinex)*sinalpha
			# plt.plot(newx,newy)
			newx = newx.astype(np.int_)
			newy = newy.astype(np.int_)
			for p in range(len(newx)):
				flux2+=aiamgn[newy[p],newx[p]] #REVERSED X AND Y DUE TO ARRAY INDEXING ("Row" - Y First, then "Column" - X)
			if (flux2 > maxflux):
				ang3 = ang2
				maxflux = flux2
				newx1 = newx
				newy1 = newy
		avg += ang3
	ang =avg/3
			# print(ang2,flux2)
	# print(ang)
	# print(ang,alpha2)
	# plt.plot(newx1,newy1)
	return ang,alpha2

#@jit
@njit
def plumetrace(x0,y0,beta1,aiamgn):
	xstep = [x0]
	ystep = [y0]
	step = 0
	negatives = 0
	total = 0
	latest = 0
	# while ((negatives < 8) and (total < 16)):
	cosalpha1 = math.cos(beta1)
	sinalpha1 = math.sin(beta1)

	# plt.plot(newx,newy)
	while ((total < 30) and (latest > -2) and (xstep[-1] > 0) and (xstep[-1] < 1799)):
		# xm = x0 + step
		ykm = int(y0 + step*cosalpha1)
		xkm = int(x0 - step*sinalpha1)
		# print(xkm,ykm) 
		# xkm = int(min(xkm, 1799))
		# xkm = int(min(xkm, 4095))
		flux = aiamgn[ykm,xkm]
		latest = aiamgn[ykm,xkm]
		if flux > 0:
			negatives = 0
		else:
			negatives += 1
			total += 1
		step = step + 1	
		# print(xkm,ykm)
		xstep.append(xkm)
		ystep.append(ykm)
	return(xstep,ystep)

#INIT ARRAY & VARIABLES

def widthmeasure(file1):
	#print(file1)
	basepoints = []
	brights = []
	widths = []
	uncertainty = []
	aiamgn = (np.zeros((4096,1800)))
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

		# print(img)
		date = img[14:24]
	#print(img)
	# aia = sunpy.map.Map(f'{path}{img}')
	# x0,y0 =85,545
		aia  = aiaprep.aiaprep(f'{path}{img}')
		try: 
			aiad = aia.data[0:4096,1148:2948]
		except AttributeError:
			print(f"Something wrong with map - {date}")
			return
		
		aiamgn += mgncv.mgn(np.float32(aiad),truncate = 2)
		
		#plt.imshow(aiamgn)
		#plt.show()
	#print(date)
	aiamgn /= len(im)
	#print(len(im))
	# aiamgn = cv.medianBlur(np.float32(aiamgn),3)
	# demomgn = copy.deepcopy(aiamgn)
	# plt.imshow(aiamgn)
	# plt.plot((900,x0),(2048,y0))
	# plt.show()
	# demomgn = copy.deepcopy(aiamgn) #Used for demonstration of mgn - for plotting loop trace on
	fwhm = 10
	loops = []
	cuts = []
	# plt.imshow(aiamgn)
	#plt.show()
	#BACKGROUND SUPPRESSION & SMOOTHING / CROPPING
	# header = {}
	x, y = np.meshgrid(*[np.arange(v.value) for v in aia.dimensions]) * u.pixel
	hpc_coords = aia.pixel_to_world(x, y)
	r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / aia.rsun_obs

	r = r[0:4096,1148:2948]
	lowerr = 1.020
	rdiff  = 0.001
	inmask = np.where(r < lowerr)
	outmask = np.where(r >=lowerr + 0.05)
	kwid = 100
	lw1 = int(2*kwid + 0.5)
	gkern = cv.getGaussianKernel(lw1,kwid)
	
	mgnorig = copy.deepcopy(aiamgn)
	aiamgn= (aiamgn - np.min(aiamgn))*255/(np.max(aiamgn)-np.min(aiamgn))
	aiamgn -= cv.sepFilter2D(np.float32(aiamgn),ddepth=-1,kernelY=gkern,kernelX=gkern,borderType=1)


	aiamgn = cv.medianBlur(np.float32(aiamgn),3)
	demomgn = copy.deepcopy(aiamgn)

	# aiamgn[np.where(aiamgn < 0)] /= -aiamgn.min()
	# demomgn[np.where(demomgn < 0)] = -1

	# plt.imshow(mgnorig)
	# plt.show()

		
	aiamgn[inmask] = -2
	aiamgn[outmask] = -2
	demomgn[outmask] = -2
	demomgn[inmask] = -2
	# aiamgn[0:10] = -3
	# aiamgn[1810:1820] = -3
	aiad[inmask] = -2
	aiad[outmask] = -2
	aiamgn[0:10] = -2
	aiamgn[-1:-10] = -2

	#Testing for Directional Bias
	aiamgn = np.flipud(aiamgn)
	aiad = np.flipud(aiad)
	demomgn = np.flipud(demomgn)
	mgnorig = np.flipud(mgnorig)

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

	mins = np.min(circalib)
	if np.isnan(mins):
		
		print("mgnerror")
		return
	#if mins == float("nan"):
	plt.imshow(demomgn)
	plt.show()
	plt.imshow(circalib)
	plt.show()
			
	#except ValueError:
	#	print("Min array issue. Possible bad frame, quitting run.")
	#	return
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

	zeroflux = np.amax(circalib)
	# print(zeroflux)
	# plt.imshow(circalib)
	# plt.show()
	# brights = np.histogram(circalib)
	# plt.hist(circalib[circalib>0],bins = 50)
	# plt.show()
	posxy = np.where(circalib > 0)
	#angls = np.vectorize(angcalc)
	#radxy = angls(posxy[1],posxy[0])
	# ncords = np.where(radxy > 0)
	# scords = np.where(radxy > 0)
	#d = {'rads':radxy,'intensity':circalib[posxy]}
	#df = pd.DataFrame(data=d)
	#df = df.sort_values('rads')
	#df1 = df[df.rads > 0]
	#df2 = df[df.rads <= 0]
	# plt.plot(np.array(df1.rads),np.array(df1.intensity))
	# plt.plot(np.array(df2.rads),np.array(df2.intensity))
	# plt.scatter(radxy[norcords],(circalib[posxy])[norcords],s=3)
	# plt.plot(radxy[scords],(circalib[posxy])[scords])
	# plt.plot(radxy[ncords],(circalib[posxy])[ncords])

	# plt.show()
	coords = np.where(circalib == zeroflux)
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
	base = zeroflux*0.333
	#print(x0,y0)
	# plt.imshow(aiamgn)
	# plt.plot(y0,x0)
	# plt.show()

	while (zeroflux > base): #This runs until optimal initial loop segment determined.
		# print(loop,zeroflux,base,x0,y0)
		loop += 1
		ang,alpha2 = stemfitter(x0,y0,aiamgn)
		# if loop >= 286:
		# 	print(ang,alpha2)
		#INITIAL LOOP FIT 
		# dire2,beta1,xm1,ym1,r2,r3= plumetrace(ang,x0,y0,aiamgn,loop)
		# if loop >= 286:
			# print(dire2,beta1,xm1,ym1,r2,r3)

		#TRACING OUTWARDS

		
		xstep,ystep = plumetrace(x0,y0,ang,aiamgn)

		# plt.imshow(aiamgn)
		# plt.plot((900,x0),(2046,y0))
		
		
		
		
		# plt.show()
		endcord = (int(ystep[-1]),int(xstep[-1]))
		# break1 = break2
		if len(xstep) > 25:
			#print(xstep,ystep)
			endcord = (int(ystep[-1]),int(xstep[-1]))
			# print("!!!")
			if aiamgn[endcord] == -2:
				# plt.plot(xstep,ystep)
				loops.append((xstep,ystep))
				loopsx.append(xstep)
				loopsy.append(ystep)
				# width = np.arange(0,15,1) - 7
				# #WIDTH COLLECTION - THIS SECTION IS A MESS, SHOULD REWRITE AT SOME POINT
				width = np.arange(0,31,1) - 15
				trims = np.zeros(31)
				#fig,ax = plt.subplots(facecolor='black')
				for leng in range(8):
					xmid = round((leng+1)*0.1*len(xstep))
					ymid = round((leng+1)*0.1*len(ystep))
					cos2,sin2 = math.cos(ang+(math.pi)/2),math.sin(ang+(math.pi)/2)
					yw1 = ystep[ymid] + (width)*cos2
					xw1 = xstep[ymid] - (width)*sin2
					# grad = (ym1 - ystep[ymid])/(xm1 - xstep[xmid])
					# b = (xm1*ystep[ymid] - xstep[xmid]*ym1)/(xm1 - xstep[xmid])
					# yw1 = xw1*grad + b
					# plt.plot(xw1,yw1)
					xw1[xw1 >= 1800] = 1799 ###
					xw1[xw1 >= 4096] = 4095 ###
					# xw1[xw1 <= 0] = 1
					
					absdiff = np.sqrt((np.max(yw1) - np.min(yw1))**2 + (np.max(xw1) - np.min(xw1))**2)
					#if absdiff > 14:
					#	rat = absdiff/14
					#	yw1 = ystep[ymid] + (yw1-ystep[ymid])/rat
					fluxw = aiad[(yw1.astype(int),xw1.astype(int))]
					trims = fluxw + trims
					cuts.append((xw1,yw1))
				
				# print(cuts)
				
				# print(cuts)
				# trims,cutsalt = bgsub(xstep,ystep,xm1,ym1,aiad)
				# print(cutsalt)
				# cuts = cuts.append(cutsalt[1:].astype(int))
				trims = trims/8
				trims2 = copy.deepcopy(trims)
				newwidth = np.arange(0,31,0.1) - 15
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
				troughs = troughs - 150
				postroughs = troughs[troughs > 0]
				negtroughs = troughs[troughs <= 0]
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
						postrough += 150
						negtrough += 150
					for b in peaks:
						b -= 150
						postrough -= 150
						negtrough -= 150
						if ((b > negtrough) and (b < postrough)):
							b += 150
							peak = b
						postrough += 150
						negtrough += 150
					# vals = max(newdist[postrough],newdist[negtrough])
					#ax.set_facecolor('black')
					
					adjdist = newdist[np.array((width+15)*10)]

					
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
						#Translates angles from arctan2 format to normal 0-360 degrees format
						alpha2 *= -1
						if alpha2 < 0:
							alpha2 += 2*np.pi
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
		if aiamgn[endcord] == -2:
			xran = ((max(xstep)+10) - (min(xstep)-10))
			# print(xran)
			yran = ((max(ystep)+2) - (min(ystep)-10)) 
			xgridl = np.arange(int(xran))
			ygridl = np.arange(int(yran))
			x, y = np.meshgrid(xgridl,ygridl)
			fill = []
			xadj = x + int(min(xstep)-10)
			yadj = y + int(min(ystep)-10)
			# if len(xstep) > 25:		
				# if aiamgn[endcord] == -2:
			for point in range(len(xstep)):
				x2 = (xadj - xstep[point])
				y2 = (yadj - ystep[point])
				r2 = np.sqrt(x2**2 + y2**2)
				fill = np.where(r2 <= fwhm)
				#print(fill)
				x3 = xadj[fill]
				y3 = yadj[fill]
				x3[x3>=1800] = 1799 ###
				aiamgn[(y3,x3)] = -1
				circalib[(y3,x3)] = -1
		aiamgn[(y0,x0)] = -1
		circalib[(y0,x0)] = -1
		flux = circalib[circle]
		coords = (np.nonzero(circalib == np.max(flux)))
		zeroflux = np.max(flux)
		x0 = coords[1][0]
		y0 = coords[0][0]

	# widths = np.array(widths)

	#fig,ax = plt.subplots(facecolor='black')
	# print(len(widths))
	#plt.imshow(mgnorig,cmap = 'sdoaia171')
	# print(loopsx)
	# print(loopsy)
	#for x in range(len(loops)):
		# fig.plot_coord(loops[x][1],loops[x][0],linewidth = 3, color = 'white')
		#plt.plot(loops[x][0],loops[x][1],linewidth = 1, color = 'white')
	# for p in range(len(cuts)):
	# 	# fig.plot_coord(cuts[p][1],cuts[p][0],linewidth = 2, color = 'red',linestyle = '--')
		#plt.plot(cuts[x][0],cuts[x][1],linewidth = 2, color = 'red',linestyle = '--')
	plt.show()
	# plt.savefig(f'{len(im)}-{kwid}.png')
	# print(im[:24])
	# print(widths)
	# print(im[:24])
	print(f'{len(widths)} - {img}')
	del aiamgn
	del aiad
	del aia
	if len(widths) > 0:
	#	print(len(widths), date)
		return (widths, np.array(date), np.degrees(basepoints), brights, uncertainty)

#angstroms = ["193","211","304"]
angstroms = ["171"]

for angstrom in angstroms:
	path = f"./{angstrom}/"
	files = os.listdir(path)
	no = 0
	dates = []
	files.sort()
	# print(files)
	for d in files:
		date = d[14:24]
		dates.append(date)
	dates = np.unique(dates)
	combi1 = []
	for img in dates:
		p1 = []
		for f in files:
			if (f[14:24] == img):
				p1.append(f)
		# while len(p1) < 3:
		# p1.append(p1[0])
		no +=1
		combi1.append((p1,no))
	#combi1 = combi1[0:23]
	#files = files[1:]
	# print(combi)
	if __name__ == '__main__':
		with Pool(24) as p:
			widths1 = list(tqdm.tqdm(p.imap(widthmeasure,combi1),total = len(combi1)))
	f = open(f'{angstrom}PlumesV03Flipped.txt',"w+")
	f.write('width,date,angle,flux,error' + '\n')
	loo = []
	print(widths1[0])
	n=0
	for line in widths1:
		print(line)
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


