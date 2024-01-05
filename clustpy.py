#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
import astropy.io.fits as fits
from astropy.wcs import WCS
import matplotlib.colors as colors
from matplotlib.patches import Circle
from astropy.table import Table
from photutils.aperture import CircularAperture, CircularAnnulus
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture
from photutils.aperture import aperture_photometry
from photutils.aperture import ApertureStats
from photutils.utils import calc_total_error
from photutils.aperture import BoundingBox
from photutils.aperture import ApertureMask 
from scipy.ndimage import gaussian_filter

# things i need for desctibtions:
    # what the function does from a literal code sense
    # description of varibles: input and output
    # in some cases a descprition of what the function does for algo?

def info(cx,cy, R, data, section):
    # info(23.54,54.2, 20, full_data_file, True)
    # the use of this function is to create different datacuts of the main data centered on the cluster to help with computation power
    # function also returns the neccesary defining values of these image cuts 
    
    ## note that this function's outputs set up most of the values that the algorithm feeds into the other functions in this package for the algorithm to run.


    #function takes an image file and returns different cuts of that image and the cuts parameters dependent on that of input parameters 
    # The 'section' boolean determines what is returned
        # section = True: returns the x, y coordinates of the image cut and wide cut ('cutcen' & 'wide_cen') and the annulus object along with its radii ('ann' & 'rI' & 'rO')
        # section = False: returns the normal data cut determined by the guess radius ('datacut') and its blurred version ('datacutB') by its sigma ('SIGMA'), 
        #   the data cut with annulus area ('datacutWIDE')

    #inputs: 
        # 'cx' & 'cy' <float>, coordinates of the cluster 
        # 'R' <integer>, the inital guess of the clusters radius in pixels
        # 'data' <2d array>, the full data where the cluster is located in 
        # 'section' <Boolean>, varible determines what function returns 

    #outputs:
        # True:
        #   'cutcen', 'wide_cen', 'rI', 'rO' <floats>
        #   'ann' <photutil annulus object>
        # Flase: 
        #   'datacut', 'datacutWIDE', 'datacutB' <2d arrays>
        #   'SIGMA' <float>
 

    cen = cx, cy
    x1 = round(cx - R)
    x2 = round(cx + R)
    y1 = round(cy - R)
    y2 = round(cy + R)
    cutcen = (R, R)
    
    #annulus consideration 
        #boudning radii
    rI = R + 2
    rO = R + 12
        #coords
    wx1 = round(cx - rO)
    wx2 = round(cx + rO)
    wy1 = round(cy - rO)
    wy2 = round(cy + rO)
    wide_cen = (rO, rO)
    
    #creation of annulus and bkg_mean value
    ann = CircularAnnulus(wide_cen, r_in = rI, r_out = rO)
    
    #Blurring factor based on maxR
    #SIGMA = (R//20) * 2
    SIGMA = 2

    #datacut creation (note flipped y and x; .fits are weird af)
    datacut = data[y1:y2, x1:x2]
    datacutWIDE = data[wy1:wy2, wx1:wx2]
    datacutB = gaussian_filter(datacut, SIGMA)
    
    # Boolean value to determine what set of varibles are returned 
    if section == True:
        return cutcen, wide_cen, rI, rO, ann
    elif section == False:
        return datacut, datacutWIDE, datacutB, SIGMA

def stat(dW , ann):
    # stat(data_includes_annulus, annulus_object)

    # retruns the background mean ('bkg_mean' <float>) of the inputed annulus object given the data inputed
    #dW : image file including the annulus <2d array>
    #ann : corresponding annulus object <photutil aperture object>
    apstat = ApertureStats(dW, ann)
    bkg_mean = apstat.mean
    return bkg_mean

def boundingbox(R, data):
    # boundingbox(25, image_data)
    #using photutil bounding box, function takes in an image file and returns a set of 4 image file cut outs using the R varible as a center guide
    
    ## note that 'R' can be any value but is used in this case to center square image data ('data')

    #varibles:
        #input: 
            # R: <integer>, determines the center and size of where the cuts are made.
            # data: <2d array>, works best as square arrray

        #outputs:
            # M: list of mask objects from photutil <photutil mask objects>
            # D: list of data cutout <2d arrrays>
    bbox1 = BoundingBox(ixmin=R, ixmax=2*R, iymin=R, iymax=2*R)
    bboxAp1 = bbox1.to_aperture()
    mask1 = bboxAp1.to_mask(method='exact')
    data1 = mask1.cutout(data)
    
    bbox2 = BoundingBox(ixmin=0, ixmax=R, iymin=R, iymax=2*R)
    bboxAp2 = bbox2.to_aperture()
    mask2 = bboxAp2.to_mask(method='exact')
    data2 = mask2.cutout(data)
    
    bbox3 = BoundingBox(ixmin=0, ixmax=R, iymin=0, iymax=R)
    bboxAp3 = bbox3.to_aperture()
    mask3 = bboxAp3.to_mask(method='exact')
    data3 = mask3.cutout(data)
    
    bbox4 = BoundingBox(ixmin=R, ixmax=2*R, iymin=0, iymax=R)
    bboxAp4 = bbox4.to_aperture()
    mask4 = bboxAp4.to_mask(method='exact')
    data4 = mask4.cutout(data)
    
    M = [mask1, mask2, mask3, mask4]
    D = [data1, data2, data3, data4]
    
    return M, D

def thirds(R, odata):
    # thirds(32, image_data)
    # function calculates mean value of radial thirds of an inputed data array, returns true if mean values decreases from center of circle to edge

    #varibles: 
        # inputs:
            # R: <int object> unit: pixels, radius of the object in the data file
            # odata: <data array>, data from which center is (R,R)

        # return:
            # boolean value
    #set up of radius and each segement of the circle using photutil aperture and annulus class
    radii = np.linspace(0, R, 4)
    inner = CircularAperture((R,R), r = radii[1])
    middle = CircularAnnulus((R,R), r_in = radii[1], r_out = radii[2])
    outer = CircularAnnulus((R,R), r_in = radii[2], r_out = radii[3])
    
    #photutil ApertureStats is used to obtain mean values of thirds
    apstatI = ApertureStats(odata, inner)
    apstatM = ApertureStats(odata, middle)
    apstatO = ApertureStats(odata, outer)
    
    meanI = apstatI.mean
    meanM = apstatM.mean
    meanO = apstatO.mean    

    if meanI > meanM > meanO:
        return True
    else:
        return False
    #return meanI, meanM, meanO

def aperture(R, bkg, odata):
    # aperture(25, 0.45, image_data_cut)

    #function takes in a data cut of a cluster('odata') to a maximum radius value('R') and does iterative aperture photometry on the cluster starting from a radius of 0 up to 'R'
    #   this function then outputs the radius value where background subtreacted photometry is outside of the error bar of the raw counts photometry ('end') along with the clusters
    #       signal to noise ratio & total background of each aperture ('SNR' & 'TBG') along with the clusters half light radius ('HLR')

    #inputs:
        # 'R' <integer>, radius that the image data cut is centered on
        # 'bkg' <float>, the associtated background value for the image data inputted
        # 'odata' <2d array>, image data of cluster being tested (lengths = 'R')

    #outputs:
        # 'SNR' <array of floats>, array of the signal to noise ratios of the cluster
        # 'end' <integer>, the calculated endpoint of the cluster, where after this point the added counts insignifacnt / equal to the background value
        # 'TBG' <array of floats>, array of the total background of each aperture
        # 'HLR' <integer>, the half light radius of the cluster
    TBG = []
    SNR = []
    INT = []
    PHOT = []
    for i in range(R): #runs background sub intergrated and diff photometry
        
        #set up
        aperature = CircularAperture((R,R), r = (i+1))
        area = aperature.area
        errort = calc_total_error(odata, bkg, 1.55)
        
        
        #normal stuff
        Tbkg = bkg * area 
        TBG.append(Tbkg)
        photo = aperture_photometry(odata, aperature, error = errort)
        SnR = photo[0][3] / Tbkg
        SNR.append(SnR)
        photSbkg = photo[0][3] - Tbkg
        PHOT.append(photSbkg)
        Intensity = photSbkg / area
        INT.append(Intensity)

        if photSbkg >= (photo[0][3] - photo[0][4]):
            end = i + 1

    return SNR, end, TBG, PHOT

def half_light(R, bkg, bdata):
    INT = []
    for i in range(R): #runs background sub intergrated
        
        #set up
        aperature = CircularAperture((R,R), r = (i+1))
        area = aperature.area
        errort = calc_total_error(bdata, bkg, 1.55)
        
        #normal stuff
        Tbkg = bkg * area 
        photo = aperture_photometry(bdata, aperature, error = errort)
        photSbkg = photo[0][3] - Tbkg
        Intensity = photSbkg / area
        INT.append(Intensity)
        
    peaksI = find_peaks(INT)[0]
    if len(peaksI) == 0:
            HLR = 0
    else:
        prom_i = peak_prominences(INT, peaksI)[0]
        locPP1 = np.argmax(prom_i) #index of prom peak
        HLRs = peak_widths(INT,peaksI)[0]
        HLR = HLRs[locPP1]

    return INT, HLR

def quad_ann(data, ann, R):
    # quad_ann(image_datacut, annulus, 25)
    #function takes in an image data ('data'), phot util annulus object ('ann'), and 'R' int value(used to center the data).
    #outputs an array of background means of 4 sections of the inputted annulus object with values from 'data'

    #creating quardants of the data
    qwdata = boundingbox(R, data)[-1]
    
    #using photutils with 'qwdata' & 'ann' to get 4 seperate annulus regions
    apstat1 = ApertureStats(qwdata[0], ann)
    apstat2 = ApertureStats(qwdata[1], ann)
    apstat3 = ApertureStats(qwdata[2], ann)
    apstat4 = ApertureStats(qwdata[3], ann)
    
    #obtain each mean value of individual annulus region
    bkg1 = apstat1.mean
    bkg2 = apstat2.mean
    bkg3 = apstat3.mean
    bkg4 = apstat4.mean
    
    #compile into array
    Qbkg = [bkg1, bkg2, bkg3, bkg4]
    return Qbkg
    #returns a 4 size array of each individual bkg corresponding to each quadrant 

def quad_aperture(R, bkgs, qdata):
    # quad_aperture(25, array_backgrounds, array_quadrant_datacuts)
    #this is the same as aperture but for image data that is split into quadrants!
    #function takes in a 4 index array of image data ('qdata') with associated background values('bkgs') and the bounding radius ('R') 
    #function outputs a 4 index array consisting of varibles that are the radius, representing where the added counts of a larger aperture are not signigifacnt, for each quadrant

    #calculation of each error of each quadrant
    errort1 = calc_total_error(qdata[0], bkgs[0], 1.55)
    errort2 = calc_total_error(qdata[1], bkgs[1], 1.55)
    errort3 = calc_total_error(qdata[2], bkgs[2], 1.55)
    errort4 = calc_total_error(qdata[3], bkgs[3], 1.55)
    
    
    for i in range(R):
        #set up apertures for photometry
        aperature = CircularAperture((R,R), r = (i+1))
        q1Ap = CircularAperture((0,0), r = (i+1))
        q2Ap = CircularAperture((R,0), r = (i+1))
        #q3Ap can use normal aperture
        q4Ap = CircularAperture((0,R), r = (i+1))
        
        #set up background values used in aperture photometry
        area = aperature.area
        qarea = area/4
        bkgs1 = qarea * bkgs[0]
        bkgs2 = qarea * bkgs[1]
        bkgs3 = qarea * bkgs[2]
        bkgs4 = qarea * bkgs[3]
        
        #each quadrant goes through own background subtarcted photometry, 'end#' is calculated at the point where background leaves error bounds

        #quadrant 1
        photo1 = aperture_photometry(qdata[0], q1Ap, error = errort1)  #photometry <ip>
        SnR1 = photo1[0][3] / bkgs1
        Q1Sub = photo1[0][3] - bkgs1
        if Q1Sub >= (photo1[0][3] - photo1[0][4]):
            end1 = i + 1
        
        #quadrant 2
        photo2 = aperture_photometry(qdata[1], q2Ap, error = errort2)  #photometry <ip>
        SnR2 = photo2[0][3] / bkgs2
        Q2Sub = photo2[0][3] - bkgs2
        if Q2Sub >= (photo2[0][3] - photo2[0][4]):
            end2 = i + 1
            
        #quadrant 3
        photo3 = aperture_photometry(qdata[2], aperature, error = errort3)  #photometry <ip>
        SnR3 = photo3[0][3] / bkgs3
        Q3Sub = photo3[0][3] - bkgs3
        if Q3Sub >= (photo3[0][3] - photo3[0][4]):
            end3 = i + 1
            
        #quadrant 4
        photo4 = aperture_photometry(qdata[3], q4Ap, error = errort4)  #photometry <ip>
        SnR4 = photo4[0][3] / bkgs4
        Q4Sub = photo4[0][3] - bkgs4
        if Q4Sub >= (photo4[0][3] - photo4[0][4]):
            end4 = i + 1
        
    QAPh = [Q1Sub, Q2Sub, Q3Sub, Q4Sub]
    ends = [end1, end2, end3, end4]
    return ends

def quad_SNR(R, bkg, qdata):
    # quad_SNR(25, 0.32, array_quadrant_datacut)

    #function calculates the signal noise ratio for every quadrant of a given image
    #inputs: 'R' <integer> denotes the radius of the image inputed, used to created apertures for SNR calculation 
            # 'bkg' <float> background value associted with the image inputted
            # 'qdata' <array> of <2d arrays> image data split into quadrants
    #outputs: 'SNR1', 'SNR2', 'SNR3', 'SNR4' <array of float values>, Each 'SNR#' represents the SNR at each radius, of the respective quadrant


    qTBG = []
    SNR1 = []
    SNR2 = []
    SNR3 = []
    SNR4 = []
    for i in range(R): #runs background sub intergrated and diff photometry
        
        #set up
        aperature = CircularAperture((R,R), r = (i+1))
        q1Ap = CircularAperture((0,0), r = (i+1))
        q2Ap = CircularAperture((R,0), r = (i+1))
        #q3 can use normal aperture
        q4Ap = CircularAperture((0,R), r = (i+1))
        area = aperature.area
        qarea = area / 4
        qTbkg = bkg * qarea
        qTBG.append(qTbkg)

        #calculation of each error of each quadrant
        errort1 = calc_total_error(qdata[0], bkg, 1.55)
        errort2 = calc_total_error(qdata[1], bkg, 1.55)
        errort3 = calc_total_error(qdata[2], bkg, 1.55)
        errort4 = calc_total_error(qdata[3], bkg, 1.55)
        
        #quadrant 1
        photo1 = aperture_photometry(qdata[0], q1Ap, error = errort1)  #photometry <ip>
        SnR1 = photo1[0][3] / qTbkg
        SNR1.append(SnR1)
        #quadrant 2
        photo2 = aperture_photometry(qdata[1], q2Ap, error = errort2)  #photometry <ip>
        SnR2 = photo2[0][3] / qTbkg
        SNR2.append(SnR2)
        #quadrant 3
        photo3 = aperture_photometry(qdata[2], aperature, error = errort3)  #photometry <ip>
        SnR3 = photo3[0][3] / qTbkg
        SNR3.append(SnR3)
        #quadrant 4
        photo4 = aperture_photometry(qdata[3], q4Ap, error = errort4)  #photometry <ip>
        SnR4 = photo4[0][3] / qTbkg
        SNR4.append(SnR4)

    return SNR1, SNR2, SNR3, SNR4

def quad_check(Q):
    # quad_check(array_SNRs)
    # Function checks the signal to noise ratios of a tested clusters quadrants to see where the SNRs are significant to the clusters size.
    #  Q denotes the SNRS of all four quadrants tested (gotten from quad_SNR)

    ## note current version of algorithm does not use this function ## 

    Mean = []
    STD = []
    # find mean and std of quadrants with respect to radius
    for i in range(len(Q[1])):
        SNR_i = [Q[0][i], Q[1][i], Q[2][i], Q[3][i]]
        mean_i = np.mean(SNR_i)
        std_i = np.std(SNR_i)
        Mean.append(mean_i)
        STD.append(std_i)
    end1 = []
    end2 = []
    end3 = []
    end4 = []
    for k in range(len(Q)):
        
        for j in range(len(Mean)):
            testkj = Mean[j] - Q[k][j]
            if np.abs(testkj) > STD[j]:
                if k == 0:
                    end1.append(j)
                if k == 1:
                    end2.append(j)
                if k == 2:
                    end3.append(j)
                if k == 3:
                    end4.append(j)
    if len(end1) == 0:
        end1 = [0]
    if len(end2) == 0:
        end2 = [0]
    if len(end3) == 0:
        end3 = [0]
    if len(end4) == 0:
        end4 = [0]
    cappoint = [end1[0] + 1, end2[0] + 1, end3[0] + 1, end4[0] + 1]
    
    #returns 
    #   'Mean' <array>, array of the mean values of the 4 SNRs
    #   'STD' <array>, array of the standard deviation of the SNRs
    #   'cappoint' <array>, array of float values that denote the end of the siginficance of the repected SNR
    return Mean, STD, cappoint

def peak_find(data):
    # peak_find(array_SNRs)
    #function takes in the SNRs ('data') from each quadrant finds the most signifacant peak from each SNR along with the full width half max, then returns values associated with them.

    #inputs:
        # 'data' <array of arrays of float values>, SNR arrays of of the quadrants of the cluster.

    #outputs:
        # 'furthR' <integer>, The largest index of the FWHM of the SNR peaks, + 1 to have the value equate to the radius value
        # 'furthPeak' <integer>, The the largest index of the SNR peaks, + 1 to have the value equate to the radius value
        # 'quadloc' <integer>, integer that corresponds to what quadrant has 'furthR' in it. Quadrants orgainzed like mathmatical quadrants
        # 'PEAKS' <array of integers>, array of the peaks of the SNRs
        # 'endpoints' <array of integers>, array of the indexs of the FWHM of the SNR peaks + 1 


    endpoints = []
    PEAKS = []
    for i in range(len(data)):
        peak_i, _ = find_peaks(data[i])
        if len(peak_i) == 0:
            PEAKS.append(1)
            eR_i = 0
        else:
            #uses prominences to find best peak;
            prom_i = peak_prominences(data[i], peak_i)[0]
            locPP1 = np.argmax(prom_i) #index of prom peak
            
            if data[locPP1] >= data[0]:
                #finds all width maxes
                PEAKS.append(peak_i[locPP1] + 1)
                width_Mi = peak_widths(data[i], peak_i, rel_height=0.5)[-1]
                #uses most prominent peak index to determine 
                eR_i = width_Mi[locPP1]
            else:
                PEAKS.append(1)
                eR_i = 0
        endpoints.append(eR_i + 1)
    quadLoc = np.argmax(endpoints)
    furthR = np.amax(endpoints)
    furthPeak = np.amax(PEAKS)
    
    return furthR, furthPeak, quadLoc , PEAKS, endpoints

def check(x, y, R, data):
    # check(24.4, 34.4, 25, image_data)
    # function iterates the algorithm with a larger inital guess of the cluster. Continueing until the correct endpoint is calculated.
    # either returns the correct guess for the cluster, or raises an error if the algorithm iterates for too long. 

    #inputs:
        # 'x', 'y' <float>, coordinates of center of the cluster
        # 'R' <integer>, radius of the inital guess of the cluster
        # 'data' <2d array> image data that the cluster is from, whole .fits file
    
    #outputs:
        # 'iterativeR' <integer> , the correct radius to input into the algorithm to get an endpoint calculated that passes the SNR test

    ## note that 'R' & 'iterativeR' are radius inputs that are used to create a cut of the data thats centered on the cluster. not the actual radius of the cluster ##

    iterativeR = R
    rE = 1.0
    #make sure ratio Ends is a global variable: for re use this <-- personal note
    while rE >= 0.8 and iterativeR <= 80:
        rE = algo(x, y, iterativeR + 2, data, 0, False)
        iterativeR = iterativeR + 2
    if rE >= 0.8:
        print(f'Data process ran out of memory, please retry parameters' 
             f'n\ final Ratio was {rE:0.2f}')
        return iterativeR
    else:
        return iterativeR

#this marks the end of all commented code.

def plot(values, cx, cy, data, index):
    maxR = values[-1]
    cluster, cluster_W, cluster_B, SIGMA = info(cx,cy,maxR,data, False)
    
    
    radii = np.linspace(1,maxR,maxR)
    
    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(20,20)

    fg1 = ax[0,0].imshow(cluster_W, cmap = 'plasma',
                            norm = colors.LogNorm(), origin = 'lower')

    fg2 = ax[1,0].imshow(cluster_B, cmap = 'plasma',
                            norm = colors.LogNorm(), origin = 'lower')
                
    endCircleW = plt.Circle((maxR+12, maxR+12), values[3], linewidth = 5, 
                            color = 'red' , fill = False, label ='End of cluster')
    wideCW = plt.Circle((maxR+12, maxR+12), values[4], linewidth = 3, linestyle = 'dashed',
                        color = 'black' , fill = False, label = 'FWHM point')
    peakC = plt.Circle((maxR+12,maxR+12), values[5], linewidth = 3, linestyle = 'dashed', 
                       color = 'magenta', fill = False, label = 'peak point')
    ax[0,0].add_patch(endCircleW)
    ax[0,0].add_patch(wideCW)
    ax[0,0].legend(loc = 'best')
    annulusI = plt.Circle((maxR+12, maxR+12), maxR + 2, 
                          linewidth = 5, color = 'pink' , fill = False)
    annulusO = plt.Circle((maxR+12, maxR+12), maxR + 12, linewidth = 5, color = 'pink' , fill = False)
    ax[0,0].add_patch(annulusI)            
    ax[0,0].add_patch(annulusO)
    ax[0,0].title.set_text(f'Wide View of Cluster {index}')
            
    endCircle = plt.Circle((maxR, maxR), values[3], linewidth = 5, 
                           color = 'red' , fill = False, label = 'End of cluster')
    wideCircle = plt.Circle((maxR, maxR), values[4], linewidth = 3, 
                            linestyle = 'dashed', color = 'black' , fill = False, label = 'FWHM point')
    peakCircle = plt.Circle((maxR, maxR), values[5], linewidth = 3, 
                            linestyle = 'dashed', color = 'magenta' , fill = False, label = 'peak point')
    ax[1,0].add_patch(endCircle)
    ax[1,0].add_patch(wideCircle)
    ax[1,0].add_patch(peakCircle)
    ax[1,0].title.set_text(f'Blurred Cluster {index}: Sigma = {SIGMA}')
    ax[1,0].legend(loc = 'best')       
    
    
    xp, yp = quarters(values[7], maxR)
    xw, yw = quarters(values[8], maxR)
    
    ax[1,0].plot(xp[0],yp[0])
    ax[1,0].plot(xp[1],yp[1])
    ax[1,0].plot(xp[2],yp[2])
    ax[1,0].plot(xp[3],yp[3])
    
    ax[1,0].plot(xw[0],yw[0])
    ax[1,0].plot(xw[1],yw[1])
    ax[1,0].plot(xw[2],yw[2])
    ax[1,0].plot(xw[3],yw[3])
    
    ax[0,1].plot(radii, values[2][0], c = 'purple', linewidth = 3, marker = 'o', label = f'SNR')
    ax[0,1].plot(radii, values[2][1], c = 'red', linewidth = 3, marker = 'o', label = f'RL SNR2')
    ax[0,1].plot(radii, values[2][2], c = 'blue', linewidth = 3, marker = 'o', label = f'RL SNR3')
    ax[0,1].plot(radii, values[2][3], c = 'orange', linewidth = 3, marker = 'o', label = f'RL SNR4')
    ax[0,1].title.set_text(f'Cluster {index} quadrant SNR')
    ax[0,1].legend(loc = 'best')
            

    ax[1,1].plot(radii,values[1][0], c = 'purple', linewidth = 3,  label = f'RL SNR1')
    ax[1,1].plot(radii,values[1][1], c = 'red', linewidth = 3,  label = f'RL SNR2')
    ax[1,1].plot(radii,values[1][2], c = 'blue', linewidth = 3,  label = f'RL SNR3')
    ax[1,1].plot(radii,values[1][3], c = 'orange', linewidth = 3,  label = f'RL SNR4')
    
    ax[1,1].plot(radii,values[9], c = 'black', linewidth = 3, label = 'mean', linestyle = 'dashed')
    ax[1,1].errorbar(radii, values[9], yerr = values[10], c ='lime', label = 'error', linestyle = 'none')
    
    ax[1,1].axvline(values[3], c = 'red', linestyle = 'dashed', label = f'End point {values[3]}')
    ax[1,1].axvline(values[4], c = 'black', linestyle = 'dashed', label = f'Widest point {values[4]:0.2f}')
    ax[1,1].axvline(values[5], c = 'magenta', linestyle = 'dashed', label = f'Peak Location {values[5]},' 
                    f'quadrant {values[6]}')
    
    ax[1,1].axvline(values[11][0], c = 'purple', linestyle = 'dashed', label = f'cap {values[11][0]}')
    ax[1,1].axvline(values[11][1], c = 'gray', linestyle = 'dashed', label = f'cap {values[11][1]}')
    ax[1,1].axvline(values[11][2], c = 'blue', linestyle = 'dashed', label = f'cap {values[11][2]}')
    ax[1,1].axvline(values[11][3], c = 'orange', linestyle = 'dashed', label = f'cap {values[11][3]}')
    
    ax[1,1].title.set_text(f'Cluster {index} Blurred quadrant SNR')
    ax[1,1].legend(loc = 'best')

def quarters(Rs, R):
    angles = [np.linspace( 0 , np.pi / 2 , 40 ), np.linspace( np.pi / 2 , np.pi , 40 ), 
                  np.linspace( np.pi , 3* np.pi / 2 , 40 ), np.linspace( 3*np.pi / 2 , 2 * np.pi , 40 )]
    x = []
    y = []
    for i in range(4):
        xi = Rs[i] * np.cos( angles[i] ) + R
        yi = Rs[i] * np.sin( angles[i] ) + R
        x.append(xi)
        y.append(yi)
    return x,y

def quad_plot(values, cx, cy, data, index):
    maxR = values[-1]
    cluster, cluster_W, cluster_B, SIGMA = info(cx,cy,maxR,data, False)
    
    
    radii = np.linspace(1,maxR,maxR)
    
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(20,10)

    fg1 = ax[0].imshow(cluster_W, cmap = 'gray',
                            norm = colors.LogNorm(), origin = 'lower')

    fg2 = ax[1].imshow(cluster_B, cmap = 'plasma',
                            norm = colors.LogNorm(), origin = 'lower')
                
    endCircleW = plt.Circle((maxR+12, maxR+12), values[3], linewidth = 5, 
                            color = 'red' , fill = False, label ='End of cluster')
    wideCW = plt.Circle((maxR+12, maxR+12), values[4], linewidth = 3, linestyle = 'dashed',
                        color = 'black' , fill = False, label = 'FWHM point')
    peakC = plt.Circle((maxR+12,maxR+12), values[5], linewidth = 3, linestyle = 'dashed', 
                       color = 'magenta', fill = False, label = 'peak point')
    ax[0].add_patch(endCircleW)
    ax[0].add_patch(wideCW)
    ax[0].legend(loc = 'best')
    annulusI = plt.Circle((maxR+12, maxR+12), maxR + 2, 
                          linewidth = 5, color = 'pink' , fill = False)
    annulusO = plt.Circle((maxR+12, maxR+12), maxR + 12, linewidth = 5, color = 'pink' , fill = False)
    ax[0].add_patch(annulusI)            
    ax[0].add_patch(annulusO)
    ax[0].title.set_text(f'Wide View of Cluster {index}')
            
    endCircle = plt.Circle((maxR, maxR), values[3], linewidth = 5, 
                           color = 'red' , fill = False, label = 'End of cluster')
    wideCircle = plt.Circle((maxR, maxR), values[4], linewidth = 3, 
                            linestyle = 'dashed', color = 'black' , fill = False, label = 'FWHM point')
    peakCircle = plt.Circle((maxR, maxR), values[5], linewidth = 3, 
                            linestyle = 'dashed', color = 'magenta' , fill = False, label = 'peak point')
    ax[1].add_patch(endCircle)
    ax[1].add_patch(wideCircle)
    ax[1].add_patch(peakCircle)
    ax[1].title.set_text(f'Blurred Cluster {index}: Sigma = {SIGMA}')
    ax[1].legend(loc = 'best')       
    
    
    xp, yp = quarters(values[6], maxR+12)
    
    ax[0].plot(xp[0],yp[0], c = 'purple', linewidth = 3)
    ax[0].plot(xp[1],yp[1], c = 'red', linewidth = 3)
    ax[0].plot(xp[2],yp[2], c = 'blue', linewidth = 3)
    ax[0].plot(xp[3],yp[3], c = 'orange', linewidth = 3)



def half_effective(phot, end):
    cluster_counts = phot[0:end]
    phot_FRAC = cluster_counts / cluster_counts[-1]
    H_E = phot_FRAC.where(0.5)

    return H_E



    


