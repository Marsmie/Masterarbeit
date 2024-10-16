# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:51:03 2024

@author: airub
"""

from astropy.io import fits
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import (detect_sources, make_2dgaussian_kernel)
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip
from scipy.ndimage import gaussian_filter
#from sklearn.externals import joblib
import os
import os.path
from astropy.nddata import Cutout2D



import multiprocessing as mp

def zuweisung(df, col): #Benötigt alles Ordnet den DG die außerhalb des Bildes liege die gruppe 0 zu
                 # erstetzt in den Bild namen 'p' mit '.' und 'm' mit '-'
    for i in range(len(df[col])):
        df[col].iat[i] = df[col].iat[i].replace('p', '.')
        df[col].iat[i] = df[col].iat[i].replace('m', '-') 
        
def make_segm(data):    
    # use sigma-clipped statistics to (roughly) estimate the background
    # background noise levels
    mean, _ , std = sigma_clipped_stats(data)
    
    # subtract the background
    data -= mean
    
    # detect the sources
    threshold = 3. * std
    kernel = make_2dgaussian_kernel(3.0, size=3)  # FWHM = 3.
    convolved_data = convolve(data, kernel)
    segm = detect_sources(convolved_data, threshold, npixels=50)
    return segm

def asym(data):
    try:
        asym = np.sum(abs(data - np.rot90(data, k=2)))/np.sum(abs(data))
    except:
        asym = -1
        
    return asym

def asmoth(data):
    try:
        A_smoth = np.sum(data-gaussian_filter(data, sigma=4.5*0.2))/np.sum(data)
    except:
        A_smoth = -1
    return A_smoth

def background(data):
    sigma_clip = SigmaClip(sigma = 3.0)
    bkg_estimator = MedianBackground()
    box_size = 25
    filter_size = 3
    bkg = Background2D(data,box_size = box_size, filter_size= filter_size, sigma_clip= sigma_clip, bkg_estimator= bkg_estimator)
    return bkg.background

def get_data(url, dist, pos_ra_dec):
    with fits.open(url) as hdul:
        data_1 = hdul[0].data
        header_1 = hdul[0].header
        wcs_1 = WCS(header_1)
        pos1_xy = wcs_1.world_to_pixel(pos_ra_dec)
        x1 = int(pos1_xy[0])
        y1 = int(pos1_xy[1])
        pos = (x1,y1)
        size = dist
        im_dg = Cutout2D(data_1,pos, size)
    return im_dg.data
                
def index_berechnung(pointing, filt, dist, stelle):
    data = get_data(KIDS_server_url+ filt+'/'+ pointing+'_'+filt +'.fits', dist, stelle)
    try:
        a_sym = asym(data)
        a_smoth = asmoth(data)
        b_sym = 0
        b_smoth = 0
        
        
            
    except:
        a_sym = -1
        b_sym = 0
        a_smoth = -1
        b_smoth = 0
        
    try:
        bkg = background(data)
        
    except:
        print(f'Das Pointing {pointing} hat an der Stelle {stelle} keinen Hintergrund')
    
    try:
        b_sym =  np.sum(abs(bkg - np.rot90(bkg, k=2)))/np.sum(abs(data))
    except:
        print(f'Das Pointing {pointing} hat an der Stelle {stelle} keine Symmetrie berechnet')
        b_sym = 0
        
    try:
        b_smoth = np.sum(bkg-gaussian_filter(bkg, sigma=4.5*0.2))/np.sum(data)
    except:
        print(f'Das Pointing {pointing} hat an der Stelle {stelle} keine Klumpung berechnet')
        b_smoth = 0
        

    sym = a_sym-b_sym
    smoth = a_smoth-b_smoth
        
    """
    if np.isnan(sym):
        
        sym = -1
    if np.isnan(smoth):
        smoth = -1
    """
    
    return sym, smoth

def ausschnitt_speichern(point, filt, stelle, anzahl):
    dist = 512
    data = get_data(KIDS_server_url+ filt+'/'+ point+'_'+filt +'.fits', dist, stelle)
    fits.writeto(fits_speicher+ str(anzahl)+ '.fits', data)
    
 
def single_steuerung(pointing):
        point = pointing
        filter_ = ['u', 'r', 'g', 'i1', 'i2']
        df_total_index = pd.DataFrame()
        try:
            im = fits.open(KIDS_server_url+ filter_[0]+'/'+ point+'_'+filter_[0] +'.fits') #Für Fohlen13  
        except:
            print(f'Das Pointing {point} existiert nicht')
            
        data = im[0].data
        header = im[0].header
        segm = make_segm(data)   
        

    
        df_koord = pd.DataFrame()
        df_index = pd.DataFrame()
         
        for cnt_seg, seg in enumerate(segm.segments):
            label = seg.label
            bbox = seg.bbox
            center = bbox.center
            y_center = int(center[0])
            x_center = int(center[1])
            
            
            df_pos = pd.DataFrame([[x_center, y_center]],columns = ['x_center', 'y_center'], index = [label])
        
            
            
            if x_center < 500 or y_center< 500 or (x_center + 500) > data.shape[1] or (y_center + 500) > data.shape[0]:
                continue
            else:
                if seg.area < 200 or seg.area > 16000:
                    continue
                else:
                    wcs = WCS(header)
                    pos_ra_dec = wcs.pixel_to_world(x_center, y_center)
                    dist = 128
                    stelle = pos_ra_dec
                    
                    lst_sym = []
                    lst_lst_sym = []
                    lst_smoth = []
                    lst_lst_smoth = []
                    for filt in filter_:
                        #if cnt_seg >100 and cnt_seg< 110:
                            #ausschnitt_speichern(point, filt, stelle, cnt_seg)

                            
                        sym, smoth = index_berechnung(point, filt, dist, stelle)
                        lst_sym.append(sym)
                        lst_smoth.append(smoth)
                    
                    lst_sym[-2] = (lst_sym[-2]+lst_sym[-1])/2
                    lst_sym.pop(-1)
                    lst_smoth[-2] = (lst_smoth[-2]+lst_smoth[-1])/2
                    lst_smoth.pop(-1)
                    lst_lst_sym.append(lst_sym)
                    lst_lst_smoth.append(lst_smoth)
                    
                    col_sym = ['sym_u', 'sym_r', 'sym_g', 'sym_i']
                    col_smoth = ['smoth_u', 'smoth_r', 'smoth_g', 'smoth_i']
                    point_label = str(point) + '_' + str(label)
                    df_sym = pd.DataFrame(lst_lst_sym, columns = col_sym, index = [label])
                    df_smoth = pd.DataFrame(lst_lst_smoth, columns = col_smoth, index = [label])
                    df_einzel_seg = pd.concat([df_sym, df_smoth], axis = 1)
                    
                    df_index = pd.concat([df_index, df_einzel_seg], axis = 0)
                    df_koord = pd.concat([df_koord, df_pos], axis = 0)
        
        
        save_path = server_speicher + point + '_CAS.csv'           
        df_index.to_csv(save_path)
        save_koord = koord_speicher + point + '_koord.csv'
        df_koord.to_csv(save_koord)
        
        
        
def grundsteuerung(pointing):
    for cnt, point in enumerate(pointing):
        save_path = server_speicher + point + '_CAS.csv' 
        save_koord = koord_speicher + point + '_koord.csv'
        
        """
        Überprüfe ob die Dateien schon existieren
        
        """
        if os.path.isfile(save_path): 
            print(f'Die Datei {save_path} existierts bereits und wird übersprungen')
            continue
        if os.path.isfile(save_koord): 
            print(f'Die Datei {save_koord} existierts bereits und wird übersprungen')
            continue
        
        
        filter_ = ['u', 'r', 'g', 'i1', 'i2']
        df_total_index = pd.DataFrame()
        try:
            im = fits.open(KIDS_server_url+ filter_[0]+'/'+ point+'_'+filter_[0] +'.fits') #Für Fohlen13  
        except:
            print(f'Das Pointing {point} existiert nicht')
            continue
        data = im[0].data
        header = im[0].header
        segm = make_segm(data)   
        im.close()
        if cnt%(int(len(pointing)/10))== 0: print(f'Es wurden {cnt} Pointings von {len(pointing)} Pointings bearbeitet')

    
        df_koord = pd.DataFrame()
        df_index = pd.DataFrame()
         
        for cnt_seg, seg in enumerate(segm.segments):
            label = seg.label
            bbox = seg.bbox
            center = bbox.center
            y_center = int(center[0])
            x_center = int(center[1])
            
            
            df_pos = pd.DataFrame([[x_center, y_center]],columns = ['x_center', 'y_center'], index = [label])
        
            
            
            if x_center < 500 or y_center< 500 or (x_center + 500) > data.shape[1] or (y_center + 500) > data.shape[0]:
                continue
            else:
                if seg.area < 200 or seg.area > 16000:
                    continue
                else:
                    wcs = WCS(header)
                    pos_ra_dec = wcs.pixel_to_world(x_center, y_center)
                    dist = 128
                    stelle = pos_ra_dec
                    
                    lst_sym = []
                    lst_lst_sym = []
                    lst_smoth = []
                    lst_lst_smoth = []
                    for filt in filter_:  
                        sym, smoth = index_berechnung(point, filt, dist, stelle)
                        lst_sym.append(sym)
                        lst_smoth.append(smoth)
                    
                    lst_sym[-2] = (lst_sym[-2]+lst_sym[-1])/2
                    lst_sym.pop(-1)
                    lst_smoth[-2] = (lst_smoth[-2]+lst_smoth[-1])/2
                    lst_smoth.pop(-1)
                    lst_lst_sym.append(lst_sym)
                    lst_lst_smoth.append(lst_smoth)
                    
                    col_sym = ['sym_u', 'sym_r', 'sym_g', 'sym_i']
                    col_smoth = ['smoth_u', 'smoth_r', 'smoth_g', 'smoth_i']
                    df_sym = pd.DataFrame(lst_lst_sym, columns = col_sym, index = [label])
                    df_smoth = pd.DataFrame(lst_lst_smoth, columns = col_smoth, index = [label])
                    df_einzel_seg = pd.concat([df_sym, df_smoth], axis = 1)
                    
                    df_index = pd.concat([df_index, df_einzel_seg], axis = 0)
                    df_koord = pd.concat([df_koord, df_pos], axis = 0)
                    
                  
        df_index.to_csv(save_path)
        df_koord.to_csv(save_koord)



                    
                    
        
                
                
                
if __name__ == "__main__":
    
    KIDS_rechner_url = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/GAMA/KIDS_ra_dec_cuts_mittelwerte.csv'
    global KIDS_server_url
    KIDS_server_url = '/net/fohlen13/home/awright/KiDS/Legacy/Production/AstroWISE/imaging/'
    global server_speicher
    server_speicher = '/net/fohlen13/home/marsmie/Dataset/CSV_Tree/128/'
    global koord_speicer
    koord_speicher = '/net/fohlen13/home/marsmie/Dataset/Koord/'
    global fits_speicher
    fits_speicher = '/net/fohlen13/home/marsmie/Dataset/FITS/'
    global Pointing_zuweisung
    Pointing_zuweisung = '/net/fohlen13/home/marsmie/KIDS_ra_dec_cuts_mittelwerte.csv'
 

    #df_KIDS_pointings = pd.read_csv(KIDS_rechner_url)
    df_KIDS_pointings = pd.read_csv(Pointing_zuweisung)
    
    
    
    zuweisung(df_KIDS_pointings, 'Name')
    pointing_name = df_KIDS_pointings['Name']
    
    processes = []
    dat_name = []

    cpu = 5
    step = int(len(pointing_name)/cpu)
    for i in range(cpu+1):
        dat_name.append(pointing_name[step*i:step*(i+1)])
        
    for multi in range(cpu+1):
        p = mp.Process(target=grundsteuerung, args = ([dat_name[multi]]))
        p.start()
        processes.append(p)
            
    for process in processes:
        process.join()
    
        

