# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:01:52 2024

@author: airub
"""

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import plot_tree # to draw a classification tree
from sklearn.model_selection import train_test_split # to split data into training and teting sets
from sklearn.model_selection import cross_val_score # for cross validation
from sklearn.metrics import confusion_matrix  as cm# to create a confusion matrix
#from sklearn.metrics import plot_confusion_matrix # to draw a confusion matrix
from sklearn.metrics._plot import confusion_matrix as cmd
import os
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
from astropy.wcs import WCS
from sklearn.model_selection import cross_validate
import glob
from astropy.io import fits
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import (detect_sources, make_2dgaussian_kernel)
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip
from scipy.ndimage import gaussian_filter
#from sklearn.externals import joblib
import os
import os.path
from astropy.nddata import Cutout2D


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

def get_data(url, dist, ra, dec):
    with fits.open(url) as hdul:
        data_1 = hdul[0].data
        header_1 = hdul[0].header
        wcs_1 = WCS(header_1)
        ra_ = ra * u.deg
        dec_ = dec * u.deg
        pos_ra_dec = SkyCoord(ra_, dec_,frame = 'icrs')
        pos1_xy = wcs_1.world_to_pixel(pos_ra_dec)
        x1 = int(pos1_xy[0])
        y1 = int(pos1_xy[1])
        pos = (x1,y1)
        size = dist
        im_dg = Cutout2D(data_1,pos, size)
    return im_dg.data
                
def index_berechnung(pointing, filt, dist, ra, dec):
    data = get_data(KIDS_server_url+ filt+'/'+ pointing+'_'+filt +'.fits', dist, ra, dec)
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
        print(f'Das Pointing {pointing} hat an der Stelle  keinen Hintergrund')
    
    try:
        b_sym =  np.sum(abs(bkg - np.rot90(bkg, k=2)))/np.sum(abs(data))
    except:
        print(f'Das Pointing {pointing} hat an der Stelle  keine Symmetrie berechnet')
        b_sym = 0
        
    try:
        b_smoth = np.sum(bkg-gaussian_filter(bkg, sigma=4.5*0.2))/np.sum(data)
    except:
        print(f'Das Pointing {pointing} hat an der Stelle  keine Klumpung berechnet')
        b_smoth = 0
        

    sym = a_sym-b_sym
    smoth = a_smoth-b_smoth
    
    return sym, smoth

def cat_to_df(path):
    df = pd.DataFrame(fits.open(path)[1].data)
    df = df.where(df['MASK'] < 2)
    df = df.dropna()
    df = df.where(df['mstar_bestfit'] < 9)
    df = df.dropna()
    df = df.where(df['mstar_bestfit'] > 6)
    df = df.dropna()
    df_ID = df['ID']
    df_KIDS_TILE = df['KIDS_TILE']
    df_Xpos = df['Xpos']
    df_Ypos = df['Ypos']
    df_Ra = df['RAJ2000']
    df_Dec = df['DECJ2000']
    df_mstar_bestfit = df['mstar_bestfit']
    df_total = pd.concat([df_ID, df_KIDS_TILE, df_Xpos, df_Ypos, df_Ra, df_Dec, df_mstar_bestfit], axis = 1)
    return df_total
    
def save_index(df):
    df_ID = df['ID']
    df_point = df['KIDS_TILE']
    df_Xpos = df['Xpos']
    df_Ypos = df['Ypos']
    df_Ra = df['RAJ2000']
    df_Dec = df['DECJ2000']
    
    df_index = pd.DataFrame()
    for cnt, point in enumerate(df_point):
        save_path = server_speicher + point + '_CAS.csv' 
        save_koord = koord_speicher + point + '_koord.csv'
        ra = df_Ra.iloc[cnt]
        dec = df_Dec.iloc[cnt]
        id_ = df_ID.iloc[cnt]
        
        
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



    
        
        dist = 128
        
        lst_sym = []
        lst_lst_sym = []
        lst_smoth = []
        lst_lst_smoth = []
        for filt in filter_:  
            sym, smoth = index_berechnung(point, filt, dist, ra, dec)
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
        df_sym = pd.DataFrame(lst_lst_sym, columns = col_sym, index = [id_])
        df_smoth = pd.DataFrame(lst_lst_smoth, columns = col_smoth, index = [id_])
        df_einzel_seg = pd.concat([df_sym, df_smoth], axis = 1)
        
        df_index = pd.concat([df_index, df_einzel_seg], axis = 0)
        if cnt% int(len(df_point)/10) == 0 : print(f"Es wurden {cnt} von {len(df_point)} im Pointing {point} bearbeitet.")
        
                    
    try:           
        df_index.to_csv(save_path)
    except:
        print(f'Die Datei mit Namen {point} konnte nicht gespeichert werden')
        
        
        
def multi_save_index(lst_df):
    for i in lst_df:
        df_cat = cat_to_df(i)
        try:
            save_index(df_cat)
        except:
            print(f' Die Datei {i} kann nicht verarbeitet werden')
        
        
        
if __name__ == "__main__":
    import multiprocessing as mp
    
    global KIDS_server_url
    KIDS_server_url = '/net/fohlen13/home/awright/KiDS/Legacy/Production/AstroWISE/imaging/'
    global server_speicher_speicher
    server_speicher = '/net/fohlen13/home/marsmie/Dataset/CSV_Tree/128/'
    global koord_speicer
    koord_speicher = '/net/fohlen13/home/marsmie/Dataset/Koord/'
    global proba_load
    proba_load = '/net/fohlen13/home/marsmie/Dataset/Proba/zwischenspeicher/902559.csv'
    global csv_server
    csv_server = '/net/fohlen13/home/marsmie/Dataset/CSV_Tree/128/'
    global KIDS_cat_only_on_server
    KIDS_cat_only_on_server = '/net/fohlen13/home/awright/KiDS/Legacy/Production/KiDS/SuperUser_PostUpdate/'
    KIDS_cat_pc = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/Cat/'
    
    processes = []
    dat_name = []
    lst_cat = glob.glob(KIDS_cat_only_on_server + '*')
    lst_cat = lst_cat[3:]
    cpu = 5
    step = int(len(lst_cat)/cpu)
    for i in range(cpu+1):
        dat_name.append(lst_cat[step*i:step*(i+1)])
        
    for multi in range(cpu+1):
        p = mp.Process(target=multi_save_index, args = ([dat_name[multi]]))
        p.start()
        processes.append(p)
            
    for process in processes:
        process.join()
    
    """
    for cnt_cat, cat_url in enumerate(lst_cat):
        df_cat = cat_to_df(cat_url)
        filter_ = ['u', 'r', 'g', 'i1', 'i2']
        save_index(df_cat)
        if (cnt_cat)%(len(lst_cat)/10) == 0: print(f"Es wurden {cnt_cat} von {len(lst_cat)} bearbeitet")
    """
        



