# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:05:50 2024

Dieses Programm soll von den Fohlen Server die Bilder meiner Galaxien Laden und aus einem Katalog Cut outs erstellen.

Dafür muss ich in einer Datei auf dem Server nachschauen unter welchem Verzeichnis meine Galaxie zu finden ist,
danach wo die Galaxie auf dem Bild ist und mir überlegen wie viel ich ausschneiden soll.

/net/fohlen13/home/awright/KiDS/Legacy/Production/AstroWISE/imaging/ Ort der Bilder 

/net/fohlen11/home/awright/src/PhotoPipe/config/KIDS_ra_dec_cuts.txt  Ort wo zu welchem Bild welche Koordinaten gelten


überlege wie ich auf diese Dateistruktur zugreifen kann oder soll?
Soll das Programm auf den Fohlen Server laufen oder kann das auch hier auf dem SFB Rechner 
laufen und remote auf die Fohlen zugreifen?

@author: airub
"""

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
import astropy.units as u
#from sdss import Region
#from PIL import Image
#import time
import multiprocessing as mp
#import math
from astropy.visualization import SqrtStretch
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.datasets import load_star_image
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
from astropy.nddata import Cutout2D



def zuweisung(): #Benötigt alles Ordnet den DG die außerhalb des Bildes liege die gruppe 0 zu
                 # erstetzt in den Bild namen 'p' mit '.' und 'm' mit '-'
    for i in range(len(ID)):
        Image_name[i] = Image_name[i].replace('p', '.')
        Image_name[i] = Image_name[i].replace('m', '-')    
        
            
    
    
def source_detection(data, show=False):
    detect_data = data
    mean, median, std = sigma_clipped_stats(detect_data, sigma=3.0)  
    #print((mean, median, std))  

    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)  
    sources = daofind(detect_data - median) 
    for col in sources.colnames:
        if col not in ('id', 'npix'):
            sources[col].info.format = '%.2f'
    sources.pprint(max_width=76)

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=4.0)
    norm = ImageNormalize(stretch=SqrtStretch())
    
    if show:
        plt.imshow(detect_data, cmap='Greys', origin='lower', norm=norm,interpolation='nearest')
        apertures.plot(color='blue', lw=1.5, alpha=0.5)
    
    return sources
        
    
    



def bild_cut(filter_, sky, Name, dist=100, show=False):
    #im = fits.open(ordner+'Data/'+ Image_name[i] + filter_) FÜR SFB RECHNER
    im = fits.open(KIDS_ordner+ filter_+'/'+ Name+'_'+filter_ +'.fits') #Für Fohlen13
    #im.info()
    im_dat = im[0].data
    wcs = WCS(im[0].header)
    x_pix, y_pix = wcs.world_to_pixel(sky)
    x_pix = int(x_pix)
    y_pix = int(y_pix)
    pos = (x_pix,y_pix)
    size = dist
    im_dg = Cutout2D(im_dat,pos, size)

    if show:
        plt.imshow(im_dg.data, origin='lower')
        plt.title(str(i)+str(filter_))
        plt.show()
    im.close()
    return im_dg.data, im[0].header
    
### Test Bereich cut out für die erste ID Groupe



def cuts(CATAI, Ra_de, Dec_de, Name):
    for i in range(0, len(CATAI)):
        sky = SkyCoord(Ra_de[i], Dec_de[i], frame='icrs', unit = 'deg')
        name = Name[i]
        dist = 512
        try:
            im_u_dat, im_u_header = bild_cut(filter_u, sky, name,  dist)
            im_r_dat, im_r_header  = bild_cut(filter_r, sky, name, dist)
            im_g_dat, im_g_header  = bild_cut(filter_g, sky, name, dist)
            im_i1_dat, im_i1_header  = bild_cut(filter_i1, sky, name, dist)
            im_i2_dat, im_i2_header  = bild_cut(filter_i2, sky, name, dist)
            
            prim = fits.PrimaryHDU((1,1))
            im_hud_u = fits.ImageHDU(im_u_dat, im_u_header)
            im_hud_r = fits.ImageHDU(im_r_dat, im_r_header)
            im_hud_g = fits.ImageHDU(im_g_dat, im_g_header)
            im_hud_i1 = fits.ImageHDU(im_i1_dat, im_i1_header)
            im_hud_i2 = fits.ImageHDU(im_i2_dat, im_i2_header)
            
            hdul = fits.HDUList([prim, im_hud_u, im_hud_r, im_hud_g, im_hud_i1, im_hud_i2])
            
            hdul.writeto(KIDS_output_ordner+str(CATAI[i])+'.fits', overwrite =True)
        except:
            print(CATAI[i], 'wurde nicht bearbeitet')
        if i%100==0: print(i)
        
if __name__ == "__main__":


    ###KIDS URLS
    global KIDS_ordner
    KIDS_ordner = '/net/fohlen13/home/awright/KiDS/Legacy/Production/AstroWISE/imaging/'
    global KIDS_output_ordner
    KIDS_output_ordner = '/net/fohlen13/home/marsmie/Cutouts/Vorontsov/'
    global KIDS_bildzuweisung
    KIDS_bildzuweisung = '/net/fohlen13/home/marsmie/ORyan_in_KIDS_0.515_degree.fits'
    speicherort = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/ORyan_in_KIDS_0.515_degree.fits'

    filter_u = 'u'
    filter_r = 'r'
    filter_g = 'g'
    filter_i1 = 'i1'
    filter_i2 = 'i2'

    fits_file = fits.open(KIDS_bildzuweisung)

    bildzuweisung_data = fits_file[1].data

    ID = bildzuweisung_data['SourceID']
    #Z = bildzuweisung_data['zsp']
    Ra_deg = bildzuweisung_data['RA']
    Dec_deg = bildzuweisung_data['Dec']
    Image_name = bildzuweisung_data['Name']


    fits_file.close()

    cnt = 0
    zuweisung()         
    processes = []
    data_CAT_cut = []
    data_Ra_cut = []
    data_Dec_cut = []
    dat_name = []
    cpu = 1
    step = int(len(ID)/cpu)
    
    for i in range(cpu+1):
        data_CAT_cut.append(ID[step*i:step*(i+1)])
        data_Ra_cut.append(Ra_deg[step*i:step*(i+1)])
        data_Dec_cut.append(Dec_deg[step*i:step*(i+1)])
        dat_name.append(Image_name[step*i:step*(i+1)])
        
        
    
    for multi in range(cpu+1):
        p = mp.Process(target=cuts, args = (data_CAT_cut[multi], data_Ra_cut[multi], data_Dec_cut[multi], dat_name[multi]))
        p.start()
        processes.append(p)
        
    for process in processes:
        process.join()
           