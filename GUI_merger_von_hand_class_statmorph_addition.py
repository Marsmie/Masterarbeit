# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:23:40 2024

@author: airub
"""

import tkinter as tk
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,  NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
from io import BytesIO
import os
from scipy.ndimage import rotate
from astropy.io import fits

class App:
    def __init__(self, master):
        self.master = master
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.index = 0
        self.satur_original = 1
        self.zoom_original = 510
        self.ordner = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/Classification/'
        self.pdFrame = pd.DataFrame()
        self.pdFrame = df
        self.leng = len(self.pdFrame)-1
        self.progress_label = tk.Label(master, text="")
        self.progress_label.pack()
        self.save_label = tk.Label(master, text="")
        self.save_label.pack()
        self.master.bind('<m>', self.set_merger)
        self.master.bind('<n>', self.set_no_merger)
        self.master.bind('<Right>', self.low_satur)
        self.master.bind('<Left>', self.raise_satur)
        self.master.bind('<k>', self.set_chaotic_disk)
        
        """
        self._button = tk.Button(self.master, text="Tutorial", command=self.open_tutorial)
        self._button.pack()
        """
        
        self.frame = tk.Frame(self.master)
        self.frame.pack(side = 'left')
        
        self.load_cat_button = tk.Button(self.frame, text="Katalog laden", command=self.load_cat)
        self.load_cat_button.pack()
        
        self.load_cat_label = tk.Label(self.frame, text="")
        self.load_cat_label.pack()
        
        self._label = tk.Label(self.frame, text="Benutzername")
        self._label.pack()
        
        self.username_entry = tk.Entry(self.frame, text = "Benutzername")
        self.username_entry.pack()
        
        self.slider = tk.Scale(self.frame, from_=0, to=10000, orient='horizontal', command=self.update_row)
        self.slider.pack()
        self.slider.set(0)

        self.next_button = tk.Button(self.frame, text="Naechstes Bild", command=self.next_image)
        self.next_button.pack()

        self.prev_button = tk.Button(self.frame, text="Vorheriges Bild", command=self.prev_image)
        self.prev_button.pack()

        self.space_label = tk.Label(self.frame, text="")
        self.space_label.pack()

        self.merger_button = tk.Button(self.frame, text="Merger", command=self.set_merger)
        self.no_merger_button = tk.Button(self.frame, text="Kein Merger", command=self.set_no_merger)
        self.chaotic_disk_button = tk.Button(self.frame, text="Chaotische Scheibe", command=self.set_chaotic_disk)
        self.merger_button.pack()
        self.chaotic_disk_button.pack()
        self.no_merger_button.pack()
        
        self.load_label = tk.Label(self.frame, text="")
        self.load_label.pack()
        
        self.load_button = tk.Button(self.frame, text="Datei laden", command=self.load_file)
        self.load_button.pack()
        
        self.close_label = tk.Label(self.frame, text="")
        self.close_label.pack()
        
        self.close_button = tk.Button(self.frame, text="Speichern & Schließen", command=self.on_close)
        self.close_button.pack()

        
        self.frame_r = tk.Frame(self.master)
        self.frame_r.pack(side = 'right')
        
        self.satur_slider_label = tk.Label(self.frame_r, text="Saturierung")
        self.satur_slider_label.pack()
        
        self.slider_satur = tk.Scale(self.frame_r, from_=200, to=0, orient='horizontal', command=self.update_satur)
        self.slider_satur.pack()
        self.slider_satur.set(195)
        
        self.zoom_slider_label = tk.Label(self.frame_r, text="Zoom")
        self.zoom_slider_label.pack()
        
        self.slider_zoom = tk.Scale(self.frame_r, from_=2, to=512, orient='horizontal', command=self.update_zoom)
        self.slider_zoom.pack()
        self.slider_zoom.set(256)
        
        

        self.fig = Figure(figsize=(15,15 ))
        self.ax_array = self.fig.subplots(2,2, squeeze=False)
        self.ax1 = self.ax_array[0,0]
        self.ax2 = self.ax_array[0,1]
        self.ax3 = self.ax_array[1,0]
        self.ax4 = self.ax_array[1,1]
        
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        
        self.update_image()
        
    """
    def open_tutorial(self):
        # Erstellen Sie ein neues Fenster
        self.tutorial_window = tk.Toplevel(self.master)
        self.tutorial_window.title("Tutorial")
        
        self.tutor_frame = tk.Frame(self.tutorial_window)
        self.tutor_frame.pack(side = 'left')
        
        self.load_merger_button = tk.Button(self.tutor_frame, text="Merger", command=self.load_merger)
        self._load_no_merger_button = tk.Button(self.tutor_frame, text="Kein Merger", command=self.load_no_merger)
        self.load_merger_button.pack()
        self.load_no_merger_button.pack()
        
        self.fig_merger = plt.Figure(figsize=(9, 9))
        self.canvas_merger = FigureCanvasTkAgg(self.fig_merger, master=self.tutorial_window)  # A tk.DrawingArea.
        self.canvas_merger.draw()
        self.canvas_merger.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas_merger, self.tutorial_window)
        self.toolbar.update()
        self.canvas_merger.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
    
        # Laden Sie die Bilder aus dem Tutorial-Ordner
        tutorial_directory = "Pfad zu Ihrem Tutorial-Ordner"
        tutorial_images = [ImageTk.PhotoImage(Image.open(os.path.join(tutorial_directory, filename))) for filename in os.listdir(tutorial_directory)]
    
        # Erstellen Sie ein Label für jedes Bild und fügen Sie es zum Fenster hinzu
        for image in tutorial_images:
            label = tk.Label(tutorial_window, image=image)
            label.pack()
        
        
    def load_merger(self):
        print('Load merger')
        
        
    def load_no_merger(self):
        print('load no merger')
    """
    
    
    def update_row(self, value):
        value = self.slider.get()
        self.index = value
        self.update_image()
    
    def update_satur(self, value):
        value = self.slider_satur.get()
        self.satur_original = value
        self.update_image()
        
    def update_zoom(self, value):
        value = self.slider_zoom.get()
        self.zoom_original = value
        self.update_image()
    

        
    def set_value(self):
        self.slider.set(self.index)
        
    def delete_old_files(self):
        username = self.username_entry.get()
        directory = self.ordner

        for filename in os.listdir(directory):
            # Extrahieren Sie den Index und den Benutzernamen aus dem Dateinamen
            try:
                file_username, file_index = filename.split('_')[0], int(filename.split('_')[-1][:-4])
            except:
                file_username = '_'
                file_index = 0

            # Ueberpruefen Sie, ob der Benutzername und der Index der Datei den Bedingungen entsprechen
            if file_username == username and file_index < self.index:
                # Loeschen Sie die Datei
                os.remove(os.path.join(directory, filename))
        
    
    def save_DF(self):
        if self.index > 0:
            username = self.username_entry.get()
            self.delete_old_files()
            #df.to_csv(f'{username}_gespeichertes_dataframe.csv', index=False)
            self.pdFrame.to_csv(f'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/Classification/{username}_classificaiton_{self.index}.csv', index=False)
            self.save_label.config(text = 'Zwischen gespeichert')
        
        
    def load_file(self):
        username = self.username_entry.get()
        filepath = askopenfilename(initialdir=f"C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/Classification/",
                                   title="Waehlen Sie eine Datei aus",
                                   filetypes=(("CSV Dateien", "*.csv"), ("Alle Dateien", "*.*")))
        if filepath:
            self.pdFrame = pd.read_csv(filepath)
            #self.paths = self.pdFrame['path'].values
            self.index = int(filepath.split('_')[-1][:-4])
            self.set_value()
            self.username_entry.insert(0, str(filepath.split('/')[-1].split('_')[0]))
            
            self.update_image()
            
    def load_cat(self):
        self.loaded = True
        #username = self.username_entry.get()
        self.filepath_cat = askopenfilename(initialdir=f"C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/solo_test/",
                                   title="Waehlen Sie einen Katalog aus",
                                   filetypes=(("CSV Dateien", "*.csv"), ("Alle Dateien", "*.*")))
        
        fits_ordner = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/FITS/512/'
        
        Kado = 'Kado_in_KIDS_512/'
        Paudel = 'Paudel_in_KIDS_512/'
        DR3 = 'Cutouts_KIDS_512/'
        Voro = 'Vorontsov/'
        
        ##lst = os.listdir(ordner)
        
        
        if self.filepath_cat:
            modus = self.filepath_cat.split('_')[0]
            training_set = self.filepath_cat.split('trainiert_auf_')[1].split('_')[0]
            feature_set = self.filepath_cat.split('_feature')[0].split(training_set+ '_')[-1]
            teilnehmer_set = self.filepath_cat.split('_trainiert')[0].split(modus+'_')[1].split('_')
            
            self.df_cat = pd.read_csv(self.filepath_cat).sort_values(by = '1', ascending = False, ignore_index = True)
            
            self.pdFrame = self.df_cat.copy()
            
            gruppe = self.df_cat['Unnamed: 0']
            id_ = self.df_cat['Unnamed: 1']
            rang = self.df_cat.index
            wahr = self.df_cat['1']
            
            pfad_new = []

            for i in range(len(self.df_cat)):
                
                groupe = gruppe.iloc[i]
                id_name = id_.iloc[i]
                rang_ = rang[i]
                wahr_ = wahr.iloc[i]
                
                if groupe == 'Paudel':
                    datei_name = fits_ordner + Paudel + 'Id' + str(id_name) + '.fits'
                    pfad_new.append(datei_name)
                    
                
                if groupe == 'Voro':
                    datei_name = fits_ordner + Voro + str(id_name) + '.fits'
                    pfad_new.append(datei_name)
                    
                
                if groupe == 'Kado':
                    datei_name = fits_ordner + Kado + str(id_name) + '.fits'
                    pfad_new.append(datei_name)
                    
                    
                if groupe == 'DR3':
                    datei_name = fits_ordner + DR3 + str(id_name) + '.fits'
                    pfad_new.append(datei_name)
                    
            self.pdFrame['Path'] = pfad_new
            self.pdFrame['NoMerger'] = np.zeros(len(pfad_new))
            self.pdFrame['Merger'] = np.zeros(len(pfad_new))
            self.pdFrame['Chaotic_disk'] = np.zeros(len(pfad_new))
            self.pfad = pfad_new      
            self.index = 0
            
            
            self.update_image()
        
    def on_close(self):
        self.save_DF()
        self.master.destroy()
        
    def next_image(self):
        self.index += 1
        self.set_value()
        self.update_image()

    def prev_image(self):
        self.index -= 1
        self.set_value()
        self.update_image()
    
    def low_satur(self, event=None):
        self.satur_original -=1
        self.slider_satur.set(self.satur_original)
        self.update_image()
    
    def raise_satur(self, event=None):
        self.satur_original +=1
        self.slider_satur.set(self.satur_original)
        self.update_image()

    def set_merger(self, event=None):
        self.pdFrame.loc[int(self.index), 'Merger'] = int(1)
        self.pdFrame.loc[int(self.index), 'Chaotic_disk'] = int(0)
        self.index += 1
        self.set_value()
        self.update_image()

    def set_no_merger(self, event=None):
        self.pdFrame.loc[int(self.index), 'NoMerger'] = int(1)
        self.pdFrame.loc[int(self.index), 'Merger'] = int(0)
        self.pdFrame.loc[int(self.index), 'Chaotic_disk'] =int(0)
        self.index += 1
        self.set_value()
        self.update_image()
    
    def set_chaotic_disk(self, event=None):
        self.pdFrame.loc[int(self.index), 'Chaotic_disk'] = int(1)
        self.index += 1
        self.set_value()
        self.update_image()


    def update_image(self):
        self.save_label.config(text = pfad[int(self.index)])
        if (self.index+1) %100 == 0:
            
            self.save_DF()
        if self.index > self.leng:
            self.on_close()
    
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        with fits.open(pfad[int(self.index)]) as datei:
            data = datei[2].data
            #data += datei[2].data
            #data += datei[3].data
            #data += datei[4].data
        vmin = np.percentile(data, 0.5)
        vmax = np.percentile(data, 99.9)
        #Bild 1
        self.ax1.imshow(data, norm = 'symlog', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        self.ax1.set_title('Original Bild')
        
        #Bild 2 asymetisches Bild 
        difference_data = data
        vmin = np.percentile(data, 0.5)
        vmax = np.percentile(data, 99.5)
        self.ax2.imshow(difference_data, norm = 'symlog', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        self.ax2.set_title('saturiertes Bild 99.5%')
        
        #Bild 3 asymetisches Bild 
        rotated_data = rotate(data,180)
        im_size = 512
        crop_size = self.zoom_original
        
        
        cropped_data = data[256-int(crop_size/2):256+ int(crop_size/2), 256- int(crop_size/2):256+ int(crop_size/2)]
        vmin = np.percentile(cropped_data, 0.5)
        satur_max = 97.9 + self.satur_original/100
        vmax = np.percentile(cropped_data, satur_max)
        self.ax3.imshow(cropped_data, norm = 'symlog', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        self.ax3.set_title(f'saturiertes Bild {satur_max}%')
        
        #Bild 4 asymetisches Bild 
        try:
            difference_data = abs(data-rotated_data) / data
        except:
            difference_data = abs(data-rotated_data)
        diff_rot = rotate(difference_data, 180)
        vmin = np.percentile(difference_data, 0.5)
        vmax = np.percentile(difference_data, 99.5)
        self.ax4.imshow(diff_rot, norm = 'symlog',  cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        self.ax4.set_title('asymetrisches Bild')
        
        self.canvas.draw()
        
        # Update progress label
        self.progress_label.config(text=f"Bild {self.index} von {len(df)}")
        
        self.toolbar.update()
            

        
ordner = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/solo_test/'


fits_ordner = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/FITS/512/'

Kado = 'Kado_in_KIDS_512/'
Paudel = 'Paudel_in_KIDS_512/'
DR3 = 'Cutouts_KIDS_512/'
Voro = 'Vorontsov/'

lst = os.listdir(ordner)



i = 5


setting = askopenfilename(initialdir=f"C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/solo_test/",
                           title="Waehlen Sie einen Katalog aus",
                           filetypes=(("CSV Dateien", "*.csv"), ("Alle Dateien", "*.*")))


df = pd.read_csv(setting)
lange = len(df)
test_Paudel = df.where(df['Unnamed: 0'] == 'DM').dropna().sort_values(by = '1', ascending = False)
test_DR3 = df.where(df['Unnamed: 0'] == 'DR3').dropna().sort_values(by = '1', ascending = False)
test_Kado = df.where(df['Unnamed: 0'] == 'Kado-Fong').dropna().sort_values(by = '1', ascending = False)
grup_Kado = 'Kado-Fong'
if len(test_Kado) == 0:
    test_Kado = df.where(df['Unnamed: 0'] == 'Kado').dropna().sort_values(by = '1', ascending = False)
    grup_Kado = 'Kado'
    
test_Voro = df.where(df['Unnamed: 0'] == 'Voro').dropna().sort_values(by = '1', ascending = False)



print(ordner+ lst[i])
df = df.sort_values(by = '1', ascending = False)
gruppe = df['Unnamed: 0']
id_ = df['Unnamed: 1']
rang = df.index
wahr = df['1']

pfad_1 = []

for i in range(len(df)):
    
    groupe = gruppe.iloc[i]
    id_name = id_.iloc[i]
    rang_ = rang[i]
    wahr_ = wahr.iloc[i]
    
    if groupe == 'Paudel':
        datei_name = fits_ordner + Paudel + 'Id' + str(id_name) + '.fits'
        pfad_1.append(datei_name)
        
    
    if groupe == 'Voro':
        datei_name = fits_ordner + Voro + str(id_name) + '.fits'
        pfad_1.append(datei_name)
        
    
    if groupe == 'Kado':
        datei_name = fits_ordner + Kado + str(id_name) + '.fits'
        pfad_1.append(datei_name)
        
        
    if groupe == 'DR3':
        datei_name = fits_ordner + DR3 + str(id_name) + '.fits'
        pfad_1.append(datei_name)
        


#merger_proba_csv = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/Paudel_obj_kado_sorted_and_proba_13_06_24.csv'
#df = pd.read_csv(merger_proba_csv)
#df = df.head(100)
df['Merger'] = int(0)
df['NoMerger'] = int(0)
df['Chaotic_disk'] = int(0)

#pfad = df['path'].values
pfad = pfad_1
#pfad_1 


root = tk.Tk()
app = App(root)
root.mainloop()