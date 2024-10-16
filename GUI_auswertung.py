# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:23:40 2024

@author: marsmie
"""

import tkinter as tk
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        self.index = 20
        self.bin_size = 50
        self.dfs = []
        self.entries = []
        self.classification = pd.DataFrame()
        
        self.classification = pd.read_csv('C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/Classification/classificaiton_0.csv')
        
        
        self.user_name = []
        self.indexes = []
        self.df = pd.DataFrame()
        
        self.frame_l = tk.Frame(self.master)
        self.frame_l.pack(side = 'left')
        
        self.frame_r = tk.Frame(self.master)
        self.frame_r.pack(side = 'right')

        self.load_button = tk.Button(self.frame_l, text="CSV laden", command=self.load_csv)
        self.load_button.pack()
        
        self.bin_button = tk.Button(self.frame_l, text=f"Bin = {self.bin_size}", command=self.change_bin_size)
        self.bin_button.pack()
        
        self.top_10_button = tk.Button(self.frame_l, text="Top 9", command=self.plot_fits_files)
        self.top_10_button.pack()
        


        self.fig_hist = plt.Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig_hist, master=master)
        self.canvas.get_tk_widget().pack(side = 'left')
        
        """
        self.fig_merg, self.axs = plt.subplots(2, 5, figsize=(2, 2))
        self.canvas = FigureCanvasTkAgg(self.fig_merg, master=self.frame_r)
        self.canvas.get_tk_widget().pack(side = 'right')
        """
        
        self.fig_top9 = plt.Figure(figsize=(9, 9))
        self.canvas_top9 = FigureCanvasTkAgg(self.fig_top9, master=self.master)  
        self.canvas_top9.draw()
        self.canvas_top9.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas_top9, self.master)
        self.toolbar.update()
        self.canvas_top9.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.file_listbox = tk.Listbox(self.frame_l, selectmode='single')
        self.file_listbox.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
  
    def on_close(self):
        self.classification.to_csv('C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/Classification/classificaiton.csv', index = False)
        self.master.destroy()
        
    def load_csv(self):
        filepath = askopenfilename(initialdir=f"C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/Classification/",
                                   title="Wählen Sie eine Datei aus",
                                   filetypes=(("CSV Dateien", "*.csv"), ("Alle Dateien", "*.*")))
        if filepath:
            df = pd.read_csv(filepath)
            self.classification['Merger'] += df['Merger'].apply(lambda x: 1.0 if x==1 else 0)
            self.classification['Chaotic_disk'] += df['Chaotic_disk'].apply(lambda x: 1.0 if x==1 else 0)
            self.dfs.append(df)
            
            self.user_name.append(str(filepath.split('/')[-1].split('_')[0]))
            self.indexes.append(int(filepath.split('_')[-1][:-4]))
           
            self.entries.append((str(filepath.split('/')[-1].split('_')[0]), int(filepath.split('_')[-1][:-4])))
            self.update_listbox()
            
            self.update_plots()
    
            
    def update_listbox(self):
        self.file_listbox.delete(0,len(self.entries))
        for i, entry in enumerate(self.entries):
            self.file_listbox.insert(i, entry)
            
    def change_bin_size(self):
        if self.bin_size == 100: #50
            self.bin_size = 200 #100
        elif self.bin_size == 200: #100
            self.bin_size = 300 #200
        elif self.bin_size == 300: #200
            self.bin_size = 400
        elif self.bin_size == 400:
            self.bin_size = 500
        elif self.bin_size == 500:
            self.bin_size = 50   #20
        else:
            self.bin_size = 100   #50
        self.bin_button.config(text=f"Bin = {self.bin_size}")
        
        self.update_plots()


    
    def update_plots(self):
        self.fig_hist.clear()
        ax = self.fig_hist.add_subplot(111)
        max_len = max(self.indexes)
        
        # Erstellen Sie eine Farbkarte mit so vielen Farben wie DataFrames
        cmap = plt.get_cmap('tab20', len(self.dfs))
        
        
        # Fuer jedes DataFrame
        for i, df in enumerate(self.dfs):
            # Erstellen Sie Bins und zaehlen Sie die Anzahl der "Merger" in jedem Bin
            bins = pd.cut(df.index, range(0, max_len , self.bin_size))
            merger_counts = df.groupby(bins)['Merger'].sum()/self.bin_size *100
    
            # Zeichnen Sie die Anzahl der "Merger" in jedem Bin
            merger_counts.plot(kind='bar',subplots = True, ax=ax, color=cmap(i), label=self.user_name[i])
            #merger_counts.plot()
            print(merger_counts)
    
        plt.show()
        # Fuegen Sie eine Legende hinzu
        ax.legend()
    
        self.canvas.draw()
        
    def get_top_9_paths(self):
        # Zaehlen Sie die Anzahl der Eintraege in der Spalte 'Merger' für jeden Pfad
        merger_sort = self.classification.sort_values(by = ['Merger'], ascending = False)[10:19]
    
        return merger_sort['path'].values



        
    def plot_fits_files(self):
        top_9_paths = self.get_top_9_paths()
        #self.canvas_top9.update()

        
        gs = gridspec.GridSpec(3, 3, figure=self.fig_top9)

        for i, path in enumerate(top_9_paths):
            with fits.open(path) as hdul:
                data = hdul[2].data
                vmin = np.percentile(data, 0.5)
                vmax = np.percentile(data, 99.5)
                ax = self.fig_top9.add_subplot(gs[int(i/ 3), i % 3])
                ax.imshow(data, norm='symlog', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title(path.split('/')[-1])
                ax.axis('off')
        
        self.canvas_top9.draw()
        self.canvas_top9.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        #toolbar = NavigationToolbar2Tk(self.canvas_top9, self.master)
        self.toolbar.update()
        



root = tk.Tk()
app = App(root)
root.mainloop()
