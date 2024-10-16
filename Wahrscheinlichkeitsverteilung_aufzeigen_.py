# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:25:09 2024

@author: airub
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
import pandas as pd

Local_DR4_speicher_url = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/Local_DR5_proba_speicher/*'


Local_DR4_link = glob.glob(Local_DR4_speicher_url)

#test_pd = pd.read_csv(Local_DR4_link[4]).drop(columns = 'Unnamed: 0.1')

list_index = [1, 2000, 20000, 200000, 2000000]#, 10000000, 13406500]
list_index = [1, 1000, 2000, 5000, 10000, 20000]


for i in list_index:
    proba_liste = pd.read_csv(f'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/Local_DR5_proba_speicher_30_08/Index_proba_sorted_bei_DR5_von_{i}.csv').drop(columns = 'Unnamed: 0.1')
    #proba_liste.boxplot(column = '0', by= 'Unnamed: 0', rot = 0)
    proba_liste.hist(column = '0', by= 'Unnamed: 0', xrot = 0)
 
    
    DM_proba_sorted = proba_liste.where(proba_liste['Unnamed: 0'] == 'DM').dropna()
    obj_proba_sorted = proba_liste.where(proba_liste['Unnamed: 0'] == 'DR3').dropna()
    kado_proba_sorted = proba_liste.where(proba_liste['Unnamed: 0'] == 'Kado-Fong').dropna()
    dr4_proba_sorted = proba_liste.where(proba_liste['Unnamed: 0'] == 'DR5').dropna()
    dr4_proba_null = dr4_proba_sorted.where(dr4_proba_sorted['0'] == 0).dropna()
    
    #dr4_proba_sorted['0'].plot(kind = 'hist', grid= True, logy = True)
    plt.show()
   
    
    #index_plot = pd.concat([pd.DataFrame(DM_proba_sorted.index), pd.DataFrame(obj_proba_sorted.index), pd.DataFrame(kado_proba_sorted.index), pd.DataFrame(dr4_proba_sorted.index)], keys = ['DM', 'obj', 'kado', 'dr4'], axis = 0)
    #index_plot.plot.hist()

#dr4_proba_null.to_csv('C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/Local_DR4_proba_speicher/DR4_proba_null.csv')
#dr4_proba_sorted.hist(column = '0')
#dr4_proba_sorted['0'].plot(kind = 'hist', grid = True, logy = True)
    
