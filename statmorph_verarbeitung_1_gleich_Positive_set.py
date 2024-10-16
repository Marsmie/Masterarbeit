# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:16:36 2024

@author: airub
"""


"""

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.modeling.models import Sersic2D
from astropy.convolution import convolve, Gaussian2DKernel
from photutils.segmentation import detect_threshold, detect_sources
import time
import statmorph
import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
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

    
def labeln(df, label):
    lst_y_train = []
    for row in df.index:
        if str(row[0]) == label:
            lst_y_train.append(1)
        else:
            lst_y_train.append(0)
            
    df_y = pd.DataFrame(lst_y_train)
    return df_y   

def absolute_maximum_scale(series):
    return series / series.abs().max()

def normalize_df(df):
    for col in df.columns:
        df[col] = absolute_maximum_scale(df[col])
        

url = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/statmorph/'
lst = os.listdir(url)

lst_CAM = ['OCAM_u_SDSS', 'OCAM_g_SDSS', 'OCAM_r_SDSS', 'OCAM_i_SDSS']


def filter_select(lst, filter_, Name = False):
    df_return = pd.DataFrame()
    for cnt, entry in enumerate(lst):
        prefix = entry.split('_')[0]
        filt = filter_.split('_')[1]
        
        if prefix == 'Paudel':
            nummer = int(entry.split('_')[1].split('.')[0][2:].split('-')[0]+
                         entry.split('_')[1].split('.')[0][2:].split('-')[1])

        else:
            nummer = int(entry.split('_')[1].split('.')[0])
        
    
        df_new = pd.DataFrame()
        df = pd.read_csv(url + entry)
        df = df.where(df['Filter_ID']== filter_).dropna()
        df.index = [nummer]
        
        
        if Name:
            df = df.drop(columns = ['Unnamed: 0', 'Filter_ID']) #'Quality',
            df.columns = ['Namen', f'C_{filt}', f'A_{filt}', f'S_{filt}', f'G_{filt}',
                          f'M20_{filt}', f'r20_{filt}', f'r80_{filt}', f'Q_{filt}']
            df['Name'] = prefix
        else:
            df = df.drop(columns = ['Unnamed: 0', 'Filter_ID', 'Name']) #'Quality',
            df.columns = [ f'C_{filt}', f'A_{filt}', f'S_{filt}', f'G_{filt}',
                          f'M20_{filt}', f'r20_{filt}', f'r80_{filt}', f'Q_{filt}']
            
            
        df = df.T
        for i in range(0,len(df)):
            #print(i)
            df_new = pd.concat([df_new, df.iloc[i]], axis = 1)
        df_return = pd.concat([df_return, df_new], axis =0)
        
    return df_return
        
df_u = filter_select(lst, lst_CAM[0], True)
df_g = filter_select(lst, lst_CAM[1], False)
df_r = filter_select(lst, lst_CAM[2], False)
df_i = filter_select(lst, lst_CAM[3], False)
    
df_test = pd.concat([df_u, df_g, df_r, df_i], axis = 1)

df_DR3 = df_test.where(df_test['Name'] == 'DR3').dropna()
df_Vorontsov = df_test.where(df_test['Name'] == 'Vorontsov').dropna()
df_Kado = df_test.where(df_test['Name'] == 'Kado').dropna()
df_Paudel = df_test.where(df_test['Name'] == 'Paudel').dropna()


df_DR3 = df_DR3.drop(columns = ['Name', 'Namen'])
df_Kado = df_Kado.drop(columns = ['Name', 'Namen'])
df_Paudel = df_Paudel.drop(columns = ['Name', 'Namen'])
df_Vorontsov = df_Vorontsov.drop(columns = ['Name', 'Namen'])

df_Index = pd.concat([df_Paudel, df_Vorontsov, df_DR3, df_Kado], keys = ['Paudel', 'Voro', 'DR3', 'Kado'])
df_Index_ohne_Voro = pd.concat([df_Paudel, df_DR3, df_Kado], keys = ['Paudel',  'DR3', 'Kado'])

"""
df_Index = df_Index.drop(columns = ['r20_g', 'r80_g','r80_r'])

df_Index = df_Index.drop(columns = ['C_u', 'G_u', 'M20_u', 'r20_u', 'r80_u',
                         'C_g','G_g', 'M20_g', 'r20_g', 'r80_g',
                         'C_r','G_r', 'M20_r', 'r20_r', 'r80_r',
                         'C_i','G_i', 'M20_i', 'r20_i', 'r80_i'])

df_Index = df_Index.drop(columns = ['C_u','S_u', 'G_u',
                         'C_g','S_g', 'G_g','M20_g', 
                         'S_r','G_r', 
                         'S_i'])

df_Index = df_Index.drop(columns = ['C_u','S_u', 'G_u','M20_u', 'r80_u',
                         'C_g','S_g', 'G_g','M20_g', 
                         'A_r','S_r','G_r', 
                         'C_i','S_i', 'G_i'])
"""
df_Index_selfmade = pd.concat([df_Paudel, df_DR3, df_Kado], keys = ['Paudel', 'DR3', 'Kado'])

df_Index_reduced_to_A_and_S = df_Index_selfmade.drop(columns = ['C_u', 'G_u', 'M20_u', 'r20_u', 'r80_u',
                         'C_g','G_g', 'M20_g', 'r20_g', 'r80_g',
                         'C_r','G_r', 'M20_r', 'r20_r', 'r80_r',
                         'C_i','G_i', 'M20_i', 'r20_i', 'r80_i'])


df_Index_reduced_12 = df_Index.drop(columns = ['C_u', 'S_u', 'G_u', 'r80_u', 'C_g', 'S_g', 'G_g', 'M20_g',
                                              'S_r', 'G_r', 'C_i', 'G_i'])

df_Index_reduced_19 = df_Index_reduced_12.drop(columns = ['C_r', 'A_r', 'M20_r', 'r20_r', 'S_i', 'M20_i', 'r20_i'])

#df_Index_reduced_16 = df_Index_reduced_12.drop(columns = ['M20_u','r80_u' ,'A_r', 'r20_r'])

#df_Index = df_Index_reduced_16

df_Index_A_S = df_Index.drop(columns = ['C_u', 'G_u', 'M20_u', 'r20_u', 'r80_u',
                         'C_g','G_g', 'M20_g', 'r20_g', 'r80_g',
                         'C_r','G_r', 'M20_r', 'r20_r', 'r80_r',
                         'C_i','G_i', 'M20_i', 'r20_i', 'r80_i'])



def train_Forest(df_Index, class_type, save_name, second_class = '', feature_search = True):
    
    test_bool = df_Index.isin([-999])
    #test_drop = test_bool.where(test_bool ==False).dropna() #erzeugt DataFrame gefüllt mit False von Einträgen ohne -999
    test_Index = df_Index.where(test_bool ==False).dropna()
    test_Index = test_Index.drop(columns = ['Q_u', 'Q_g', 'Q_r', 'Q_i'])
    test_bool = test_Index.isin([-99])
    #test_drop = test_bool.where(test_bool ==False).dropna()
    test_Index = test_Index.where(test_bool ==False).dropna()
    #test_Index.to_csv('C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/statmorph_df_Index_ohne_99.csv')
    normalize_df(test_Index)
    
    #df_label = test_Index.index
    
    df_y = labeln(test_Index, class_type)
    df_y = df_y + labeln(test_Index, second_class)
    
    
    
    #####################################
    
    clf = RandomForestClassifier(criterion= "gini", 
                                 max_depth = 4,
                                 random_state= 42,
                                 class_weight='balanced', 
                                 n_jobs = 16)
    
    arr_y = np.array(df_y)
    arr_y = arr_y.flatten()
    
    if feature_search:
        clf = clf.fit(test_Index, df_y)
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
        
        forest_importances = pd.Series(importances, index=test_Index.columns)
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
    
    cv = len(arr_y) - np.sum(arr_y) -1
    
    #cv = 13
    
    Y_post = cross_val_predict(clf, test_Index, arr_y, cv=cv, method = 'predict_proba')
    Y_proba = pd.DataFrame(Y_post, index = test_Index.index)
    #######################################
    Index_proba = pd.concat([test_Index.head(len(Y_proba)), Y_proba], axis = 1)
    Index_proba_sorted = pd.DataFrame(Index_proba.sort_values(1, ascending = False).values, columns = Index_proba.columns, index = test_Index.index)
    #save_file = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/Explizit_test/statmorph_Paudel_Voro_Kado_DR3_trainiert_auf_Paudel_all_feature.csv'
    save_file = save_name
    Index_proba.to_csv(save_file)
    #DM_proba_sorted = Index_proba_sorted.where(Index_proba_sorted['Unnamed: 0'] == 'DM').dropna()

    Index_represent = pd.read_csv(save_file)
    Index_represent = Index_represent.drop(columns = 'Unnamed: 1')
    Index_represent.hist('1', by = 'Unnamed: 0')
    plt.show()
    
    return Y_proba, Index_represent


#Y_proba , Index_represent = train_Forest(df_Index, 'Paudel', save_name = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/solo_test/statmorph_Paudel_Voro_Kado_DR3_trainiert_auf_Voro_all_feature.csv')



#Y_proba, Index_represent = train_Forest(df_Index, 'Voro', save_name = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/solo_test/statmorph_Paudel_Voro_Kado_DR3_trainiert_auf_Paudel_all_feature.csv')

#Y_proba, Index_represent = train_Forest(df_Index, 'Voro', second_class = 'Paudel',save_name = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/solo_test/statmorph_Paudel_Voro_Kado_DR3_trainiert_auf_VorouPaudel_all_feature.csv')

#Y_proba_with_a_and_s, Index_A_S = train_Forest(df_Index_reduced_to_A_and_S, 'Voro')

def add_merger_statistic(df):
    #Merger Statistic S(g,M20) = 0.139*M20 + 0.990*G - 0.327
    lst = ['u', 'g', 'r', 'i']
    for i in lst:
        G = df[f'G_{i}']
        M20 = df[f'M20_{i}']
        
        S = 0.139 * M20 + 0.990 * G -0.327
        
        df[f'S(g,m20)_{i}'] = S
    
    
    return df

def plot_G_M20(df):
    lst = ['u', 'g', 'r', 'i']
    for i in lst:
        G = df[f'G_{i}']
        M20 = df[f'M20_{i}']
        plt.plot(G, M20, '*')
        plt.show()
        
    
    
    


drop_lst = [204, 206, 459,460,843,851,852,853,855,1605,2076,2088,2339,2340,2341,3409,3410,3411]

df_Voro_drop = df_Vorontsov.drop(axis = 'index', labels = drop_lst)

df_Index_drop =  pd.concat([df_Paudel, df_Voro_drop, df_DR3, df_Kado], keys = ['Paudel', 'Voro', 'DR3', 'Kado'])

df_Index_drop_reduced12 = df_Index_drop.drop(columns = ['C_u', 'S_u', 'G_u', 'C_g', 'S_g', 'G_g', 'M20_g', 
                                                      'C_r', 'A_r', 'S_r', 'G_r', 'S_i', ])

df_Index_drop_reduced18 = df_Index_drop_reduced12.drop(columns = ['M20_u', 'r80_u', 'M20_r', 'C_i', 'G_i', 'r20_i'])

Y_proba_Paudel , Index_represent_Paudel = train_Forest(df_Index_reduced_19, 'Paudel', save_name = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/final/final_statmorph_Paudel_Kado_DR3_trainiert_auf_Paudel_reduced_19_feature.csv')

Y_proba_Voro , Index_represent_Voro = train_Forest(df_Index_reduced_19, 'Voro', save_name = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/final/final_statmorph_Paudel_Voro_Kado_DR3_trainiert_auf_Voro_reduced_19_feature_.csv')#Y_proba_Paudel_Voro , Index_represent_Paudel_Voro = train_Forest(df_Index_reduced_16, 'Paudel',second_class = 'Voro', save_name = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/solo_test/drop_statmorph_Paudel_Voro_Kado_DR3_trainiert_auf_Voro_reduced16_feature.csv')

#Y_proba_Paudel_u_Voro , Index_represent_Paudel_u_Voro = train_Forest(df_Index, 'Voro', second_class = 'Paudel', save_name = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/final/final_statmorph_Paudel_Voro_Kado_DR3_trainiert_auf_Paudel_u_Voro_all_feature_all.csv')#Y_proba_Voro , Index_represent_Voro = train_Forest(df_Index_reduced_16, 'Voro', save_name = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Dataset/RandomForest/solo_test/drop_statmorph_Paudel_Voro_Kado_DR3_trainiert_auf_Voro_reuced16_feature.csv')


