# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:25:32 2024

@author: airub
"""

from astropy.io import fits
import pandas as pd
from astropy.table import Table

url_kado = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Kado-Fong_e_und_Greene_je_und_Greco_jp_J_AJ_159_103_table2.dat'
url_Kids_bildzuweisung = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/GAMA/KIDS_ra_dec_cuts_mittelwerte.csv'
speicherort = 'C:/Users/airub/Desktop/Masterarbeit_Marcel/Kado_KIDS_bildzuweisung.fits'
kado = fits.open(url_kado)
#kado.info()
kado_data = kado[1].data
df_kado = pd.DataFrame(kado_data)
df_kids = pd.read_csv(url_Kids_bildzuweisung)


df_kado_subset = df_kado[['ID', 'RAdeg', 'DEdeg', 'zsp', 'logM*', 'InVis']]

# Filtern der relevanten Spalten in df_kids
df_kids_subset = df_kids[['Name', 'Ra_1', 'Ra_2', 'Dec_1', 'Dec_2']]


# Überprüfen der Bedingungen und Filtern der entsprechenden Datensätze
filtered_data = []
for index_kado, row_kado in df_kado_subset.iterrows():
    ra_kado = row_kado['RAdeg']
    dec_kado = row_kado['DEdeg']
    zsp_kado = row_kado['zsp']
    invis_kado = row_kado['InVis']
    logM_kado = row_kado['logM*']
    if index_kado%200==0: print(f'Es sind {index_kado} von {len(df_kado)} erreicht.')
    for index_kids, row_kids in df_kids_subset.iterrows():
        ra1_kids = row_kids['Ra_1']
        ra2_kids = row_kids['Ra_2']
        dec1_kids = row_kids['Dec_1']
        dec2_kids = row_kids['Dec_2']
        if ra1_kids <= ra_kado <= ra2_kids and dec1_kids <= dec_kado <= dec2_kids:
            filtered_data.append({'ID_kado': row_kado['ID'],
                                  'RAdeg': ra_kado,
                                  'DEdeg': dec_kado,
                                  'zsp': zsp_kado,
                                  'InVis': invis_kado,
                                  'logM_star': logM_kado,
                                  'Name_kids': row_kids['Name']})

# Erstellen eines neuen DataFrames aus den gefilterten Daten
df_filtered = pd.DataFrame(filtered_data)

table = Table.from_pandas(df_filtered)
table.write(speicherort, format = 'fits', overwrite=True)

