# ===========================================================================================================
#      Dependancies
# ===========================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#from pkg import data_importer, iteration_counter, bh_recognition, folder_recognition
from pkg_g19 import *


# ===========================================================================================================
#      Functions
# ===========================================================================================================

def stars_survey(path_docs, file, N = 2, final_masses = False, black_holes=False):
    
    """path_docs = folder with all the data
    file = file selected
    N = iteration
    final_iteration = takes the last iteration
    black_holes = only black holes survey"""
    
    folders_list = folder_recognition(path_docs)
    stars_data = pd.DataFrame()
    
    for fol in folders_list:
        path = path_docs + '/' + fol + '_D1.6_Z0.0002'
        
        if final_masses == True:
            N = iterations_counter(path, file)
        
        data = data_importer(path, file, N-1)
        
        if black_holes == True:
            data = bh_recognition(data, file)
        
        stars_data = stars_data.append(data, ignore_index=True)

    return stars_data

def mass_wrapper(data_single, data_binary, data_merger):
    '''Some further manipulation of the data, with the aim of determining the mass distribution.
    Returns:
    - an one-column dataframe with the masses of the single black holes;
    - an one-column dataframe with the masses of the black holes in binary systems;
    - an one-column dataframe with the masses of the black holes in merger systems;
    - an one-column dataframe with all the masses of the black holes.'''
    # Filtering the black holes inside the binary dataframe
    mask = data_binary['Type1'] == 14
    df1 = data_binary[mask]['Mass1']
    mask = data_binary['Type2'] == 14
    df2 = data_binary[mask]['Mass2']
    
    binary_masses = pd.concat([df1, df2], ignore_index=True)

    
    #Same but with mergers

    mask = data_merger['Type1'] == 14
    df1 = data_merger[mask]['Mass1']
    mask = data_merger['Type2'] == 14
    df2 = data_merger[mask]['Mass2']
    mask = data_merger['Type3'] == 14
    df3 = data_merger[mask]['Mass3']

    merger_masses = pd.concat([df1, df2, df3], ignore_index=True)

    # All masses dataframe
    #all_masses = pd.concat([data_single['Mass'], binary_masses, merger_masses], ignore_index = True)
    return data_single['Mass'], binary_masses, merger_masses
    
def histplotter(histlist, histlabel, mode):
    plt.figure(figsize=(7,7))

    if mode == 'all':
        plt.hist(histlist, bins = 10, label = histlabel, stacked=True)
        plt.title('Mass distribution of the black holes at T = 100 Mys')
    if mode == 'binary':
        plt.hist(histlist, bins = 10, label = histlabel)
        plt.title('Focus on binary systems mass distribution at T = 100 Mys')
        
    plt.xlabel('Mass $[M_\odot]$')
    plt.ylabel('Counts')
    plt.legend()

def Mass1_vs_Mass2(binary_bhs_tot,specialBhs):
    
    
    LigoVirgo = {'M1':  [35, 50.2,14.2,31.9, 12,30.5,23.3,35.2,35.5,39.6,85,24.5,29.7,33.4,45.4,40.6,39.5 ],
             'M2':  [30, 34  ,7.5 ,19.4, 7 ,25.3,13.6,23.2,26.8,29.4,66,18.3,8.4 ,23.4,30.9,31.4,31.0],
        }

    df = pd.DataFrame (LigoVirgo, columns = ['m1','m2'])

    expelled     = binary_bhs_tot[binary_bhs_tot['Ejected']==True]
    not_expelled = binary_bhs_tot[binary_bhs_tot['Ejected']==False]

    ne=len(expelled)
    M1s_expelled=np.zeros(ne)
    M2s_expelled=np.zeros(ne)

    nn=len(not_expelled)
    M1s_not=np.zeros(nn)
    M2s_not=np.zeros(nn)

    for i in range(ne):
        m1=expelled['M1'].iloc[i]
        m2=expelled['M2'].iloc[i]
        if m1>m2: M1s_expelled[i],M2s_expelled[i]=m1,m2
        else :    M1s_expelled[i],M2s_expelled[i]=m2,m1


    for i in range(nn):
        m1=not_expelled['M1'].iloc[i]
        m2=not_expelled['M2'].iloc[i]
        if m1>m2: M1s_not[i],M2s_not[i]=m1,m2
        else :    M1s_not[i],M2s_not[i]=m2,m1

    fig, ax= plt.subplots(figsize=[10,10])

    ax.grid()

    ax.plot([0,140],[0,140],color='red',linestyle='-.',linewidth=0.3)

    ax.scatter(M1s_not,M2s_not,c='black',s=10,label='Simulated BBHs still in SCs')
    ax.scatter(M1s_expelled,M2s_expelled,c='blue',s=10,label='Simulated BBHs expelled from SCs')
    ax.scatter(LigoVirgo['M1'],LigoVirgo['M2'],s=100,c='red',marker='+',label='Ligo and Virgo detections')
    ax.plot(LigoVirgo['M1'][10],LigoVirgo['M2'][10], marker='o',c='green', fillstyle='none',markersize=40)
    plt.title('Primary mass VS secondary mass  at T= 100 Mys ')
    plt.xlim(0,140)
    plt.ylim(0,140)
    plt.xlabel('M1 [M $\odot$]')
    plt.ylabel('M2 [M $\odot$]')
    plt.legend()
    ax.set_facecolor('lightyellow')

    plt.show()
