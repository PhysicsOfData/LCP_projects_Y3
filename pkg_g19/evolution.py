# ===========================================================================================================
#      Dependancies
# ===========================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation

#from pkg import data_importer, iteration_counter, bh_recognition, folder_recognition
from pkg_g19 import *

# ===========================================================================================================
#      Functions
# ===========================================================================================================



def evolution(dir_path, file, step, out='',BBHS=False):
    folders_list = folder_recognition(dir_path)
    n_fol = len(folders_list)
    max_iter = np.zeros(n_fol)
    for fol in range(n_fol):
            path = dir_path + folders_list[fol] + '_D1.6_Z0.0002'
            max_iter[fol] = iterations_counter(path, file)

    stars_data = pd.DataFrame()

    grid = pd.DataFrame()
    years = np.linspace(0,100,step) 
    for i in range(step):
        
        #Abbiamo aggiunto questa linea di codice------------------
        stars_data = pd.DataFrame()
        #----------------------------------------------------------
        
        for fol,ix in zip(folders_list,max_iter):
            # -------qua sotto
            curr_iter=round(np.linspace(0,ix-1,step)[i])
            #-------------------------
            path = dir_path + fol + '_D1.6_Z0.0002'
                
            data = data_importer(path, file, curr_iter)
            data = bh_recognition(data, file,BBHS=BBHS)
          
            stars_data = stars_data.append(data, ignore_index=True)

            data_binary = stars_data
        
        mask = data_binary['Type1'] == 14
        df1 = data_binary[mask]['Mass1']
        mask = data_binary['Type2'] == 14
        df2 = data_binary[mask]['Mass2']
        binary_masses = pd.concat([df1, df2], ignore_index=True)
        if len(binary_masses) > 0:
            binary_masses.name = years[i]
            grid = grid.append(binary_masses.T, ignore_index=False)
        else:
            a = pd.Series()
            a.name = years[i]
            grid = grid.append(a, ignore_index=False) # to add a Nan line if there's no bh
    if out != '':
        grid.to_csv(out)
    return grid

#def evo_plotter(data,bins,h,b):
def evo_plotter(time, mass, grid, ax):
    #bin_edge   = np.linspace(0, data.max().max()+data.min().min(),bins+1)
    #bin_center = (bin_edge[:-1]+bin_edge[1:])/2
    #n_iter = (np.shape(data.values)[0])
    #grid = np.zeros((n_iter,bins))
    #for i in range(n_iter):
    #    sample = data.iloc[i].values
    #    sample = sample[np.isfinite(sample)]
    #    if len(sample) != 0:
    #        grid[i], _ = np.histogram(sample, bins=bin_edge, density=True)
    
    #fig = plt.figure(figsize=(b,h))
    #ax = plt.contourf(data.index.values,bin_center,grid.T,levels=200, cmap='inferno')
    prova = ax.contourf(time,mass,grid.T,levels=200, cmap='inferno')
    ax.set_xlabel('Time [Mys]')
    ax.set_ylabel('Mass $[M_\odot]$')
    plt.show()
    return prova
    
#def evo_plotter3D(data,bins,h,b):
def evo_plotter3D(time, mass, grid, ax):
    #bin_edge   = np.linspace(0, data.max().max()+data.min().min(),bins+1)
    #bin_center = (bin_edge[:-1]+bin_edge[1:])/2
    #n_iter = (np.shape(data.values)[0])
    #grid = np.zeros((n_iter,bins))
    #for i in range(n_iter):
    #    sample = data.iloc[i].values
    #    sample = sample[np.isfinite(sample)]
    #    if len(sample) != 0:
    #        grid[i], _ = np.histogram(sample, bins=bin_edge, density=True)
            
    #fig = plt.figure(figsize=(b,h))    
    #ax = plt.axes(projection='3d')
    #ax.axes(projection='3d')
    #X, Y = np.meshgrid(bin_center,data.index.values)
    X, Y = np.meshgrid(mass,time)
    ax.contour3D(X,Y,grid, 100, cmap='inferno', alpha=0.8) 
    ax.view_init(20,60)
    ax.set_xlabel('Mass $[M_\odot]$')
    ax.set_ylabel('Time [Mys]')
    ax.set_zlabel('Counts',rotation=270)
    plt.show()
    

def mass_distribution(data, bins,density=False):
    '''Function to calculate black holes distribution at every time step:
        - data = black holes masses at every time step
        - bins = number on bars in the histogram'''
    bin_edge   = np.linspace(0, data.max().max()+data.min().min(),bins+1)
    bin_center = (bin_edge[:-1]+bin_edge[1:])/2
    n_iter = (np.shape(data.values)[0])
    grid = np.zeros((n_iter,bins))
    for i in range(n_iter):
        sample = data.iloc[i].values
        sample = sample[np.isfinite(sample)]
        if len(sample) != 0:
            grid[i], _ = np.histogram(sample, bins=bin_edge, density=density)
    times= data.index.values
    return bin_center, times, grid

def mass_animation(times, bin_center,  grid,log_scale=True):
    '''Animates the black holes distribution over time:
        - times = time steps
        - bin_center = central values of the bars in the histogram
        - grid = black holes distribution over time'''
    
    fig, ax = plt.subplots()
    
    width=bin_center[1]-bin_center[0]
    y_max = np.amax(grid)*1.1
    
    
    def init_hist():
    #The first frame conditions
        init_plot=plt.bar(bin_center, height=grid[0,:],width=width, color ='navy')
        plt.title('t = %4.f' % 0)
        plt.xlabel('(Binary) BH mass[M*]')
        plt.ylabel('Number of Black holes')
        if log_scale:
            ax.set_yscale('log')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0.00001,y_max))
        return init_plot

    def update_hist(frame):
        plt.cla()
        plot=plt.bar(bin_center, height=grid[frame,:], width=width, color ='navy')
        plt.title('t = %4.f [Myr]'% times[frame])
        plt.xlabel('(Binary) BH mass[$M_{\odot}$]')
        plt.ylabel('Number of Black holes')
        if log_scale:
            ax.set_yscale('log')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0.00001,y_max))
        return plot

    anim = animation.FuncAnimation(fig, update_hist, frames=range(0,times.size-1), init_func=init_hist, blit=True)

    return anim
