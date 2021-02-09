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

def cluster_importer(path_docs,start=1,step=1,mode='step'):
    
    '''Function to import an entire star cluster information at every timestep.
    
    RETURNS:: 3 python lists containing the single, binary and merger information at every step.
    
    start and step arguments: self explanatory
    
    
    
        '''
    

    #path_docs = '/data_2019/stellar_blackholes/M6001_D1.6_Z0.0002'
    #name1 = '/single.40_'
    name2 = 'binary'
    name3 = 'merger'

    #path_docs = '/data_2019/stellar_blackholes/M6001_D1.6_Z0.0002'
    name1 = 'single.40_'

    frames=iterations_counter(path_docs, name1)-1
    
    #iterations=np.arange(start,frames,step=step)
    
    iterations=[int(round(j)) for j in np.linspace(start,frames,step)]
    
    n=len(iterations)
    #print(iterations)
    ds=[[]]*n
    db=[[]]*n
    dm=[[]]*n
    T=[it*(100/frames) for it in iterations]
    #print(iterations)
    

    name1 = 'single'

    j=0
    for i in iterations:
        ds[j] = data_importer(path_docs, name1, i, 'full')
        db[j] = data_importer(path_docs, name2, i, 'full')
        try:
            dm[j] = data_importer(path_docs, name3, i)
        except:
            dm[j]=pd.DataFrame(columns=['Name1', 'Name2', 'Name3', 'NameCM', 'M1','M2','M3','X','Y','Z','Type1', 'Type2', 'Type3', 'TypeCM'])
        j+=1
        
    return ds,db,dm,T

def bh_finder(i,data,mode='3df'):
    
    '''RETURNS:: 3 or 1 dataframe/s containing just the BH information on the i-th iteration of the cluster
    
        MODES::
        
            '3df'   :: returns 3 dataframes, one for singles, binaries and mergers each
            '1df'   :: returns 1 dataframe with info from single, binary and merger
            'binary':: returns 1 dataframe containing just binary BH systems
                        
                        
                        '''
    ds=data[0]
    db=data[1]
    dm=data[2]
    T =data[3]
    
    bhs=ds[i]['Type']==14
    bhd1=db[i]['Type1']==14
    bhd2=db[i]['Type2']==14
    bhm1=dm[i]['Type1']==14
    bhm2=dm[i]['Type2']==14
    bhm3=dm[i]['Type3']==14
    
    if mode=='binary' :
        js=[]
    
        for j in range(len(bhd1)):
            #print(s1)
            if (bhd1[j] and bhd2[j]) :
                js.append(j)
                
        return db[i].iloc[js]
                
                
                
    elif mode=='1df':
        bh1=ds[i][['Name','M','X','Y','Z']].loc[bhs]
        bh1=bh1.append(db[i][['Name1','M1','X','Y','Z']].loc[bhd1].rename({'Name1':'Name','M1':'M'},axis=1))
        bh1=bh1.append(db[i][['Name2','M2','X','Y','Z']].loc[bhd2].rename({'Name2':'Name','M2':'M'},axis=1))
        bh1=bh1.append(dm[i][['Name1','M1','X','Y','Z']].loc[bhm1].rename({'Name1':'Name','M1':'M'},axis=1))
        bh1=bh1.append(dm[i][['Name2','M2','X','Y','Z']].loc[bhm2].rename({'Name2':'Name','M2':'M'},axis=1))
        bh1=bh1.append(dm[i][['Name3','M3','X','Y','Z']].loc[bhm3].rename({'Name3':'Name','M3':'M'},axis=1))
        return bh1
       
                

    else:
        bh1=ds[i].loc[bhs]
        bh2=db[i].loc[bhd1]
        bh2=bh2.append(db[i].loc[bhd2])
        bh3=dm[i].loc[bhm1]
        bh3=bh3.append(dm[i].loc[bhm2])
        bh3=bh3.append(dm[i].loc[bhm3])
    
        return bh1,bh2,bh3
    
    
    

def cluster_plotter(i,data,ax1='X',ax2='Y',lims=[-50,50,-50,50],size=[5,5],ax=None,BH_plotter=True):
    ''' Simple function to plot the state of the star cluster at any given time.
    Blue stars are singles, Red stars are binaries and mergers objects.
    
    Black '+' are singles black holes, while black  'x' are black hole in binaries and mergers.
    
    Red 'x' means a binary black hole  
    
    arguments:
    
    -- i:
    time iterate that we want to plot
    
    -- ax1='X', ax2='Y'
    axes that we care about
    
    --lims=[-50,50,-50,50]
    limits of the plot 
    
    --size=[5,5]
    equal to figsize arg in matplotlib 
    
    ax=None
    axis object from matplotlib
    
    
    
    
    
    '''
    
    ds=data[0]
    db=data[1]
    dm=data[2]
    T =data[3]
    
    if ax==None:
        fig,ax=plt.subplots(figsize=size)
    #lims=50
    
    size1=1
    size2=150



    ax.scatter(ds[i][ax1],ds[i][ax2],color='lightyellow',s=size1*ds[i]['M'],alpha=0.8,marker='.')
    ax.scatter(db[i][ax1],db[i][ax2],color='lightyellow', s=size1*db[i]['M1'],alpha=0.8,marker='.')
    if BH_plotter:
        
        bh=bh_finder(i,mode='1df',data=data)
        bh_binary=bh_finder(i,mode='binary',data=data)
        ax.scatter(bh[ax1],bh[ax2],color='gold',s=size2,marker='X')
        ax.scatter(bh_binary[ax1],bh_binary[ax2],color='red',s=size2*1.5,marker='X')


    
    ax.set_xlabel(ax1 +' [PC]')
    ax.set_ylabel(ax2+ ' [PC]')
    ax.set_xlim(lims[0],lims[1])
    ax.set_ylim(lims[2],lims[3])
    ax.set_title('T= '+("%.2f" % T[i]) +' Mys')
    #fig.patch.facecolor('white')
    ax.set_facecolor([0.11,0.11,0.11])
    return ax
    #plt.scatter(db[i][ax1],db[i][ax2],color='red',s=0.1)
    
    
def collapse_time(bbh):
    G=6.674*10**(-11)
    c=3*10**8
    k=5/256
    
    return k*c**5*(bbh['Semi']*149*10**9)**4*(1-bbh['e']**2)**(7/2)/((1.98*10**30)**3*G**3*bbh['M1']*bbh['M2']*(bbh['M1']+bbh['M2']))

def cluster_animator(ax1, ax2, ds, db, dm, T):
    '''Animates the evolution of the star cluster:
    - ax1, ax2 = axes to plot
    - ds, dm, db, T = data of single, binary, merger and time to import'''

    ax1='X'
    ax2='Y'
    lims=[-50,50,-50,50]
    size=[5,5]

    fig, ax = plt.subplots(figsize=size)
    ax.set_xlabel(ax1 +' [PC]')
    ax.set_ylabel(ax2 + ' [PC]')
    ax.set_xlim(lims[0],lims[1])
    ax.set_ylim(lims[2],lims[3])
    ax.set_facecolor([0.11,0.11,0.11])


#Lines (matplotlib objects) to modify in the update function

    datasingle, = plt.plot([], [], 's', markersize=1, color='lightyellow', alpha=0.1)
    databinary, = plt.plot([], [], 's', markersize=1, color='white', alpha=0.1)
    datamerger, = plt.plot([], [], 'x', markersize=5, color='red')

#First frame
    def init():  
        ax.set_title('T = %4.2f Mys' % T[0])
        x = np.array(ds[0][ax1])
        y = np.array(ds[0][ax2])
        datasingle.set_data(x, y)
        x = np.array(db[0][ax1])
        y = np.array(db[0][ax2])
        databinary.set_data(x,y)
        x = np.array(dm[0][ax1])
        y = np.array(dm[0][ax2])
        datamerger.set_data(x,y)
        return (datasingle,) + (databinary,) + (datamerger,)

#Update function
    def update(i):
        ax.set_title('T = %4.2f Mys' %T[i])
        x = np.array(ds[i][ax1])
        y = np.array(ds[i][ax2])
        datasingle.set_data(x, y)
        x = np.array(db[i][ax1])
        y = np.array(db[i][ax2])
        databinary.set_data(x, y)
        x = np.array(dm[i][ax1])
        y = np.array(dm[i][ax2])
        datamerger.set_data(x,y)

        return (datasingle,) + (databinary,) + (datamerger,)


    ani = animation.FuncAnimation(fig, update, init_func=init, frames=range(0,300, 10), blit=True)
    return ani