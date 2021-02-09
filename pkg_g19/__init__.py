#   ######   ########   #######  ##     ## ########        ##    #######  
#  ##    ##  ##     ## ##     ## ##     ## ##     ##     ####   ##     ## 
#  ##        ##     ## ##     ## ##     ## ##     ##       ##   ##     ## 
#  ##   #### ########  ##     ## ##     ## ########        ##    ######## 
#  ##    ##  ##   ##   ##     ## ##     ## ##              ##          ## 
#  ##    ##  ##    ##  ##     ## ##     ## ##              ##   ##     ## 
#   ######   ##     ##  #######   #######  ##            ######  #######  

# ===========================================================================================================
#      Dependancies
# ===========================================================================================================

import numpy as np
import pandas as pd
import os

# ===========================================================================================================
#      General function
# ===========================================================================================================

def data_importer(path_docs, name, number=None, mode='base'):
    # ATTENZIONE è modificato, ho aggiunto la riga successiva
    number = int(number)
    if mode == 'base':
        name_vec = [ ['Name', 'Mass', 'Type'],
                     ['Name1', 'Name2', 'NameCM', 'Mass1', 'Mass2', 'Type1', 'Type2', 'TypeCM'],
                     ['Name1', 'Name2', 'Name3', 'NameCM', 'Mass1', 'Mass2', 'Mass3', 'Type1', 'Type2', 'Type3', 'TypeCM']]
        col_vec  = [ [0, 1, 14],
                     [0, 1, 2, 3, 4, 32, 33, 34],
                     [0, 1, 2, 3, 4, 5, 6, 47, 48, 49, 50]]
    elif mode =='full':
        name_vec = [ ['Name','M','X','Y','Z','Type'],
                     ['Name1', 'Name2', 'NameCM','M1','M2','X','Y','Z','Semi','e','P', 'Type1', 'Type2', 'TypeCM'],
                     ['Name1', 'Name2', 'Name3', 'NameCM', 'M1','M2','M3','X','Y','Z','Type1', 'Type2', 'Type3', 'TypeCM']]
        col_vec  = [ [0,1,2,3,4,14],
                     [0, 1, 2,3,4,5,6,7,18,19,20, 32, 33, 34],
                     [0, 1, 2, 3,4,5,6,7,8,9, 47, 48, 49, 50]]
    
    if name == 'single':
        file_path = path_docs + '/' + name + '.40_' + str(number)
        data = pd.read_csv(file_path, sep='\s+', skiprows= 1, usecols=col_vec[0], header=None, names=name_vec[0])
        return data        
    if name == 'binary':
        file_path = path_docs + '/' + name + '.40_' + str(number)
        data = pd.read_csv(file_path, sep='\s+', skiprows=1, usecols=col_vec[1], header=None, names=name_vec[1])
        return data        
    if name == 'merger':
        file_path = path_docs + '/' + name + '.40_' + str(number)
        data = pd.read_csv(file_path, sep='\s+', skiprows=1, usecols=col_vec[2], header=None, names=name_vec[2])
        return data        
    if name == 'sev':
        file_path = path_docs + '/' + name + '.83_' + str(number)
        data = pd.read_csv(file_path, sep='\s+', skiprows= 1, usecols=[2,3,5], header=None, names=['Name', 'Type', 'Mass'])
        return data    
    if name == 'event':
        file_path = path_docs + '/' + name + '.35'
        data = pd.read_csv(path_docs + name, sep='\s+', skiprows= 1, usecols=[33], header=None, names=['New BH'])
        return data 
    if name == 'bdat':
        file_path = path_docs + '/' + name + '.9_' + str(number)
        data = pd.read_csv(file_path, sep='\s+', skiprows= 4, usecols=[0,1,2,3,10,11], header=None, names=['Name1', 'Name2', 'Mass1', 'Mass2', 'Type1', 'Type2'])
        return data
    if name == 'bwdat':
        file_path = path_docs + '/' + name + '.19_' + str(number)
        data = pd.read_csv(file_path, sep='\s+', skiprows= 4, usecols=[0,1,2,3,10,11], header=None, names=['Name1', 'Name2', 'Mass1', 'Mass2', 'Type1', 'Type2'])
        return data

def iterations_counter(path_docs, name):
    
    '''Counts the number of iterations in each simulation. '''
    
    os.system('ls '+path_docs+' | grep '+name+' > dummy.txt')
    dum = open('dummy.txt')
    content = dum.readlines()
    dum.close()
    os.system('rm dummy.txt')
    
    return len(content)    

def bh_recognition(data, filetype,BBHS=False):
    
    '''Depending on the file, reports the black holes in the data.'''
    if BBHS:
        if filetype == 'binary' or 'bwdat' or 'bdat':
            return  data[[all([a, b]) for a, b in zip(data['Type1']==14, data['Type2']==14)]] 
    else:

        if filetype == 'single' or filetype == 'sev':
            return data[data['Type']==14]
        if filetype == 'binary' or 'bwdat' or 'bdat':
            return  data[[any([a, b]) for a, b in zip(data['Type1']==14, data['Type2']==14)]] 
        if filetype == 'merger':
            return data[[any([a, b, c]) for a, b, c in zip(data['Type1']==14, data['Type2']==14, data['Type3']==14)]]

def folder_recognition(path_docs):
    
    '''Returns a list with the names of the folders.'''
    command = 'ls ' + path_docs + ' | grep ' + '6' + ' | cut -d ' + '"_"' + ' -f 1 > dummy.txt'
    os.system(command)
    dum = open('dummy.txt')
    content = dum.readlines()
    dum.close()
    os.system('rm dummy.txt')
    return list(map(lambda s: s.strip(), content))
