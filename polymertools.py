# -*- coding: utf-8 -*-
"""

Tools designed to make post-processing of molecular simulations simpler

@author: Michael Grant
"""

import pandas as pd
import numpy as np

def dump_file_to_dataframe(dumpfile, column_headers):
    
    """ 
    Read in a simulation dumpfile (LAMMPs) and output 1 dataframe with each timestep
    
    Arguments:
        
        dumpfile (str): The name of the LAMMPS dump file to read
        column_headers (list): The names of the desired pandas dataframe column headers 
    
    """
    data = []
    
    standard_columns = ['Timestep', 'Entry']
    
    # Create an empty dataframe
    column_headers[0:0] = standard_columns
    
    # Define if we are at the beginning of a timestep or actively reading data
    new_frame = True
    
    with open(dumpfile, 'r') as file:
        
        #Ignore first three lines
        for line in file.readlines()[3:]:
            line = line.strip().replace("\n", "").split(" ")
            
            #The first line read is the 0th timestep and the number of entries
            #We can use the number of entries to find the location of the next frame
            
            if new_frame == True:
                timestep = line[0]
                entries = int(line[1])
                new_frame = False
                counter = 1
                continue
                
            else:
                if counter < entries:
                    line.insert(0, timestep)
                    data.append(line)
                    counter += 1
                elif counter == entries:
                    line.insert(0, timestep)
                    data.append(line)
                    new_frame = True
                    
    return pd.DataFrame(data, columns=column_headers)
    
def trj_file_to_dataframe(trjfile, column_headers, sort=[]):
    
    """ 
    Read in a simulation trajectory (LAMMPs) and output 1 dataframe with each timestep
    
    Arguments:
        
        dumpfile (str): The name of the LAMMPS dump file to read
        column_headers (list): The names of the desired pandas dataframe column headers 
        sort (list): Optional - Declares how you wish the dataframe to be organized after creation
        
    Returns:
        
        A Pandas DataFrame with the desired columns as well as box dimensions and timesteps
    
    """
    
    data = []
    
    standard_columns = ['Timestep', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi']
    
    column_headers[0:0] = standard_columns
    
    with open(trjfile, 'r') as file:
        for line in file:

            line = line.strip().replace("\n", "")

            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(next(file))
                    
            elif line.startswith("ITEM: NUMBER"):   
                next(file)
                
            elif line.startswith("ITEM: BOX BOUNDS"):
                xlo, xhi = [float(i) for i in next(file).split(" ")]
                ylo, yhi = [float(i) for i in next(file).split(" ")]
                zlo, zhi = [float(i) for i in next(file).split(" ")]
                

            elif line.startswith("ITEM: ATOMS"):
                continue

            else:
                line = line.split(" ")
                line[0:0] = [timestep, xlo, xhi, ylo, yhi, zlo, zhi]
                data.append(line)
        
    return pd.DataFrame(data, columns=column_headers).sort_values(by=sort).reset_index(drop=True)
            
def unscale_dataframe(df):
    
    """
    Takes a normalized data frame (must contain box dimensions) and returns true coordinates
    Atoms are still forced into the periodic box and should have image flags. Keeping the 
    wrapping allows this function to be used for calculations stuch as structure factors.
    
    Arguments:
        dataframe (DataFrame): the pandas dataframe that contains the trajectory
        This dataframe must contain the 6 box dimensions as well as scaled coordinates.
        
    Returns:
        A dataframe with the timestep, molecule information x,y,z coordinates as well as image flags
    """
    
    #Create 3 new columns that are the unscaled coordinates
    df['x'] = (df['xhi']-df['xlo']) * df['xs'].astype(float)
    df['y'] = (df['yhi']-df['zlo']) * df['ys'].astype(float)
    df['z'] = (df['zhi']-df['ylo']) * df['zs'].astype(float)

    #Copy the dataframe to a new dataframe ignored the boundaries and scaled coordinates
    return df.drop(['xs', 'ys', 'zs'], axis=1)

def unwrap_dataframe(df):
    """
    Takes a dataframe (must contain image flags and box dimensions) and returns true coordinates
    Atoms will no longer be forced into the periodic simulation cell. This output is good for
    conformational calculations such as end-to-end distance, pitch and persistence length.
    
    Arguments:
        dataframe (DataFrame): the pandas dataframe that contains the trajectory
        This dataframe must contain the image flags assoaciated with PBC as well as unscaled coordinates and box boundaries
        
    Returns:
        A dataframe with the timestep, molecule information x,y,z coordinates
    """
    
    df['x'] += (df['xhi']-df['xlo']) * df['ix'].astype(float)
    df['y'] += (df['yhi']-df['ylo']) * df['ix'].astype(float)
    df['z'] += (df['zhi']-df['zlo']) * df['ix'].astype(float)
    
    return df.drop(['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi', 'ix', 'iy', 'iz'], axis=1)

def log_file_to_dataframe(logfile, column_headers):
    pass


#Test Cases
#test_dump = dump_file_to_dataframe("gyrationB.out", ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'])
#test_trj = trj_file_to_dataframe("chiral.lammpstrj", ["id", "mol", "type", "xs", "ys", "zs", "ix", "iy", "iz"], sort=["mol", "id"])
#us = unscale_dataframe(test_trj)
#uw = unwrap_dataframe(us)


