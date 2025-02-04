# -*- coding: utf-8 -*-
"""

Tools designed to make post-processing of particle basedmolecular simulations simpler

Suitable for files generated from LAMMPS
    
@author: Michael Grant
"""

import pandas as pd
import random
import numpy as np
import math
from scipy import stats
from scipy.linalg import eigh
import scipy.signal as sc
from scipy.signal import argrelextrema
from scipy.integrate import simps
import os
import csv

def dump_file_to_dataframe(dumpfile, column_headers):
    
    """ 
    Read in a simulation dumpfile (LAMMPS) and return the contents in a pandas dataframe
    
    Arguments:
        
        dumpfile (str): The name of the LAMMPS dump file to read
        column_headers (list): The names of the desired pandas dataframe column headers 
        
    Returns:
        dataframe with the associated contents desired from the dump file
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
                    
    df = pd.DataFrame(data, columns=column_headers).astype(float)
    
    return df
    
def trj_file_to_dataframe(trjfile, column_headers=['id', 'mol', 'type', 'xs', 'ys', 'zs', 'ix', 'iy', 'iz'], sort=['Timestep']):
    
    """ 
    Read in a simulation trajectory (LAMMPs) and return the contents in a pandas dataframe
    
    Arguments:
        
        trjfile (str): The name of the LAMMPS trajectory file to read
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
                

            elif line.startswith("ITEM: ATOMS") or line.startswith("ITEM: ENTRIES"):
                continue

            else:
                line = [float(i) for i in line.split(" ")]
                line[0:0] = [timestep, xlo, xhi, ylo, yhi, zlo, zhi]
                data.append(line)
    
    
    #If all headers are not identified we will create column names to populate 
    #the dataframe with
    
    missing_columns = len(line) - len(column_headers)
    
    default_columns = [f'Column_{i}' for i in range(1, missing_columns + 1)]
    
    column_headers += default_columns 
    
    return pd.DataFrame(data, columns=column_headers).sort_values(by=sort).reset_index(drop=True)
            
def log_file_to_dataframe(logfile, output_file=""):
    
    """ 
    Takes a LAMMPS log file and converts the contents into a lammps dataframe
    with the option of outputting a CSV file. If there is more than one thermo
    block and they do not contain the same headers then the values for the missing
    thermo values will just read as NaN in the dataframe
    
    Arguments: 
    
        logfile (str): name of LAMMPS log file
        output_file (str): name of the desired output file 
        
    Returns:
    
        A pandas DataFrame containing the contents of the thermo output from a LAMMPS log file
        (Optional) A CSV file associated with the dataframe.

    """
    
    #Initiate an empty dataframe
    df = pd.DataFrame()
    
    with open(logfile, "r") as log_file:
        in_thermo_block = False
        current_data = []
        
        for line in log_file:
            
            if line.startswith("Per MPI rank memory allocation"):
                column_headers = next(log_file).strip().replace("\n","").split()
                in_thermo_block = True
                continue
                
            elif line.startswith("Loop time"):
                in_thermo_block = False
                current_df = pd.DataFrame(current_data, columns=column_headers)
                current_data = []
                df = pd.concat([df, current_df], ignore_index=True).astype(float)
                continue
                
            if in_thermo_block:
                if line.startswith("WARNING"):
                    continue
                current_data.append(line.strip().replace("\n","").split())
    
    if output_file != '':
        df.to_csv(output_file, index=False)
    
    return df

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
    df['y'] = (df['yhi']-df['ylo']) * df['ys'].astype(float)
    df['z'] = (df['zhi']-df['zlo']) * df['zs'].astype(float)

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
    df['y'] += (df['yhi']-df['ylo']) * df['iy'].astype(float)
    df['z'] += (df['zhi']-df['zlo']) * df['iz'].astype(float)
    
    return df.drop(['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi', 'ix', 'iy', 'iz'], axis=1)

def calculate_Ree(df, monomer_type=2, output_file=""):

    """
    Calculates the end to end distance of a block within a polymer (if it is a homopolymer then the entire chain will be calculated)

    Arguments:
        dataframe (DataFrame): pandas dataframe that contains the trajectory - needs to be unscaled and unwrapped

    Returns:
        A dataframe consisting of timestep and end to end distance.

    """
    
    df = df[df['type']==monomer_type].drop('type',axis=1)
    
    timesteps = df['Timestep'].unique()

    results = {'Timesteps': timesteps, 'Ree': []}

    for timestep in timesteps:

        current_step = df[df['Timestep']==timestep]

        Rees = []

        for mol_id, molecule_frame in current_step.groupby(['mol']):

            molecule_data = molecule_frame.sort_values(by='id')

            chain_start = molecule_data.iloc[0]
            chain_end = molecule_data.iloc[-1]

            end_to_end_vector = [chain_end['x'] - chain_start['x'],
                                 chain_end['y'] - chain_start['y'],
                                 chain_end['z'] - chain_start['z']]
            
            mag = np.linalg.norm(end_to_end_vector)

            Rees.append(mag)

        average_Rees = np.mean(np.array(Rees))

        results['Ree'].append(average_Rees)

    final_df = pd.DataFrame(results)

    if output_file != '':
        final_df.to_csv(output_file, index=False)
        
    return final_df

def calculate_gyration_tensor(df, burn_in=50, block_size=10, confidence_interval=0.95, output_file="", average=True):
    
    """
    Takes the dataframe generated from a LAMMPS dump file that contains the gyration
    tensors, xx, yy, zz, xy, xz, yz. Yields a dataframe with each timesteps tensor
    radius of gyration, acylindricity and asphericity. Option to write frame to a csv.
    
    Arguments:
        df (DataFrame): pandas dataframe that contains the gyration tensors from a 
        LAMMPS dump file. Inputs from data frame should be:
            Timestep
            Entry (which is just the molecule)
            xx
            yy
            zz
            xy
            xz
            yz
            
        ouput_file (optional, string): Desired file name (must be CSV). If left
        blank no csv will be written
        
        average: Returns a dataframe where each timestep is averaged. CSV will
        also have average values per timestep as well as standard deviation.
        
    Returns:
        
        Pandas dataframe with calculated quantities and timesteps
            timestep
            molecule (if not averaged, if average is selected this will not return)
            rg
            acylindricity
            asphericity
            anisotropy
    """
    
    def calculate_principal_components(row):
        
        gyration_tensor = np.array([[row['xx'], row['xy'], row['xz']],
                                    [row['xy'], row['yy'], row['yz']],
                                    [row['xz'], row['yz'], row['zz']]])
    
        # Calculate the eigenvalues and eigenvectors of the tensor
        eigenvalues = np.linalg.eigh(gyration_tensor)[0]
    
        # Sort the eigenvalues in ascending order
        eigenvalue_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[eigenvalue_indices]

        return eigenvalues
    
    #Calculate principle components of gyration tensor - sorted in descending order.
    df['eigenvalues'] = df.apply(calculate_principal_components, axis=1)
    df[['L1', 'L2', 'L3']] = df['eigenvalues'].apply(lambda x: pd.Series(x))
    
    #Calculate radius of gyration
    df['Rg'] = np.sqrt(df['L1']+df['L2']+df['L3'])
    
    #Calculate aclindricity
    df['Acylindricity'] = df['L2'] - df['L1']
    
    #Calculate asphericity
    df['Asphericity'] = df['L3'] - 0.5*(df['L1'] + df['L2'])
    
    #Calculate anisotropy
    df['Anisotropy'] = (df['Asphericity']**2 + 0.75 * df['Acylindricity'])/(df['Rg']**4)
    
    if average == True:
        df.drop(['Entry','xx','yy','zz','xy','xz','yz','eigenvalues'], axis=1, inplace=True)
        df = create_block_average_results_dump(df, burn_in, block_size, confidence_interval)
        if output_file != "":
            df.to_csv(output_file, index=False)
        return df
    
    else:
        df.drop(['xx','yy','zz','xy','xz','yz', 'eigenvalues'], axis=1, inplace=True)
        if output_file != "":
            df.to_csv(output_file, index=False)
    
        return df
    
def create_block_average_results_log(df, burn_in, block_size, confidence_interval, output_file=""):
    
    """ 
    Takes a dataframe and conducts block averaging on each column in the frame
    
    Arguments:
        df (DataFrame): The dataframe the user wishes to average over
        burn_in (integer): The number of columns a user wishes to ignore
        block_size (integer): Number of rows per block
        confidence_interval (float between 0-1 exclusive): the confidence interval in which the error is to be calculated
        
    Returns:
        The block-averaged dataframe
    """
    
    #Ignore the burn in period
    df = df[df['Step'] >= df['Step'].unique()[burn_in]] 
    
    data = {}
    
    for column in df.columns.difference(['Step']):
        averages_for_column = []
        errors_for_column = []
        
        num_blocks = len(df) // block_size
        current_frame = df[column].reset_index(drop=True)
        
        block_means = []

        for i in range(num_blocks):
            
            start_index = i * block_size
            end_index = (i+1) * block_size
            block_average = np.mean(current_frame[start_index:end_index])
            block_means.append(block_average)
            
        mean_of_blocks = np.mean(block_means)
        standard_error_of_the_mean = stats.sem(block_means)
         
        if len(block_means) < 30:
            confidence = mean_of_blocks - stats.norm.interval(confidence_interval, loc=mean_of_blocks, scale=standard_error_of_the_mean)[0]
        else:
            confidence = mean_of_blocks - stats.t.interval(confidence_interval, len(block_means) - 1, loc=mean_of_blocks, scale=standard_error_of_the_mean)[0]
         
        averages_for_column.append(mean_of_blocks)
        errors_for_column.append(confidence)
     
        data[column + "_mean"] = np.array(averages_for_column)
        data[column + "_error"] = np.array(errors_for_column)
         
    final_df = pd.DataFrame(data)
    
    return final_df

def create_block_average_results_dump(df, burn_in, block_size, confidence_interval, output_file = ''):
    
    """ 
    Takes a dataframe and conducts block averaging on each column in the frame
    
    Arguments:
        df (DataFrame): The dataframe the user wishes to average over
        burn_in (integer): The number of columns a user wishes to ignore
        block_size (integer): Number of rows per block
        confidence_interval (float between 0-1 exclusive): the confidence interval in which the error is to be calculated
        
    Returns:
        The block-averaged dataframe
    """
    
    if 'Entry' in df.columns:
        df.drop(columns=['Entry'], inplace=True)

    #Ignore the burn in period
    df = df[df['Timestep'] >= df['Timestep'].unique()[burn_in]]
    
    
    #Iterate through every column in the dataframe
    data = {'Timestep': df['Timestep'].unique()}

    for column in df.columns.difference(['Timestep']):
        averages_for_column = []
        errors_for_column = []
        for timestep, group in df.groupby('Timestep'):            
            num_blocks = len(group) // block_size
            current_frame = group[column].reset_index(drop=True)
            
            block_means = []

            for i in range(num_blocks):
                
                start_index = i * block_size
                end_index = (i+1) * block_size
                block_average = np.mean(current_frame[start_index:end_index])
                block_means.append(block_average)
            
            # Calculate Average, standard error and confidence interval
            
            mean_of_blocks = np.mean(block_means)
            standard_error_of_the_mean = stats.sem(block_means)
            
            if len(block_means) < 30:
                confidence = mean_of_blocks - stats.norm.interval(confidence_interval, loc=mean_of_blocks, scale=standard_error_of_the_mean)[0]
            else:
                confidence = mean_of_blocks - stats.t.interval(confidence_interval, len(block_means) - 1, loc=mean_of_blocks, scale=standard_error_of_the_mean)[0]
            
            averages_for_column.append(mean_of_blocks)
            errors_for_column.append(confidence)
        
        data[column + "_mean"] = np.array(averages_for_column)
        data[column + "_error"] = np.array(errors_for_column)
        
        final_df = pd.DataFrame(data)
        
    if output_file != '':
        final_df.to_csv(output_file, index=False)
        
    return final_df

def block_average_dump_dataframe(df, burn_in, block_size, confidence_interval, output_file=''):
    """ 
    Typically the dump file will need to be averaged over each timestep. To get
    the averaged quantity after some burn in another round of averaging needs to
    be conducted. This function takes an already averaged dataframe and returns
    scalar values for each column in the frame -> with error propogated. Will
    return a csv with each columns final average value
    
    Arguments:
        df (DataFrame): The dataframe the user wishes to average over
        burn_in (integer): The number of columns a user wishes to ignore
        block_size (integer): Number of rows per block
        confidence_interval (float between 0-1 exclusive): the confidence interval in which the error is to be calculated
        
    Returns:
        The block-averaged dataframe
    """
    
    if burn_in + block_size > len(df):
        raise ValueError("Not enough rows to perform block averaging with the given block size.")

    df.drop(columns=['Timestep'], inplace=True)
    
    columns_to_drop = [col for col in df.columns if col[-5:] == 'error']
    df = df.drop(columns=columns_to_drop)

    df = df.iloc[burn_in:]
    
    num_blocks = len(df) // block_size
    
    block_averages = []
    block_errors = []
    
    for col in df.columns:
        
        col_data = df[col].values
        
        blocks = col_data[:num_blocks * block_size].reshape(-1, block_size)
        
        # Calculate block averages
        block_avg = np.mean(blocks, axis=1)
        
        #Average each blocks average
        mean_of_blocks = np.mean(block_avg)
        
        # Calculate block errors using standard error of the mean (SEM)
        block_error = np.std(block_avg, axis=0) / np.sqrt(block_size)
        
        if len(blocks) < 30:
            confidence = mean_of_blocks - stats.norm.interval(confidence_interval, loc=mean_of_blocks, scale=block_error)[0]
        else:
            confidence = mean_of_blocks - stats.t.interval(confidence_interval, len(block_avg) - 1, loc=mean_of_blocks, scale=block_error)[0]
            
        block_averages.append(mean_of_blocks)
        block_errors.append(confidence)
    
    result_df = pd.DataFrame({f"{col}_avg": avg for col, avg in zip(df.columns, block_averages)}, index=[0])
    result_df_err = pd.DataFrame({f"{col}_error": err for col, err in zip(df.columns, block_errors)}, index=[0])
    result_df = pd.concat([result_df, result_df_err], axis=1)
    
    if output_file != '':
        result_df.to_csv(output_file, index=False)
    
    return result_df
    
def generate_master_csv_from_multiple(csv_folder, header, output_file):
    
    """ 
    Some of these functions create a CSV file with a singular row. This function
    iterates through all csv files and combines them into a master CSV file under
    the assumption that all columns in each CSV are identical.
    
    Arguments:
        csv_folder: the path to the folder containing the similar CSV files
        output_file: The name of the combined CSV
        
    Returns:
        A combined CSV file of your desired metrics.
    """

    # List all CSV files in the input folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    
    # Check if there are any CSV files
    if not csv_files:
        raise ValueError("No CSV files found in the specified folder.")

    # Iterate through each CSV file and append its data to the combined DataFrame
    data = {'values': [], 'files': []}
    for csv_file in csv_files:
        
        if csv_file == output_file:
            continue
        
        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)

        data['values'].append(df[header].values[0])
        data['files'].append(csv_file)

    #Create a column that adds the filenames
    df = pd.DataFrame(data)

    # Save the combined DataFrame to a master CSV file
    df.to_csv(output_file, index=False)
      
def extract_domain_spacing_trajectory_dataframe(df, bin_size=0.1, output_file=""):
    
    """ 
    Extracts the z coordinates (according to bin size) and the relative densities
    of both type of copolymers in the structure.
    
    Arguments:
        df (DataFrame): The dataframe generated from a trajectory file
        bin_size (int): The bin size that contains a set of beads.
        output_file (optional): A CSV file derived from the generated dataframe
        
    Returns:
        A DataFrame with "Density1", "Density2" "z" and optionally a csv corresponding
        to said dataframe
    """
    
    df = df[df['Timestep'] == df['Timestep'].unique()[-1]]
    
    #Sorts by z axis and then by type
    df = df.sort_values(['z','type']).reset_index()
    
    df['x'] = df['x'] - df['x'].min()
    df['y'] = df['y'] - df['y'].min()
    df['z'] = df['z'] - df['z'].min()
    
    data = []
    
    #Loop through dataframe according to bin size
    z = 0
    
    while z <= round(df['z'].max()):
        
        temp_df = df.copy()
        
        temp_df = temp_df[(temp_df['z'] >= z) & (df['z'] < z+bin_size)]
        
        z += bin_size
        
        number_of_type_1 = temp_df[temp_df['type']==1.0].count()['type']
        number_of_type_2 = temp_df[temp_df['type']==2.0].count()['type']
        
        total = number_of_type_1 + number_of_type_2
        
        if total == 0:
            continue
        
        density_type_1 = number_of_type_1/total
        density_type_2 = number_of_type_2/total
        
        data.append([density_type_1, density_type_2, z])
        
        z += bin_size

    column_headers = ['Density1', 'Density2', 'z']
    
    final_frame = pd.DataFrame(data, columns=column_headers)
    
    if output_file != '':
        final_frame.to_csv(output_file, index=False)
    
    return final_frame

def extract_interfacial_width_from_domain_spacing(df, output_file = ''):
    
    #Find the maximum of Type 1
    #Find first z value that is less than 90% of that maximum
    #Find the minimum of Type 1
    #Find first z value that is greater than 10% of that minimum
    
    density = df['Density1'].array
    z = df['z'].array
    
    max_value = np.max(density)
    max_indicies = np.where(density == max_value)[0][0]
    
    min_value = np.min(density)
    min_indicies = np.where(density == min_value)[0][0]
    
    #From the max value, iterate to the right until we find a value less than 90% of max
    for i in range(max_indicies, len(z)):
        if density[i] <= max_value*0.9:
            z_90_max = z[i]
            break
    
    
    #From the min value, iterate to the left until we find a value greater than 10% of min
    for i in range(0, min_indicies):
        n = min_indicies - i
        if min_value == 0:
            if density[n] >= 0.1:
                z_90_min = z[n]
                break
        else:
            if density[n] >= 0.1*min_value:
                z_90_min = z[n]
                break
            
    Interfacial_Width = abs(z_90_min - z_90_max)
    
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow(['Interfacial Width'])

        csv_writer.writerow([Interfacial_Width])

def generate_structure_factor(df, q_cutoff=0.5, mon_type=2.0, output_file=''):

    """ 
    Calculate the static structure factor of a given monomer type from a LAMMPS simulation
    
    Arguments:
        df (DataFrame): The dataframe generated from a trajectory file (trj_file_to_dataframe()) (MUST NOT BE UNSCALED OR UNWRAPPED)
        q_cutoff (float): Wave vectors with magnitudes larger than the cutoff will be truncated.
        monomer_type (float): The monomer type that will interact with the wave vectors.
        output_file (optional): A CSV file containing the results of hte calculation.
        
    Returns:
        A DataFrame containing the magnitude of the wave vector (q) and the associated structure factor (Sq)
    """
    
    df = df.copy()
    
    #Takes only the last frame of the trajectory file
    df = df[(df['Timestep'] == df['Timestep'].unique()[-1]) & (df['type'] == mon_type)]
    
    #Specifies box lengths
    Lx = df['xhi'].unique()[0] - df['xlo'].unique()[0]
    Ly = df['yhi'].unique()[0] - df['ylo'].unique()[0]
    Lz = df['zhi'].unique()[0] - df['zlo'].unique()[0]

    #Unscales the coordinates
    df = unscale_dataframe(df)
    
    #Creates wave vectors that will interact with structure
    vectors = []
    x_points = 2 * np.pi * np.arange(Lx) / Lx
    y_points = 2 * np.pi * np.arange(Ly) / Ly
    z_points = 2 * np.pi * np.arange(Lz) / Lz
    
    x, y, z = np.meshgrid(x_points, y_points, z_points, indexing='ij')
    mags = np.linalg.norm(np.array([x, y, z]), axis=0)
    
    mask = (mags > 0) & (mags <= q_cutoff)
    
    vectors = np.column_stack((mags[mask], x[mask], y[mask], z[mask]))
        
    waves = pd.DataFrame(vectors, columns=['q','x', 'y', 'z'])

    #Using the wave vectors, calculate the structure factor
    qs = {}

    # Convert 'df' coordinates to a NumPy array for faster calculations
    atom_coordinates = df[['x', 'y', 'z']].values
    
    for index, row in waves.iterrows():
        q = row['q']
        
        if q not in qs:
            q_array = np.array([row['x'], row['y'], row['z']])
            dot_products = np.dot(q_array, atom_coordinates.T)
            cos_sum = np.sum(np.cos(dot_products))
            sin_sum = np.sum(np.sin(dot_products))
            
            sQ = (cos_sum ** 2 + sin_sum ** 2) / len(df)
            
            qs[q] = sQ
    
    # Convert the dictionary into a DataFrame
    final_Qs = pd.DataFrame(list(qs.items()), columns=['q', 'Sq']).sort_values(by=['Sq'])
    
    if output_file != '':
        final_Qs.to_csv(output_file, encoding='utf-8', index=False)
    
    return final_Qs

def generate_structure_factor_bias(df, mon_type=2.0, output_file=''):
    """ 
    Calculate the static structure factor of a given monomer type from a LAMMPS simulation - this function includes
    the wave vectors associated with each signal. Useful for understanding what vectors lead to your signal.
    
    Arguments:
        df (DataFrame): The dataframe generated from a trajectory file (trj_file_to_dataframe()) (MUST NOT BE UNSCALED OR UNWRAPPED)
        monomer_type (float): The monomer type that will interact with the wave vectors.
        output_file (optional): A CSV file containing the results of hte calculation.
        
    Returns:
        A DataFrame containing the magnitude of the wave vector (q) and the associated structure factor (Sq)
    """
    
    df = df.copy()
    
    
    #Takes only the last frame of the trajectory file
    df = df[(df['Timestep'] == df['Timestep'].unique()[-1]) & (df['type'] == mon_type)]
    
    #Specifies box lengths
    Lx = df['xhi'].unique()[0] - df['xlo'].unique()[0]
    Ly = df['yhi'].unique()[0] - df['ylo'].unique()[0]
    Lz = df['zhi'].unique()[0] - df['zlo'].unique()[0]

    #Unscale the coordinates
    df = unscale_dataframe(df)
    
    # Create integer wave vector components based on box size
    n_x = np.arange(int(Lx))
    n_y = np.arange(int(Ly))
    n_z = np.arange(int(Lz))
    
    # Generate mesh for each component
    nx, ny, nz = np.meshgrid(n_x, n_y, n_z, indexing='ij')
    q_vectors = np.column_stack((nx.ravel(), ny.ravel(), nz.ravel()))
    
    # Filter vectors by magnitude cutoff
    mask = (q_vectors[:, 0]**2 + q_vectors[:, 1]**2 + q_vectors[:, 2]**2) > 0
    q_vectors = q_vectors[mask]
    
    # Convert filtered wave vector components to physical wave vectors
    vectors = 2 * np.pi * q_vectors / np.array([Lx, Ly, Lz])

    #Using the wave vectors, calculate the structure factor
    results = []

    # Convert 'df' coordinates to a NumPy array for faster calculations
    atom_coordinates = df[['x', 'y', 'z']].values
    
    for i, (nx, ny, nz) in enumerate(q_vectors):
        q_vector = vectors[i]
        
        # Calculate dot products for each atom with the wave vector
        dot_products = np.dot(q_vector, atom_coordinates.T)
        cos_sum = np.sum(np.cos(dot_products))
        sin_sum = np.sum(np.sin(dot_products))
        
        # Calculate structure factor for this individual wave vector
        sQ = (cos_sum ** 2 + sin_sum ** 2) / len(df)
        
        # Append individual wave vector index and intensity to results
        results.append({'nx': nx, 'ny': ny, 'nz': nz, 'Sq': sQ})
    
    # Convert the dictionary into a DataFrame
    final_Qs = pd.DataFrame(results).sort_values(by=['Sq'], ascending=False)
    
    if output_file != '':
        final_Qs.to_csv(output_file, encoding='utf-8', index=False)
    
    return final_Qs

def generate_structure_factor_all_frames(df_original, q_cutoff=0.5, mon_type=2.0, output_file=''):
    """ 
    Calculate the static structure factor of a given monomer type from a LAMMPS simulation. This time iterates over all timesteps.
    This may lead to larger workup times depending on your system size. This only returns the maximum signal from each step.
    
    Arguments:
        df (DataFrame): The dataframe generated from a trajectory file (trj_file_to_dataframe()) (MUST NOT BE UNSCALED OR UNWRAPPED)
        q_cutoff (float): Wave vectors with magnitudes larger than the cutoff will be truncated.
        monomer_type (float): The monomer type that will interact with the wave vectors.
        output_file (optional): A CSV file containing the results of hte calculation.
        
    Returns:
        A DataFrame containing the magnitude of the wave vector (q) associated the maximum structure factor (Sq) for each timestep
    """
    
    maxSq = []
    timesteps = df_original['Timestep'].unique()
    
    for timestep in timesteps:
        df = df_original.copy()
        
        df = df[(df['Timestep'] == timestep) & (df['type'] == mon_type)]

        #Specifies box lengths
        Lx = df['xhi'].unique()[0] - df['xlo'].unique()[0]
        Ly = df['yhi'].unique()[0] - df['ylo'].unique()[0]
        Lz = df['zhi'].unique()[0] - df['zlo'].unique()[0]
    
        #Unscales the coordinates
        df = unscale_dataframe(df)
        
        #Creates wave vectors that will interact with structure
        vectors = []
        x_points = 2 * np.pi * np.arange(Lx) / Lx
        y_points = 2 * np.pi * np.arange(Ly) / Ly
        z_points = 2 * np.pi * np.arange(Lz) / Lz
        
        x, y, z = np.meshgrid(x_points, y_points, z_points, indexing='ij')
        mags = np.linalg.norm(np.array([x, y, z]), axis=0)
        
        mask = (mags > 0) & (mags <= q_cutoff)
        
        vectors = np.column_stack((mags[mask], x[mask], y[mask], z[mask]))
            
        waves = pd.DataFrame(vectors, columns=['q','x', 'y', 'z'])
    
        #Using the wave vectors, calculate the structure factor
        qs = []
    
        # Convert 'df' coordinates to a NumPy array for faster calculations
        atom_coordinates = df[['x', 'y', 'z']].values
        
        for index, row in waves.iterrows():
            q = row['q']
            
            if q not in qs:
                q_array = np.array([row['x'], row['y'], row['z']])
                dot_products = np.dot(q_array, atom_coordinates.T)
                cos_sum = np.sum(np.cos(dot_products))
                sin_sum = np.sum(np.sin(dot_products))
                
                sQ = (cos_sum ** 2 + sin_sum ** 2) / len(df)
                
                qs.append(sQ)
                
        maxSq.append(max(qs))
        
    # Convert the dictionary into a DataFrame
    final_Qs = pd.DataFrame(maxSq, columns=['q'])
        
    if output_file != '':
        final_Qs.to_csv(output_file, encoding='utf-8', index=False)
        
    return final_Qs

def extract_pitch_from_trajectory_dataframe(df, monomer_type, output_file=''):

    """
    Soon to be deprecated function that would extract the pitch and persistence length of particle-based helical polymers

    DO NOT USE THIS FUNCTION - ONLY KEPT FOR SAKES OF PRIOR COMPARISON

    Use calculate_curtor() function to calculate curvature and torsion and derive helical pitch/radius from this!
    """

    chiral_chains = df[df['type'] == monomer_type].sort_values(['Timestep', 'mol', 'id']).drop(['type'], axis=1)
    
    chiral_chains['tx'] = chiral_chains.groupby('mol')['x'].diff()
    chiral_chains['ty'] = chiral_chains.groupby('mol')['y'].diff()
    chiral_chains['tz'] = chiral_chains.groupby('mol')['z'].diff()
    
    chiral_chains['tangent_distance'] = np.sqrt(chiral_chains['tx']**2 + chiral_chains['ty']**2 + chiral_chains['tz']**2)
    
    chiral_chains['utx'] = chiral_chains['tx'] / chiral_chains['tangent_distance']
    chiral_chains['uty'] = chiral_chains['ty'] / chiral_chains['tangent_distance']
    chiral_chains['utz'] = chiral_chains['tz'] / chiral_chains['tangent_distance']
    
    chiral_chains = chiral_chains.fillna(0)
    timesteps = chiral_chains['Timestep'].unique()
    
    #results = {'Timestep': [], 'Pitch': [], 'Monomers': [], 'PersistenceLength': [], 'MonomersLP': []}
    results = {'Pitch': [], 'Monomers': [], 'PersistenceLength': [], 'MonomersLP': []}

    dfs = []

    timesteps = timesteps[int(0.75*len(timesteps)):]
    
    for timestep in timesteps:
        
        #results['Timestep'].append(timestep)
        
        current_frame = chiral_chains[chiral_chains['Timestep'] == timestep]
        
        for mol_id, group in current_frame.groupby('mol'):
            
            molecule = current_frame[current_frame['mol'] == mol_id]

            currentTanTable = np.zeros((len(molecule)-1, 4))
            currentTanTable = pd.DataFrame(currentTanTable,
                                            columns=["Sum of tan", "Number", "Avg Tan",
                                                    "Avg Distance"])
            
            for i in range(len(molecule)-1):
                
                tangent_index_i = i+1
                t0 = [molecule['utx'].iloc[tangent_index_i],
                    molecule['uty'].iloc[tangent_index_i],
                    molecule['utz'].iloc[tangent_index_i]]
                
                atom1 = np.array([molecule['x'].iloc[i], molecule['y'].iloc[i], molecule['z'].iloc[i]])
                
                for n in range(i, len(molecule)-1):
                    
                    atomN = np.array([molecule['x'].iloc[n], molecule['y'].iloc[n], molecule['z'].iloc[n]])
                    
                    distance = np.sqrt(sum((atomN-atom1)**2))
                    
                    tn = [molecule['utx'].iloc[n],
                        molecule['uty'].iloc[n],
                        molecule['utz'].iloc[n]]
                    
                    s = n-i
                    
                    dot = np.dot(t0, tn)
                    s = n-i
                    currentTanTable["Sum of tan"][s] = currentTanTable["Sum of tan"][s] + dot
                    currentTanTable["Number"][s] = currentTanTable["Number"][s]+1
                    currentTanTable["Avg Distance"][s] = currentTanTable["Avg Distance"][s] + distance
            
            currentTanTable["Avg Tan"] = \
                currentTanTable["Sum of tan"] / currentTanTable["Number"]
            currentTanTable["Avg Distance"] = \
                currentTanTable["Avg Distance"] / currentTanTable["Number"]
            
            dfs.append(currentTanTable)

    stacked = pd.concat(dfs, axis=0, ignore_index=False)
    currentTanTable = stacked.groupby(stacked.index).mean()

    peaks, _ = sc.find_peaks(currentTanTable["Avg Tan"][:], height=0)
    
    #Take the log of the tangent vectors
    peak_table = {"Location": peaks,
                  "Number": currentTanTable['Number'][peaks],
                        "Avg Distance": currentTanTable["Avg Distance"][peaks],
                        "Avg Tan": currentTanTable["Avg Tan"][peaks]}
    
    peak_table = pd.DataFrame(peak_table)
    
    #peak_table.iloc[0] = [0,0,1]
    peak_table = peak_table.sort_values('Location')
    peak_table.reset_index(drop=True, inplace=True)

    CUTOFF = 50

    expoFrame = pd.DataFrame(peak_table['Location'].loc[peak_table["Location"] <= CUTOFF])
    expoFrame["Expo Fit"] = peak_table['Avg Tan'].loc[peak_table["Location"] <= CUTOFF]
    expoFrame['Expo Fit'] = np.log(expoFrame["Expo Fit"])
    
    pitch = peak_table["Avg Distance"].diff(1)[1]
    s_pitch = peak_table["Location"].diff(1)[1]

    expoFrame.dropna(inplace=True)
    expoMin = expoFrame.iloc[argrelextrema(expoFrame["Expo Fit"].values, np.less_equal, order=2)[0]]

    expoFrame = expoFrame[0:expoMin.index[0] + 1]

    peak_fit = np.polyfit(expoFrame["Location"], expoFrame["Expo Fit"], 1)
    x_peak_fit = range(0, expoMin["Location"].iloc[0], 1)
    y_peak_fit = []
    for i in range(len(x_peak_fit)):
        y_peak_fit.append(peak_fit[0]*x_peak_fit[i] + peak_fit[1])

    s_lp = -1/peak_fit[0]

        # Extrapolates/interpolates average distance for s_lp found
    if s_lp > 0:
        currentTanTable.index = currentTanTable.index.map(float)
        # Extrapolates value for s_lp if value found higher than cutoff
        if s_lp > currentTanTable.index[-1]:
            slope = ((currentTanTable["Avg Distance"].iloc[-2] -
                        currentTanTable["Avg Distance"].iloc[-1]) /
                        (currentTanTable.index[-2]-currentTanTable.index[-1]))
            b = currentTanTable["Avg Distance"].iloc[-1] \
                - (slope * currentTanTable.index[-1])
            currentTanTable.loc[s_lp] = [np.nan, np.nan, np.nan, np.nan]
            currentTanTable["Avg Distance"][s_lp] = (slope*s_lp) + b
        else:  # Interpolates if within cutoff
            currentTanTable.loc[s_lp] = [np.nan, np.nan, np.nan, np.nan]
            currentTanTable.sort_index(inplace=True)
            currentTanTable.interpolate(method="linear", inplace=True)
        lp = currentTanTable["Avg Distance"][s_lp]
    else:
        # Print error message stating the s_lp does not exist
        print("\n" + "s_lp = " + str(round(s_lp, 3)) + ". Cannot find lp.")

    results['Pitch'].append(pitch)
    results['Monomers'].append(s_pitch)
    results['PersistenceLength'].append(lp)
    results['MonomersLP'].append(s_lp)
        
    results_df = pd.DataFrame(results)
    
    if output_file != '':
        results_df.to_csv(output_file, encoding='utf-8', index=False)
        
    return results_df

def calculate_nematic_ordering(df, monomer_type, output_file=''):

    """
    Calculates the orientational order parameter for a polymer of specific monomer type from a LAMMPs simulation.

    Arguments:
        df (DataFrame): The dataframe generated from a trajectory file (trj_file_to_dataframe())
        monomer_type (float): The monomer type that will be considered
        output_file (optional): A CSV file containing the results of the calculation.
        
    Returns:
        A DataFrame containing the nematic order parameter at each timestep in the simulation.
    
    """
    
    df = df[df['type']==monomer_type].drop('type',axis=1)
    
    timesteps = df['Timestep'].unique()

    results = {'Timesteps': timesteps, 'Order_Parameter': []}

    for timestep in timesteps:

        end_to_end_vectors = []

        current_step = df[df['Timestep']==timestep]

        for mol_id, molecule_frame in current_step.groupby(['mol']):

            molecule_data = molecule_frame.sort_values(by='id')

            chain_start = molecule_data.iloc[0]
            chain_end = molecule_data.iloc[-1]

            end_to_end_vector = [chain_end['x'] - chain_start['x'],
                                 chain_end['y'] - chain_start['y'],
                                 chain_end['z'] - chain_start['z']]
            
            mag = np.linalg.norm(end_to_end_vector)

            unit_vector = end_to_end_vector/mag

            end_to_end_vectors.append(unit_vector)

        nematic_matrix = np.zeros((3,3))
        N = len(end_to_end_vectors)

        for array in end_to_end_vectors:
        
            x = array[0]
            y = array[1]
            z = array[2]
            
            nematic_matrix[0, 0] += x*x/N
            nematic_matrix[0, 1] += x*y/N
            nematic_matrix[0, 2] += x*z/N
            nematic_matrix[1, 0] += y*x/N
            nematic_matrix[1, 1] += y*y/N
            nematic_matrix[1, 2] += y*z/N
            nematic_matrix[2, 0] += z*x/N
            nematic_matrix[2, 1] += z*y/N
            nematic_matrix[2, 2] += z*z/N
        
        nematic_matrix[0, 0] = 0.5*((3*nematic_matrix[0, 0])-1)
        nematic_matrix[0, 1] = 0.5*(3* nematic_matrix[0, 1])
        nematic_matrix[0, 2] = 0.5*(3*nematic_matrix[0, 2])
        nematic_matrix[1, 0] = 0.5*(3*nematic_matrix[1, 0])
        nematic_matrix[1, 1] = 0.5*((3*nematic_matrix[1, 1])-1)
        nematic_matrix[1, 2] = 0.5*(3*nematic_matrix[1, 2])
        nematic_matrix[2, 0] = 0.5*(3*nematic_matrix[2, 0])
        nematic_matrix[2, 1] = 0.5*(3*nematic_matrix[2, 1])
        nematic_matrix[2, 2] = 0.5*((3*nematic_matrix[2, 2])-1)
            
        eigenvalues, eigenvectors = eigh(nematic_matrix)
        order_parameter = max(eigenvalues.real)
    
        results['Order_Parameter'].append(order_parameter)

    final_df = pd.DataFrame(results)

    if output_file != '':
        final_df.to_csv(output_file, encoding='utf-8', index=False)

    return final_df

def calculate_curtor(df, monomer_type, output_file=''):

    """
    Calculates the curvature and torsion of a helical polymer chain of a specific monomer type.

    Arguments:
        df (DataFrame): The dataframe generated from a trajectory file (trj_file_to_dataframe())
        monomer_type (float): The monomer type within the polymer chain that will be calculated
        output_file (optional): A CSV file containing the results of hte calculation.
        
    Returns:
        A DataFrame containing the curvature and torsion for each timestep in the simulation
    
    """

    df = df[df['type']==monomer_type].drop('type',axis=1)
    
    timesteps = df['Timestep'].unique()

    results = {'Timestep': timesteps, 'curvature': [], 'torsion': []}
    
    for timestep in timesteps:
        
        current_step = df[df['Timestep']==timestep].sort_values(by='id')

        curvature_list = []
        torsion_list = []

        #Iterate through every molecule in the frame
        for mol_id, molecule_frame in current_step.groupby('mol'):  
            
            residue_curvature_list = []
            residue_torsion_list = []

            #Iteratre throug hthe whole molecule calculating curvature and torsion for each helical residue
            for iteration in range(0, len(molecule_frame)-3):
                
                residue = molecule_frame.iloc[iteration:iteration+4]

                atom1 = residue.iloc[0][['x', 'y', 'z']]
                atom2 = residue.iloc[1][['x', 'y', 'z']]
                atom3 = residue.iloc[2][['x', 'y', 'z']]
                atom4 = residue.iloc[3][['x', 'y', 'z']]

                b1 = np.abs(atom2-atom1)
                b2 = np.abs(atom3-atom2)

                tangent = (atom2 - atom1) / b1 #dr1

                bond_length = np.mean(np.array([b1,b2]))
                
                d_tangent = (atom3 - 2*atom2 + atom1) / (bond_length)**2 #dr2

                curvature = np.linalg.norm(np.cross(tangent, d_tangent)) / np.linalg.norm(tangent)**3

                third_derivative = (atom4 - 3*atom3 + 3*atom2 - atom1) / (0.97**3)

                numerator = np.dot(np.cross(tangent, d_tangent), third_derivative)
                denominator = (np.linalg.norm(d_tangent)/(0.97**2)/np.linalg.norm(tangent))**2
                
                torsion = numerator/denominator


                residue_curvature_list.append(curvature)
                residue_torsion_list.append(torsion)

            #Average the radius from the molecule
            residue_curvature_list = np.array(residue_curvature_list)
            average_curvature = np.mean(residue_curvature_list)

            residue_torsion_list = np.array(residue_torsion_list)
            average_torsion = np.mean(residue_torsion_list)

            #Add to master list within the the timestep
            curvature_list.append(average_curvature)
            torsion_list.append(average_torsion)

        average_curvature = np.mean(np.array(curvature_list))
        average_torsion = np.mean(np.array(torsion_list))

        results['curvature'].append(average_curvature)
        results['torsion'].append(average_torsion)

    final_df = pd.DataFrame(results)

    if output_file != '':
        final_df.to_csv(output_file, index=False)
        
    return final_df     

def calculate_helicity(angleFile, dihedralFile, theta, phi, tolerance, output_file=''):

    SET_PHI = phi - 180

    # Number of angles from first timestep of angle file
    headerinfo = pd.read_csv(angleFile, nrows=9)
    angleNum = int(headerinfo.iloc[2, 0])

    # Number of dihedrals from first timestep of dihedral file
    headerinfo = pd.read_csv(dihedralFile, nrows=9)
    dihedralNum = int(headerinfo.iloc[2, 0])

    angle_chunker = pd.read_csv(angleFile, chunksize=angleNum + 9)
    dihedral_chunker = pd.read_csv(dihedralFile, chunksize=dihedralNum + 9)

    TimeHelicityData = pd.DataFrame(columns=["Time", "Helicity"])
    TIMESTEPS = 0

    # For single timeset of data in angle and dihedral files
    for angleChunk, dihedralChunk in zip(angle_chunker, dihedral_chunker):
        angle_time = int(angleChunk.iloc[0, 0])
        dihedral_time = int(dihedralChunk.iloc[0, 0])
        if angle_time != dihedral_time:  # checks that on same timestep
            print("Time error")

        # Removes header and creates data table sorted by atom ID for angle data
        currentAngle = [item[0].split() for item in
                        angleChunk.iloc[8:angleNum+8].values.tolist()]
        currentAngle = pd.DataFrame(currentAngle,
                                    columns=['id', "type", 'atom1', 'atom2',
                                             'atom3', 'Theta', 'Eng'],
                                    dtype='float')
        # Sorts by atom id
        currentAngle["id"] = currentAngle["id"].astype("int64")
        currentAngle.set_index("atom1", inplace=True)
        currentAngle.index = currentAngle.index.astype("int64")
        currentAngle = currentAngle.sort_values("atom1")

        # Removes header and create data table sorted by atom ID for dihedral data
        currentDihedral = [item[0].split() for item in
                           dihedralChunk.iloc[8: dihedralNum + 8].values.tolist()]
        currentDihedral = pd.DataFrame(currentDihedral,
                                       columns=['id', "type", 'atom1', 'atom2',
                                                'atom3', 'atom4', 'Phi'],
                                       dtype='float')
        # Sorts by atom id
        currentDihedral["id"] = currentDihedral["id"].astype("int64")
        currentDihedral.set_index("atom1", inplace=True)
        currentDihedral = currentDihedral.sort_values("atom1")

        # Creates dataframe for current timestep
        currentHelicity = pd.DataFrame(index=list(currentDihedral.index.values),
                                       columns=["Theta", "Theta Diff",
                                                "Theta Spec", "Phi",
                                                "Phi Diff", "Phi Spec", "Residue"],
                                       dtype="float")
        currentHelicity.index = currentHelicity.index.astype("int64")
        currentHelicity.update(currentAngle)
        currentHelicity.update(currentDihedral)

        # Finds difference between setpoint and theta with wrapping
        currentHelicity["Theta Diff"] = currentHelicity["Theta"] - theta
        currentHelicity.loc[(currentHelicity["Theta Diff"]) >
                            180, "Theta Diff"] -= 360
        currentHelicity.loc[(currentHelicity["Theta Diff"]) < -180,
                            "Theta Diff"] += 360

        # Theta Spec = 1 if theta is within spec
        currentHelicity["Theta Spec"] = np.where(abs(currentHelicity["Theta Diff"])
                                                 <= tolerance, 1, 0)

        # Finds difference between setpoint and phi with wrapping
        currentHelicity["Phi Diff"] = currentHelicity["Phi"] - SET_PHI
        currentHelicity.loc[(currentHelicity["Phi Diff"]) >
                            180, "Phi Diff"] -= 360
        currentHelicity.loc[(currentHelicity["Phi Diff"]) <
                            -180, "Phi Diff"] += 360
        # Phi Spec = 1 if within Tolerance
        currentHelicity["Phi Spec"] = np.where(abs(currentHelicity["Phi Diff"])
                                               <= tolerance, 1, 0)
        currentHelicity["Residue"] = np.where((currentHelicity["Theta Spec"] == 1)
                                              & (currentHelicity["Phi Spec"] == 1),
                                              1, 0)
        helicity = currentHelicity["Residue"].sum()/dihedralNum
        TimeHelicityData = TimeHelicityData.append({"Time": angle_time,
                                                    "Helicity": helicity},
                                                   ignore_index=True)
        TIMESTEPS += 1


    if output_file != '':
        TimeHelicityData.to_csv(output_file, index=False)
        
    return TimeHelicityData

def calculate_end_to_end_vector_distribution(df, monomer_type, output_file=''):

    """
    Calculates the end to end distance of a block within a polymer (if it is a homopolymer then the entire chain will be calculated)

    Arguments:
        dataframe (DataFrame): pandas dataframe that contains the trajectory - needs to be unscaled and unwrapped

    Returns:
        A dataframe consisting of timestep and end to end distance.

    """
    
    df = df[df['type']==monomer_type].drop('type',axis=1)
    
    timesteps = df['Timestep'].unique()

    results = {'Molecule': [], 'x': [], 'y': [], 'z': []}

    current_step = df[df['Timestep']==timesteps[-1]]

    for mol_id, molecule_frame in current_step.groupby(['mol']):

        results['Molecule'].append(mol_id)

        molecule_data = molecule_frame.sort_values(by='id')

        chain_start = molecule_data.iloc[0]
        chain_end = molecule_data.iloc[-1]

        end_to_end_vector = [chain_end['x'] - chain_start['x'],
                             chain_end['y'] - chain_start['y'],
                             chain_end['z'] - chain_start['z']]
        

        mag = np.linalg.norm(end_to_end_vector)

        end_to_end_vector[0] /= mag
        end_to_end_vector[1] /= mag
        end_to_end_vector[2] /= mag

        results['x'].append(end_to_end_vector[0])
        results['y'].append(end_to_end_vector[1])
        results['z'].append(end_to_end_vector[2])

    final_df = pd.DataFrame(results)

    if output_file != '':
        final_df.to_csv(output_file, index=False)
        
    return final_df

def calculate_work_from_rdf(df_working, burn_in=0.5, k=1, T=1, output_file=None):
    
    timesteps = df_working['Timestep'].unique()
    timesteps = timesteps[int(len(timesteps)*burn_in):]

    work = np.zeros(len(timesteps))

    for i in range(len(timesteps)):
        df = df_working.copy()
        df = df[df['Timestep'] == timesteps[i]]
        df.replace({'c_gr[2]': 0}, np.nan, inplace=True)
        df.dropna(subset=['c_gr[2]'], inplace=True)
        df['wr'] = -k*T*np.log(df['c_gr[2]'])

        work[i] = simps(df['wr'], x=df['c_gr[1]'])

    average = np.mean(work)
    variance = np.var(work)

    F = average - variance/2/k/T

    final_df = pd.DataFrame({'FreeEnergy': [F]})

    if output_file is not None:
        final_df.to_csv(output_file, index=False)
        
    return F

def write_lammp_input_file(file, num_chains, monomers_per_chain, bond_length, density, volume_fraction):
    
    """ 
    Creates a cubic simulation box of randomly place block copolymer chains. It
    is important to note that if you create a starting system this way, you must
    use soft potentials for the initial push off to eliminate any overlaps between
    atoms. This datafile will additionally generate dihedral and bond angles for
    B type atoms.
    
    Arguments:
        file (string): Name of output file.
        num_chains (int): Number of polymer chains you wish to initiate
        monomers_per_chain (int): Degree of polymerization
        bond_length (float): Bond length between successive beads
        density (float): Density of simulation box, will determine Lx,Ly,Lz
        volume_fraction (float): Volume fraction of A type beads in the simultion and will give 1-volume_fraction for B_beads
        
    Returns:
        A text document named after the file argument that contains the data file that
        can be read in by LAMMPs to initiate a disordered melt simulation.
        
    
    """
    
    def generateImageFlag(low, high, coord):
        if coord == 0:
            return 0
        x = high/coord
        if coord > high:
            return math.floor(1/x)
        elif coord < low:
            return math.ceil(1/x)
        else:
            return 0
    

    #Volume Fraction Determines number of Achiral beads
    A_type = int(monomers_per_chain * volume_fraction)
    
    #Box Dimensions
    total_volume = (num_chains * monomers_per_chain)/density
    side_length = total_volume**(1/3)
    
    xlo, xhi = -side_length/2, side_length/2
    ylo, yhi = xlo, xhi
    zlo, zhi = xlo, xhi
    
    # Generate data arrays/lists
    atoms_data = []
    bonds_data = []
    angles_data = []
    dihedrals_data = []
    
    bond_number = 1
    angle_number = 1
    dihedral_number = 1
    
    for chain in range(1, num_chains + 1):
        
        x, y, z = [random.uniform(xlo,xhi) for _ in range(3)]
        
        for monomer in range(1, monomers_per_chain + 1):
            
            #Randomly Place the next monomer
            theta = random.uniform(0, 360)
            phi = random.uniform(0,360)
            x += bond_length*np.cos(theta*np.pi/180)*np.sin(phi*np.pi/180)
            y += bond_length*np.sin(theta*np.pi/180)*np.sin(phi*np.pi/180)
            z += bond_length*np.cos(theta*np.pi/180)
            
            #Set Image flags to keep Periodicity (not needed due to soft pushoff)
            nx = 0 #generateImageFlag(xlo, xhi, x)
            ny = 0 #generateImageFlag(ylo, yhi, y)
            nz = 0 #generateImageFlag(zlo, zhi, z)
            
            
            #Populate Atoms
            
            atom_type = 1 if monomer <= A_type else 2
            atoms_data.append((monomers_per_chain * (chain - 1) + monomer, chain, atom_type, x, y, z, nx, ny, nz))
            
            if monomer < A_type:
                bonds_data.append((bond_number, 
                                   1, 
                                   monomers_per_chain * (chain - 1) + monomer,
                                   monomers_per_chain * (chain - 1) + monomer + 1))
                bond_number += 1
            elif monomer == A_type and volume_fraction != 1.0:
                bonds_data.append((bond_number, 
                                   2, 
                                   monomers_per_chain * (chain - 1) + monomer, 
                                   monomers_per_chain * (chain - 1) + monomer + 1))
                bond_number += 1
                
            elif monomer > A_type and monomer < (monomers_per_chain):
                bonds_data.append((bond_number,
                                   1,
                                   monomers_per_chain * (chain - 1) + monomer,
                                   monomers_per_chain * (chain - 1) + monomer + 1))
                bond_number += 1
            
            if monomer > A_type and monomer < (monomers_per_chain-1):
                angles_data.append((angle_number,
                                    1, monomers_per_chain * (chain - 1) + monomer,
                                    monomers_per_chain * (chain - 1) + monomer + 1,
                                    monomers_per_chain * (chain - 1) + monomer + 2))
                angle_number += 1
            
            if monomer > A_type and monomer < (monomers_per_chain-2):
                dihedrals_data.append((dihedral_number,
                                       1,
                                       monomers_per_chain * (chain - 1) + monomer,
                                       monomers_per_chain * (chain - 1) + monomer + 1,
                                       monomers_per_chain * (chain - 1) + monomer + 2,
                                       monomers_per_chain * (chain - 1) + monomer + 3))
                dihedral_number += 1
    
    # Write to the file
    with open(file, 'w+') as INPUT_FILE:
        # Write headers and other information
        INPUT_FILE.write("# Random Walk Polymer Melt\n")
        INPUT_FILE.write("# Number of polymers: %1i\n" % num_chains)
        INPUT_FILE.write("# Number of beads per polymer: %1i\n" % monomers_per_chain)
        INPUT_FILE.write("\n")
        INPUT_FILE.write("%10i    atoms\n" % (num_chains*monomers_per_chain))
        INPUT_FILE.write("%10i    bonds\n" % len(bonds_data))
        INPUT_FILE.write("%10i    angles\n" % len(angles_data))
        INPUT_FILE.write("%10i    dihedrals\n" % len(dihedrals_data))
        INPUT_FILE.write("\n")
        INPUT_FILE.write("%10i    atom types\n" % 2)
        INPUT_FILE.write("%10i    bond types\n" % 2)
        INPUT_FILE.write("%10i    angle types\n" % 1)
        INPUT_FILE.write("%10i    dihedral types\n" % 1)
        INPUT_FILE.write("\n")
        INPUT_FILE.write(" %16.8f %16.8f   xlo xhi\n" % (xlo, xhi))
        INPUT_FILE.write(" %16.8f %16.8f   ylo yhi\n" % (ylo, yhi))
        INPUT_FILE.write(" %16.8f %16.8f   zlo zhi\n\n" % (zlo, zhi))

        # Write Atoms
        INPUT_FILE.write("Atoms\n\n")
        for atom_data in atoms_data:
            INPUT_FILE.write("%6i %6i %2i %9.4f %9.4f %9.4f %6i %6i %6i\n" % atom_data)
        
        # Write Bonds
        INPUT_FILE.write("\nBonds\n\n")
        for bond_data in bonds_data:
            INPUT_FILE.write("%8i %8i %8i %8i\n" % bond_data)
        
        if volume_fraction != 1.0:
            # Write Angles
            INPUT_FILE.write("\nAngles\n\n")
            for angle_data in angles_data:
                INPUT_FILE.write("%8i %8i %8i %8i %8i\n" % angle_data)
            
            # Write Dihedrals
            INPUT_FILE.write("\nDihedrals\n\n")
            for dihedral_data in dihedrals_data:
                INPUT_FILE.write("%8i %8i %8i %8i %8i %8i\n" % dihedral_data)
        
        # Write Masses
        INPUT_FILE.write("\nMasses\n\n")
        INPUT_FILE.write("%8i %3f\n" % (1, 1.0))
        INPUT_FILE.write("%8i %3f" % (2, 1.0))
        
def write_lammp_input_file_for_rings(file, num_chains, monomers_per_chain, bond_length, num_solvent, density):
    
    """ 
    Creates a cubic simulation box of coarse-grained polymer rings
    
    Arguments:
        file (string): Name of output file.
        num_chains (int): Number of polymer chains you wish to initiate
        monomers_per_chain (int): Degree of polymerization
        bond_length (float): Bond length between successive beads
        density (float): Density of simulation box, will determine Lx,Ly,Lz
        volume_fraction (float): Volume fraction of A type beads in the simultion and will give 1-volume_fraction for B_beads
        
    Returns:
        A text document named after the file argument that contains the data file that
        can be read in by LAMMPs to initiate a disordered melt simulation.
        
    
    """
    
    #Box Dimensions
    total_volume = (num_chains * monomers_per_chain + num_solvent)/density
    side_length = total_volume**(1/3)
    
    xlo, xhi = -side_length/2, side_length/2
    ylo, yhi = xlo, xhi
    zlo, zhi = xlo, xhi
    
    # Generate data arrays/lists
    atoms_data = []
    bonds_data = []
    angles_data = []
    dihedrals_data = []
    
    bond_number = 1
    angle_number = 1
    dihedral_number = 1

    #Add each polymer chain to the simulation ( random z location )

    for chain in range(1, num_chains+1):
        
        radius = bond_length*(monomers_per_chain+1)/2/np.pi

        random_shift = random.uniform(xlo-(2*radius), xhi-(2*radius))

        x = radius * np.cos(np.linspace(0, 2*np.pi, monomers_per_chain+1)) + random_shift
        y = radius * np.sin(np.linspace(0, 2*np.pi, monomers_per_chain+1)) + random_shift
        z = np.random.normal(0, 0.25, size=len(x)) + random.uniform(xlo, xhi)

        x = x[:-1]
        y = y[:-1]
        z = z[:-1]

        for monomer in range(len(x)):
            atoms_data.append((monomers_per_chain * (chain-1) + monomer + 1,
                               chain,
                               1,
                               x[monomer],
                               y[monomer],
                               z[monomer],
                               0,
                               0,
                               0))

            if monomer < len(x) - 1:
                bonds_data.append((bond_number, 1,
                                monomers_per_chain * (chain-1) + monomer + 1,
                                monomers_per_chain * (chain-1) + monomer + 2))
            else:
                bonds_data.append((bond_number, 1,
                                monomers_per_chain * (chain-1) + monomer + 1,
                                monomers_per_chain * (chain-1) + 1))


            if monomer < len(x) - 2:
                angles_data.append((angle_number, 1,
                                monomers_per_chain * (chain-1) + monomer + 1,
                                monomers_per_chain * (chain-1) + monomer + 2,
                                monomers_per_chain * (chain-1) + monomer + 3))
                
            elif monomer == len(x) - 2:
                angles_data.append((angle_number, 1,
                                    monomers_per_chain * (chain-1) + monomer + 1,
                                    monomers_per_chain * (chain-1) + monomer + 2,
                                    monomers_per_chain * (chain-1) + monomer - monomers_per_chain + 3))
            else:
                angles_data.append((angle_number, 1,
                                    monomers_per_chain * (chain-1) + monomer + 1,
                                    monomers_per_chain * (chain-1) + monomer - monomers_per_chain + 2,
                                    monomers_per_chain * (chain-1) + monomer - monomers_per_chain + 3))
                
            if monomer < len(x) - 3:
                dihedrals_data.append((dihedral_number, 1,
                                       monomers_per_chain * (chain-1) + monomer + 1,
                                       monomers_per_chain * (chain-1) + monomer + 2,
                                       monomers_per_chain * (chain-1) + monomer + 3,
                                       monomers_per_chain * (chain-1) + monomer + 4))
            elif monomer == len(x) - 3:
                dihedrals_data.append((dihedral_number, 1,
                                       monomers_per_chain * (chain-1) + monomer + 1,
                                       monomers_per_chain * (chain-1) + monomer + 2,
                                       monomers_per_chain * (chain-1) + monomer + 3,
                                       monomers_per_chain * (chain-1) + monomer - monomers_per_chain + 4))
                
            elif monomer == len(x) - 2:
                dihedrals_data.append((dihedral_number, 1,
                                       monomers_per_chain * (chain-1) + monomer + 1,
                                       monomers_per_chain * (chain-1) + monomer + 2,
                                       monomers_per_chain * (chain-1) + monomer - monomers_per_chain + 3,
                                       monomers_per_chain * (chain-1) + monomer - monomers_per_chain + 4))
            else:
                dihedrals_data.append((dihedral_number, 1,
                                       monomers_per_chain * (chain-1) + monomer + 1,
                                       monomers_per_chain * (chain-1) + monomer - monomers_per_chain + 2,
                                       monomers_per_chain * (chain-1) + monomer - monomers_per_chain + 3,
                                       monomers_per_chain * (chain-1) + monomer - monomers_per_chain + 4))
        
            bond_number += 1
            angle_number += 1
            dihedral_number += 1

    for solvent in range(num_solvent):
        x, y, z = [random.uniform(xlo,xhi) for _ in range(3)]
        atoms_data.append((len(atoms_data)+solvent+1, monomers_per_chain+solvent+1, 2, x, y, z, 0, 0, 0))
    
    # Write to the file
    with open(file, 'w+') as INPUT_FILE:
        # Write headers and other information
        INPUT_FILE.write("# Polymer Rings in Arbitrary Solvent\n")
        INPUT_FILE.write("# Number of polymers: %1i\n" % num_chains)
        INPUT_FILE.write("# Number of beads per polymer: %1i\n" % monomers_per_chain)
        INPUT_FILE.write("\n")
        INPUT_FILE.write("%10i    atoms\n" % (num_chains*monomers_per_chain+num_solvent))
        INPUT_FILE.write("%10i    bonds\n" % len(bonds_data))
        INPUT_FILE.write("%10i    angles\n" % len(angles_data))
        INPUT_FILE.write("%10i    dihedrals\n" % len(dihedrals_data))
        INPUT_FILE.write("\n")
        INPUT_FILE.write("%10i    atom types\n" % 2)
        INPUT_FILE.write("%10i    bond types\n" % 1)
        INPUT_FILE.write("%10i    angle types\n" % 1)
        INPUT_FILE.write("%10i    dihedral types\n" % 1)
        INPUT_FILE.write("\n")
        INPUT_FILE.write(" %16.8f %16.8f   xlo xhi\n" % (xlo, xhi))
        INPUT_FILE.write(" %16.8f %16.8f   ylo yhi\n" % (ylo, yhi))
        INPUT_FILE.write(" %16.8f %16.8f   zlo zhi\n\n" % (zlo, zhi))

        # Write Atoms
        INPUT_FILE.write("Atoms\n\n")
        for atom_data in atoms_data:
            INPUT_FILE.write("%6i %6i %2i %9.4f %9.4f %9.4f %6i %6i %6i\n" % atom_data)
        
        # Write Bonds
        INPUT_FILE.write("\nBonds\n\n")
        for bond_data in bonds_data:
            INPUT_FILE.write("%8i %8i %8i %8i\n" % bond_data)
        
        # Write Angles
        INPUT_FILE.write("\nAngles\n\n")
        for angle_data in angles_data:
            INPUT_FILE.write("%8i %8i %8i %8i %8i\n" % angle_data)
        
        # Write Dihedrals
        INPUT_FILE.write("\nDihedrals\n\n")
        for dihedral_data in dihedrals_data:
            INPUT_FILE.write("%8i %8i %8i %8i %8i %8i\n" % dihedral_data)
        
        # Write Masses
        INPUT_FILE.write("\nMasses\n\n")
        INPUT_FILE.write("%8i %3f\n" % (1, 1.0))
        INPUT_FILE.write("%8i %3f" % (2, 1.0))


def write_nematic_starting_configuration(file, num_chains, monomers_per_chain, bond_length, total_molecules, density):
    
    """ 
    Creates a cubic simulation box of randomly place block copolymer chains. It
    is important to note that if you create a starting system this way, you must
    use soft potentials for the initial push off to eliminate any overlaps between
    atoms. This datafile will additionally generate dihedral and bond angles for
    B type atoms.
    
    Arguments:
        file (string): Name of output file.
        num_chains (int): Number of polymer chains you wish to initiate
        monomers_per_chain (int): Degree of polymerization
        bond_length (float): Bond length between successive beads
        density (float): Density of simulation box, will determine Lx,Ly,Lz
        volume_fraction (float): Volume fraction of A type beads in the simultion and will give 1-volume_fraction for B_beads
        
    Returns:
        A text document named after the file argument that contains the data file that
        can be read in by LAMMPs to initiate a disordered melt simulation.
        
    
    """
    
    #Box Dimensions
    total_volume = total_molecules/density
    side_length = total_volume**(1/3)
    
    xlo, xhi = -side_length/2, side_length/2
    ylo, yhi = xlo, xhi
    zlo, zhi = xlo, xhi
    
    # Generate data arrays/lists
    atoms_data = []
    bonds_data = []
    angles_data = []
    dihedrals_data = []
    
    bond_number = 1
    angle_number = 1
    dihedral_number = 1

    #Calculate Number of layers
    layers = math.floor(side_length / monomers_per_chain)
    polymers_per_layer = num_chains/layers

    z0 = zlo
    
    for chain in range(1, num_chains + 1):

        if chain % int(polymers_per_layer) == 0:
            z0 += (monomers_per_chain+1)

        #Randomly place polymers in layer
        x, y = [random.uniform(xlo,xhi) for _ in range(2)]
        
        #Starting z will be the location of first bead in chain
        z=z0

        for monomer in range(1, monomers_per_chain + 1):
            
            z += bond_length
            
            #Set Image flags to keep Periodicity (not needed due to soft pushoff)
            nx = 0 #generateImageFlag(xlo, xhi, x)
            ny = 0 #generateImageFlag(ylo, yhi, y)
            nz = 0 #generateImageFlag(zlo, zhi, z)

            #Populate Atoms    
            atom_number = monomers_per_chain * (chain - 1) + monomer
            atoms_data.append((atom_number, chain, 1, x, y, z, nx, ny, nz))

            if monomer < monomers_per_chain:
                bonds_data.append((bond_number, 
                                    1, 
                                    monomers_per_chain * (chain - 1) + monomer,
                                    monomers_per_chain * (chain - 1) + monomer + 1))
                bond_number += 1
        
            if monomer < monomers_per_chain-1:
                angles_data.append((angle_number,
                                    1, monomers_per_chain * (chain - 1) + monomer,
                                    monomers_per_chain * (chain - 1) + monomer + 1,
                                    monomers_per_chain * (chain - 1) + monomer + 2))
                angle_number += 1
        
            if monomer < monomers_per_chain-2:
                dihedrals_data.append((dihedral_number,
                                        1,
                                        monomers_per_chain * (chain - 1) + monomer,
                                        monomers_per_chain * (chain - 1) + monomer + 1,
                                        monomers_per_chain * (chain - 1) + monomer + 2,
                                        monomers_per_chain * (chain - 1) + monomer + 3))
                dihedral_number += 1


    #Populate "solvent"
    num_solvent = total_molecules - (num_chains*monomers_per_chain)
    molecule_number = num_chains
    for solvent in range(num_solvent):
        x, y, z = [random.uniform(xlo,xhi) for _ in range(3)]
        atom_number += 1
        molecule_number += 1
        atoms_data.append((atom_number, molecule_number, 2, x, y, z, nx, ny, nz))
    
    # Write to the file
    with open(file, 'w+') as INPUT_FILE:
        # Write headers and other information
        INPUT_FILE.write("# Nematic Starting Configuration\n")
        INPUT_FILE.write("# Number of polymers: %1i\n" % num_chains)
        INPUT_FILE.write("# Number of beads per polymer: %1i\n" % monomers_per_chain)
        INPUT_FILE.write("\n")
        INPUT_FILE.write("%10i    atoms\n" % (total_molecules))
        INPUT_FILE.write("%10i    bonds\n" % len(bonds_data))
        INPUT_FILE.write("%10i    angles\n" % len(angles_data))
        INPUT_FILE.write("%10i    dihedrals\n" % len(dihedrals_data))
        INPUT_FILE.write("\n")
        INPUT_FILE.write("%10i    atom types\n" % 2)
        INPUT_FILE.write("%10i    bond types\n" % 1)
        INPUT_FILE.write("%10i    angle types\n" % 1)
        INPUT_FILE.write("%10i    dihedral types\n" % 1)
        INPUT_FILE.write("\n")
        INPUT_FILE.write(" %16.8f %16.8f   xlo xhi\n" % (xlo, xhi))
        INPUT_FILE.write(" %16.8f %16.8f   ylo yhi\n" % (ylo, yhi))
        INPUT_FILE.write(" %16.8f %16.8f   zlo zhi\n\n" % (zlo, zhi))

        # Write Atoms
        INPUT_FILE.write("Atoms\n\n")
        for atom_data in atoms_data:
            INPUT_FILE.write("%6i %6i %2i %9.4f %9.4f %9.4f %6i %6i %6i\n" % atom_data)
        
        # Write Bonds
        INPUT_FILE.write("\nBonds\n\n")
        for bond_data in bonds_data:
            INPUT_FILE.write("%8i %8i %8i %8i\n" % bond_data)
        
        # Write Angles
        INPUT_FILE.write("\nAngles\n\n")
        for angle_data in angles_data:
            INPUT_FILE.write("%8i %8i %8i %8i %8i\n" % angle_data)
        
        # Write Dihedrals
        INPUT_FILE.write("\nDihedrals\n\n")
        for dihedral_data in dihedrals_data:
            INPUT_FILE.write("%8i %8i %8i %8i %8i %8i\n" % dihedral_data)
        
        # Write Masses
        INPUT_FILE.write("\nMasses\n\n")
        INPUT_FILE.write("%8i %3f\n" % (1, 1.0))
        INPUT_FILE.write("%8i %3f" % (2, 1.0))

def create_initial_helix_molecule_file(trj, output_file=''):
    # Read in unwrapped and unscaled trajectory
    # Place atoms in proper order
    trj = trj.sort_values(by='id', axis=0, ignore_index=True).reset_index(drop=True)
    # Shift atom 1 to origin and then shift rest of atoms by same
    trj['x'] -= trj.iloc[0]['x']
    trj['y'] -= trj.iloc[0]['y']
    trj['z'] -= trj.iloc[0]['z']

    N = len(trj)

    # Write molecule file
    if output_file != '':
        with open(output_file, 'w') as file:
            # Write header
            file.write(f"# {N}mer polymer chain\n")
            file.write("# header section\n")
            file.write(f"{N} atoms\n")
            file.write(f"{N-1} bonds\n")
            file.write(f"{N-2} angles\n")
            file.write(f"{N-3} dihedrals\n\n")

            # Write body section for Coords
            file.write("# body section:\n")
            file.write("Coords\n\n")
            for i in range(N):
                atom_id = int(trj.iloc[i]['id'])
                x, y, z = trj.iloc[i][['x', 'y', 'z']]
                file.write(f"{atom_id}    {x:.5f}    {y:.5f}    {z:.5f}\n")

            # Write Types
            file.write("\nTypes\n\n")
            for i in range(N):
                atom_id = int(trj.iloc[i]['id'])
                file.write(f"{atom_id}    1\n")  # Assuming all types are 1

            # Write Bonds
            file.write("\nBonds\n\n")
            for i in range(1, N):
                file.write(f"{i} 1 {i} {i + 1}\n")

            # Write Angles
            file.write("\nAngles\n\n")
            for i in range(1, N-1):
                file.write(f"{i} 1 {i} {i + 1} {i + 2}\n")

            # Write Dihedrals
            file.write("\nDihedrals\n\n")
            for i in range(1, N-2):
                file.write(f"{i} 1 {i} {i + 1} {i + 2} {i + 3}\n")

            # Write Masses
            file.write("\nMasses\n\n")
            for i in range(N):
                atom_id = int(trj.iloc[i]['id'])
                file.write(f"{atom_id} 1\n")