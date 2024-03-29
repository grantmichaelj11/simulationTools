# -*- coding: utf-8 -*-
"""

Tools designed to make post-processing of molecular simulations simpler
    
@author: Michael Grant
"""

import pandas as pd
import random
import numpy as np
import math
from scipy import stats
from scipy.linalg import eigh
import scipy.signal as sc
import os
import csv

def dump_file_to_dataframe(dumpfile, column_headers):
    
    """ 
    Read in a simulation dumpfile (LAMMPS) and output 1 dataframe with each timestep
    
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
    Read in a simulation trajectory (LAMMPs) and output 1 dataframe with each timestep
    
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
                

            elif line.startswith("ITEM: ATOMS"):
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
        
    Returns:
    
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
    
def generate_master_csv_from_multiple(csv_folder, output_file):
    
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
    
    # Create an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # List all CSV files in the input folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    
    # Check if there are any CSV files
    if not csv_files:
        raise ValueError("No CSV files found in the specified folder.")
    
    first_csv_path = os.path.join(csv_folder, csv_files[0])
    
    if first_csv_path == output_file:
        first_csv_path = os.path.join(csv_folder, csv_files[-1])
        
    first_df = pd.read_csv(first_csv_path)
    
    headers = list(first_df.columns)

    # Iterate through each CSV file and append its data to the combined DataFrame
    for csv_file in csv_files:
        
        if csv_file == output_file:
            continue
        
        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)
        
        if list(df.columns) != headers:
            raise ValueError(f"The headers in '{csv_file}' do not match the headers in the first CSV file.")
        
        combined_df = pd.concat([df, combined_df], ignore_index=True)

    #Create a column that adds the filenames
    combined_df['Filename'] = csv_files

    # Save the combined DataFrame to a master CSV file
    combined_df.to_csv(output_file, index=False)
      
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


   
def generate_structure_factor(df, q_cutoff=0.5, output_file=''):
    """ 
    Returns the structure factors for a simulation of diblock copolymers. It is
    important to note this is a very time intensive calculation and doing it over 
    hundreds of frames could take upwards of a few days to complete. Therefore it is
    highly adviseable to only run the structure factor calculation on the final frame
    If error is required it should only be done over the last few frames in a simulation
    
    Currently only takes the last frame - will update to iterate over all frames
    in a future version
    """
    
    df = df.copy()
    
    
    #Takes only the last frame of the trajectory file
    df = df[(df['Timestep'] == df['Timestep'].unique()[-1]) & (df['type'] == 2.0)]
    
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
    final_Qs = pd.DataFrame(list(qs.items()), columns=['q', 'Sq']).sort_values(by=['q'])
    
    if output_file != '':
        final_Qs.to_csv(output_file, encoding='utf-8', index=False)
    
    return final_Qs

def generate_structure_factor_all_frames(df_original, q_cutoff=3, output_file=''):
    """ 
    Returns the structure factors for a simulation of diblock copolymers. It is
    important to note this is a very time intensive calculation and doing it over 
    hundreds of frames could take upwards of a few days to complete. Therefore it is
    highly adviseable to only run the structure factor calculation on the final frame
    If error is required it should only be done over the last few frames in a simulation
    
    Currently only takes the last frame - will update to iterate over all frames
    in a future version
    """
    
    maxSq = []
    timesteps = df_original['Timestep'].unique()
    
    for timestep in timesteps:
        df = df_original.copy()
        
        df = df[(df['Timestep'] == timestep) & (df['type'] == 2.0)]

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
    
    results = {'Timestep': [], 'Pitch': [], 'Monomers': [], 'PersistenceLength': [], 'MonomersLP': []}
    
    for timestep in timesteps:
        
        results['Timestep'].append(timestep)
        
        current_frame = chiral_chains[chiral_chains['Timestep'] == timestep]
        
        pitches = []
        monomers_per_pitch_list = []
        persistence_lengths = []
        monomers_lp = []
        
        for mol_id, group in current_frame.groupby('mol'):
            
            molecule = current_frame[current_frame['mol'] == mol_id]
            
            bead_location = np.zeros(len(molecule)-1)
            distance_location = np.zeros(len(molecule)-1)
            s_counter = np.zeros(len(molecule)-1)
            peaks = []
            
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
                    bead_location[s] += dot
                    s_counter[s] += 1 
                    distance_location[s] += distance
            
            bead_location /= s_counter
            distance_location /= s_counter

            peaks, _ = sc.find_peaks(bead_location, height=0)
            
            if len(peaks) < 4:
                continue
            
            distances = distance_location[peaks]
            
            monomers_per_pitch_list.append(peaks[1])
            pitches.append(distances[1])
            
            #Take the log of the tangent vectors
            peak_table = {"Location": peaks,
                          'Avg Distance': [distance_location[i] for i in peaks],
                          'Avg Tan': [bead_location[i] for i in peaks]}
            
            peak_table = pd.DataFrame(peak_table)

            peak_table.loc[len(peak_table.index)] = [0,0,1]
            peak_table = peak_table.sort_values('Location')
            peak_table.reset_index(drop=True, inplace=True)

            peak_table['ln Tan'] = np.log(peak_table['Avg Tan'].abs())

            try:
                peak_fit = np.polyfit(peak_table['Location'], peak_table['ln Tan'], 1)

            except np.linalg.LinAlgError:
                continue

            s_lp = -1/peak_fit[0]

            if s_lp > 0:
                monomers_lp.append(s_lp)
                x = np.arange(0, len(molecule)-1,1)
                fit = np.polyfit(x, distance_location,1)
                lp = s_lp * fit[0] + fit[1]
                monomers_lp.append(s_lp)
                persistence_lengths.append(lp)


        pitch = np.mean(pitches)
        monomers_per_pitch = np.mean(monomers_per_pitch_list)
        persistence_length = np.mean(persistence_lengths)
        monomers_per_persistence = np.mean(monomers_lp)

        results['Pitch'].append(pitch)
        results['Monomers'].append(monomers_per_pitch)
        results['PersistenceLength'].append(persistence_length)
        results['MonomersLP'].append(monomers_per_persistence)
        
    results_df = pd.DataFrame(results)
    
    if output_file != '':
        results_df.to_csv(output_file, encoding='utf-8', index=False)
        
    return results_df

def calculate_nematic_ordering(df, monomer_type, output_file=''):
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

def calculate_chiral_nematic_ordering(df, monomer_type, output_file):
    df = df[df['type']==monomer_type].drop('type',axis=1)
    
    timesteps = df['Timestep'].unique()
    
    for timestep in timesteps[-1:]:
        
        current_step = df[df['Timestep']==timestep]
        
        for mol_id, molecule_frame in current_step.groupby('mol'):
            
            centers = []
            
            for iteration in range(0, len(molecule_frame)-3):
                
                residue = molecule_frame.iloc[iteration:iteration+4]
                
                atom1 = residue.iloc[0]
                atom2 = residue.iloc[1]
                atom3 = residue.iloc[2]
                atom4 = residue.iloc[3]
                
                #Calculate Torsion via Forward Difference Method
                torsion_r1 = np.array([atom2['x'] - atom1['x'], 
                                       atom2['y'] - atom1['y'],
                                       atom2['z'] - atom1['z']])
                
                torsion_r1 /= np.linalg.norm(torsion_r1)
                
                torsion_r2 = np.array([atom3['x'] - 2*atom2['x'] + atom1['x'],
                                       atom3['y'] - 2*atom2['y'] + atom1['y'],
                                       atom3['z'] - 2*atom2['z'] + atom1['z']])
                
                torsion_r2 /= np.square(np.linalg.norm(torsion_r2))
                
                torsion_r3 = np.array([atom4['x']-3*atom3['x']+3*atom2['x']-atom1['x'],
                                       atom4['y']-3*atom3['y']+3*atom2['y']-atom1['y'],
                                       atom4['z']-3*atom3['z']+3*atom2['z']-atom1['z']])
                
                torsion_r3 /= np.linalg.norm(torsion_r3)**3
                
                torsion = np.dot(np.cross(torsion_r1, torsion_r2), torsion_r3) / np.linalg.norm(np.cross(torsion_r1, torsion_r2))**2
                
                #Calculate Curvature for first 3 points
                
                curvature = np.linalg.norm(np.cross(torsion_r1, torsion_r2)) / np.linalg.norm(torsion_r1)**3
                
                #Calculate Radius
                
                radius = curvature / (curvature**2 + torsion**2)
                
                #Can we find the centroid
                
                break
                
            break

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
            elif monomer == A_type:
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
        


