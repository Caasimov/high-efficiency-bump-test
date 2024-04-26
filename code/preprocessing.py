import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import h5py
import os
from Json_code import *

def hdf5_to_df(fname, dof):
    """
    Load a HDF5 file into a pandas dataframe.
    
    Parameters:
    fname (str): The name of the test file (not path).
    dof (str): The degree of freedom to be analyzed.
    
    Returns:
    pd.DataFrame: A pandas dataframe containing the data.
    
    """
    
    file_dir = {
        "AGARD-AR-144_A": "data/hdf5/motionlog-20240301_133202.hdf5",
        "AGARD-AR-144_B+E": "data/hdf5/motionlog-20240301_141239.hdf5",
        "MULTI-SINE": "data/hdf5/motionlog-20240301_144109.hdf5",
        "BUMP": "data/hdf5/motionlog-20240301_150040.hdf5",
        "PMD": "data/hdf5/motionlog-20240301_150320.hdf5"
    }
    
    paths_cmd = [
        'data/commanded/tick',
        f'data/commanded/data/{dof}',
        f'data/commanded/data/{dof}dot',
        f'data/commanded/data/{dof}dotdot'
        ]
    paths_mes = [
        'data/measured/tick',
        'data/measured/data/actual_pos'
        ]
    
    index_match = {"x": 0, "y": 1, "z": 2, "phi": 3, "theta": 4, "psi": 5}
    column_names_cmd = ['t', 'pos_cmd', 'vel_cmd', 'acc_cmd']
    column_names_mes = ['t', 'pos_mes']
    
    with h5py.File(file_dir[fname], 'r') as f:
        # Create a list of dataframes from the paths
        dfs_cmd = [pd.DataFrame(f[path][()], dtype=np.float64) for path in paths_cmd]
        dfs_mes = [
            pd.DataFrame(f[paths_mes[0]][()], dtype=np.float64),
            pd.DataFrame(f[paths_mes[1]][()][:, index_match[dof]], dtype=np.float64)
            ]
    
    # Concatenate the dataframes along the columns axis
    df_cmd = pd.concat(dfs_cmd, axis=1)
    df_mes = pd.concat(dfs_mes, axis=1)

    # Rename the columns
    df_cmd.columns = column_names_cmd
    df_mes.columns = column_names_mes
    
    # Merge the dataframes on the time column
    merged_df = pd.merge(df_cmd, df_mes, on='t', how='inner')
    
    # Adjust for inverted z-axis
    if dof == 'z':
        merged_df['pos_mes'] = merged_df['pos_mes'] * -1
    
    return merged_df

def clean_data(df):
    """
    Remove rows with NaN values and reset index.
    
    Parameters:
    df (pd.DataFrame): The dataframe to be cleaned.
    
    Returns:
    None
    
    """
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
def apply_filter(df, window_size=3):
    """
    Apply a median filter to the data.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    window_size (int): The size of the window for the median filter.
    
    Returns:
    None
    
    """
    df['vel_mes'] = medfilt(df['vel_mes'], window_size)
    df['acc_mes'] = medfilt(df['acc_mes'], window_size)
    
def find_derivatives(df):
    """
    Compute derivatives of a specified DoF using mixed difference scheme.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    
    Returns:
    None
    
    """
    ### NOTE: change this so that the end points arent excluded ###
    df['vel_mes'] = (df['pos_mes'].shift(-1) - df['pos_mes'].shift(1)) / (df['t'].shift(-1) - df['t'].shift(1))
    df['acc_mes'] = (df['vel_mes'].shift(-1) - df['vel_mes'].shift(1)) / (df['t'].shift(-1) - df['t'].shift(1))
    
    # Remove rows with NaN values and reset index
    clean_data(df)

def find_offset(df):
    """
    Determine system offset for a specified DoF between commanded and measured data.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    
    Returns:
    float: The offset value.
    
    """
    return np.mean(df['pos_mes']) - np.mean(df['pos_cmd'])

def find_lag(df):
    """
    Determine the fixed delay between input and output signals using the commanded and measured positions using cross-correlation.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    
    Returns:
    tuple: A tuple containing the time lag and the index of the zero on the cmd and mes columns.
    
    """    
    # Determine cross correlation
    cross_corr = np.correlate(df['pos_cmd'], df['pos_mes'], mode='full')
    
    idx_lag = abs(cross_corr.argmax() - len(df['pos_mes']) + 1) 
    time_lag = df.loc[idx_lag, 't'] - df.loc[0, 't']
    
    # Actual lag value in terms of time returned along with the position of the zero on the cmd and mes columns
    return time_lag, idx_lag

def find_zero(df, dof='vel_cmd'):
    """
    Finds indexes of velocity reversals.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    dof (str): The degree of freedom to be analyzed, default is 'vel_cmd'.
    
    Returns:
    list: A list of indexes where the velocity is zero.
    
    """
    idx_zero = []
    for i in range(df.index[0], df.index[-1]):
        if df.loc[i, dof] * df.loc[i+1, dof] < 0:
            idx_zero.append(i)
    return idx_zero

def preprocess(df, freq_sample=100):
    """
    Clean the dataframe for direct analysis.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    
    Returns:
    None
    
    """
    
    # Determine time scaling factor
    delta_t = np.float64(1/freq_sample)
    scaling_factor = np.float64(delta_t/(df.loc[1, 't']-df.loc[0, 't']))
    
    # Adjust time column to be in seconds
    df['t'] = df['t']*scaling_factor

    # Add measured velocity and acceleration using forward difference formula
    find_derivatives(df)
    
    # Adjust for amplitude offset between cmd and mes signals
    df['pos_mes'] = df['pos_mes'] - find_offset(df)
    
    # Adjust for lag between cmd and mes signals
    _, idx_lag = find_lag(df)
    df['pos_mes'] = df['pos_mes'].shift(-idx_lag)
    df['vel_mes'] = df['vel_mes'].shift(-idx_lag)
    df['acc_mes'] = df['acc_mes'].shift(-idx_lag)
    clean_data(df)
    
    # Set t_0 to 0
    df['t'] = df['t'] - df.loc[0, 't']

    #Apply a median filter to the data

    #Size of the window for the median filter
    window_size = 3
    
    df['vel_mes'] = medfilt(df['vel_mes'], window_size)
    df['acc_mes'] = medfilt(df['acc_mes'], window_size)

    # Define the directory paths
    directory = 'data'
    subdirectory = 'pandas'
    file_name = 'df.csv'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create the subdirectory if it doesn't exist
    if not os.path.exists(os.path.join(directory, subdirectory)):
        os.makedirs(os.path.join(directory, subdirectory))

    # Define the full file path
    file_path = os.path.join(directory, subdirectory, file_name)

    # Save the DataFrame to CSV
    df.to_csv(file_path, index=False)

    print(f"DataFrame saved to {file_path}")

def isolate_wavelengths(df, file_type):
    """
    Isolate the wavelengths from the data.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    file_type (str): The name of the test file.
    
    Returns:
    list: A list of dataframes containing the isolated wavelengths.
    
    """
    # Extract the data from the JSON file
    extracted_data = combine_data(file_type)[0]
    # Determine the time stamps for pure test sections (no fade in/out)
    time_stamps = time_conversion(extracted_data)
    filtered_dfs = []

    # Iterate through each nested list
    timing = 0 
    for time_range in time_stamps:
        if timing % 3 == 0:
            start_time, end_time = time_range
            
            # Filter the DataFrame based on the time range
            filtered_df = df[(df['t'] >= start_time) & (df['t'] <= end_time)]
            
            # Append the filtered rows to the list
            filtered_dfs.append(filtered_df)
        timing += 1
    
    # Isolate a single wavelength of the commanded acceleration within these pure test dataframes
    df_list = []
    for df_slice in filtered_dfs:
        # Determine zero acceleration points
        zeroes = find_zero(df_slice, dof='acc_cmd')
        
        # Append the slice of the dataframe between the zero acceleration points
        try:
            df_list.append(df_slice.loc[zeroes[1]:zeroes[3], :])
        except:
            df_list.append(None)
       
    return df_list

def plot_dof(df, dof, file_type, interval=None):
    '''
    Plot position, velocity & acceleration corresponding to a specified DoF for both commanded and measured data.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    dof (str): The degree of freedom to be analyzed.
    file_type (str): The name of test file.
    interval (list): The interval of indexes to be analyzed.
    
    Returns:
    None
    
    '''
    if interval == None:
        interval = list(range(0, df.shape[0]))

    # Numerical derivatives of var_1
    vel_1 = df.loc[interval, 'vel_cmd']
    acc_1 = df.loc[interval, 'acc_cmd']
    
    # Numerical derivatives of var_2
    vel_2 = df.loc[interval, 'vel_mes']
    acc_2 = df.loc[interval, 'acc_mes']
    
    fig, axs = plt.subplots(3, 1, figsize=(10,10))

    # Position subplot
    axs[0].plot(df.loc[interval, 't'], df.loc[interval, 'pos_cmd'], label="CMD pos", color='blue')
    axs[0].plot(df.loc[interval, 't'], df.loc[interval, 'pos_mes'], label="MES pos", color='red')
    axs[0].set_title('Position')
    axs[0].legend()

    # Velocity subplot
    axs[1].plot(df.loc[interval, 't'], vel_1, label="CMD vel", color='blue')
    axs[1].plot(df.loc[interval, 't'], vel_2, label="MES vel", color='red')
    axs[1].set_title('Velocity')
    axs[1].legend()

    # Acceleration subplot
    axs[2].plot(df.loc[interval, 't'], acc_1, label="CMD acc", color='blue')
    axs[2].plot(df.loc[interval, 't'], acc_2, label="MES acc", color='red')
    axs[2].set_title('Acceleration')
    axs[2].legend()

    # Adding a xlabel for the whole figure
    for ax in axs:
        ax.set_xlabel('Time')

    # Adjust the layout
    plt.tight_layout()
    
    # Create the plots directory if it doesn't exist
    os.makedirs(f'plots/{dof}', exist_ok=True)
    
    # Save the figure
    plt.savefig(f'plots/{dof}/{file_type}.png', dpi=300)
    
    # Show the figure
    plt.show()

if __name__ == "__main__":
    dof = 'z'
    data = hdf5_to_df('MULTI-SINE', dof)
    preprocess(data)
    apply_filter(data)
    plot_dof(data, dof, 'MULTI-SINE')
    wavelengths = isolate_wavelengths(data, 'MULTI-SINE')
    #print(wavelengths)
    plt.plot(wavelengths[26]['t'], wavelengths[26]['acc_cmd'])
    plt.plot(wavelengths[26]['t'], wavelengths[26]['acc_mes'])
    plt.show()