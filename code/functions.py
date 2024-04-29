import os
import pandas as pd
import json
from typing import List
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from data_handling import DataFramePlus
from project_dir import *

def to_seconds(df: pd.DataFrame, col_t: str, sampling_freq=100) -> None:
    """ Convert a pandas Series to seconds
    
    Parameters
    __________
    df: pandas.DataFrame
        DataFrame with column to convert to seconds
    col_t: str
        Name of the time column to convert
    sampling_freq: float
        Sampling frequency of the data
        
    Returns
    __________
    None
    """
    dt = df.loc[1, col_t] - df.loc[0, col_t]
    scale = (1 / sampling_freq) / dt
    df[col_t] = df[col_t] * scale
    df[col_t] = df[col_t] - df.loc[0, col_t]
   

def filter(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """ Apply a median filter to a pandas Series
    
    Parameters
    __________
    df: pandas.DataFrame
        DataFrame with column to filter
    col: str
        Name of the column to filter
    window: int
        Window size for the median filter
    
    Returns
    __________
    pandas.DataFrame
        DataFrame with filtered column
    """
    
    df[col] = medfilt(df[col], window)
    
    return df

def load(fname: str, dof: str) -> DataFramePlus:
    """ Load and preprocess HD data
    
    Parameters
    __________
    fname: str
        File pseudonym
    dof: str
        Degree of freedom to load
    
    Returns
    __________
    DataFramePlus
        Preprocessed DataFrame
    """
    
    colidx_match = {"x": 0, "y": 1, "z": 2, "phi": 3, "theta": 4, "psi": 5}

    df_cmd = DataFramePlus()
    df_mes_tick = DataFramePlus()
    df_mes_pos = DataFramePlus()
    
    paths_cmd = paths_hdf5_cmd(dof)
    df_cmd.read_hdf5(paths_hdf5_main[fname], paths_cmd)
    
    path_mes_tick = dict([list(paths_hdf5_mes.items())[0]])
    df_mes_tick.read_hdf5(paths_hdf5_main[fname], path_mes_tick)
    
    path_mes_pos = dict([list(paths_hdf5_mes.items())[1]])
    df_mes_pos.read_hdf5(paths_hdf5_main[fname], path_mes_pos, colidx=colidx_match[dof])
    
    df_mes = pd.concat([df_mes_tick, df_mes_pos], axis=1)
    df = pd.merge(df_cmd, df_mes, on='t', how='inner')
    df = DataFramePlus(df)
    
    if dof == 'z':
        # Invert z-axis
        df['pos_mes'] *= -1
    
    return df
    
def extract_from_json(file_type: str) -> list:
    """ Extracts relevant data from a JSON file.
    
    Parameters
    __________
    file_type: str
        The type of JSON file to extract data from
    
    Returns
    __________
    list
        A list of extracted data
    """
    
    file_path = {
        "AGARD-AR-144_A": 'data/json/srs-agard144a.json',
        "AGARD-AR-144_B": 'data/json/srs-agard144b.json',
        "AGARD-AR-144_D": 'data/json/srs-agard144d.json', 
        "AGARD-AR-144_E": 'data/json/srs-agard144e.json',
        "MULTI-SINE_1": 'data/json/srs-test-motion-sines1.json',
        "MULTI-SINE_2": 'data/json/srs-test-motion-sines2.json',
        "MULTI-SINE_3": 'data/json/srs-test-motion-sines3.json'
        }
    
    with open(file_path[file_type], 'r') as f:
        # Load the JSON data
        data = json.load(f)
    
    extracted_data = [
        [move["time"],
        move["move"]["profile"]["Tfade"],
        move["move"]["profile"]["Ttotal"],
        move["move"]["profile"]["omg"],
        move["move"]["profile"]["gain"],
        move["move"]["profile"]["phi0"],
        move["move"]["axis"]]
        for move in data["moves"] if "profile" in move["move"] and "FadedSineProfile" in move["move"]["profile"]["type"]
    ]

    return extracted_data

def time_stamps(file_type: str) -> list:
    """ Extracts the time stamps from the JSON data.
    
    Parameters
    __________
    file_type: str
        The type of JSON file to extract data from
    
    Returns
    __________
    list
        A list of time stamps
    """
    
    file_directory = {
        "AGARD-AR-144_A": 'data/json/srs-agard144a.json',
        "AGARD-AR-144_B+E": 'data/json/srs-agard144b.json',
        "AGARD-AR-144_D": 'data/json/srs-agard144d.json', 
        "AGARD-AR-144_E": 'data/json/srs-agard144e.json',
        "MULTI-SINE": 'data/json/srs-test-motion-sines1.json',
        "MULTI-SINE_2": 'data/json/srs-test-motion-sines2.json',
        "MULTI-SINE_3": 'data/json/srs-test-motion-sines3.json'
    }
    
    if file_type == 'MULTI-SINE':
        comb_data = extract_from_json(f"{file_type}_1")
    
    elif file_type == 'AGARD-AR-144_B+E':
        comb_data = extract_from_json('AGARD-AR-144_B')
    
    else:
        comb_data = extract_from_json(file_type)
    
    # Check if the current file is srs-test-motion-sines1.json
    if file_directory[file_type] == 'data/json/srs-test-motion-sines1.json':
        # Extract data from srs-test-motion-sines2.json
        comb_data2 = extract_from_json(f"{file_type}_2") 
        # Adjust the time values of extracted_data2
        for i in range(len(comb_data2)):
            comb_data2[i][0] = comb_data[i][0] + comb_data[-1][0] + comb_data[-1][2]
        # Extend extracted_data with the data from extracted_data2
        comb_data.extend(comb_data2)
        # Extract data from srs-test-motion-sines3.json
        comb_data2 = extract_from_json(f"{file_type}_3") 
        # Adjust the time values of extracted_data2
        for i in range(len(comb_data2)):
            comb_data2[i][0] = comb_data[i][0] + comb_data[-1][0] + comb_data[-1][2]
        # Extend extracted_data with the data from extracted_data2
        comb_data.extend(comb_data2)

    # Check if the current file is srs-agard144b.json
    elif file_directory[file_type] == 'data/json/srs-agard144b.json':
        # Extract data from srs-agard144e.json
        comb_data2 = extract_from_json(f"{file_type[:-3]}E") 
        # Adjust the time values of extracted_data2
        for i in range(len(comb_data2)):
            comb_data2[i][0] = comb_data[i][0] + comb_data[-1][0] + comb_data[-1][2]
        # Extend extracted_data with the data from extracted_data2

        comb_data.extend(comb_data2)
      
    
    extracted_data = comb_data
    
    time_stamps = []
    for i in range(0, len(extracted_data)):
        # Calculate the start and end time of each move
        start_time = extracted_data[i][1] + extracted_data[i][0]
        #start_time = extracted_data[0]
        end_time = extracted_data[i][2] + extracted_data[i][0] - extracted_data[i][1]
        #end_time = extracted_data[2]
        time_stamps.append([start_time, end_time])     
    return time_stamps

def no_fade(df: pd.DataFrame, time_ints: list) -> pd.DataFrame:
    """ Remove fade-in and fade-out from a DataFrame
    
    Parameters
    __________
    df: pandas.DataFrame
        DataFrame to process
    time_ints: list
        List of time intervals to include
    
    Returns
    __________
    pd.DataFrame
        DataFrame with only the rows between the time intervals
    """
    
    mask = pd.Series(False, index = df.index)
    
    for start, end in time_ints:
        mask |= (df['t'] >= start) & (df['t'] <= end)
    
    return df[mask]

def wavelength(df: pd.DataFrame, col: str) -> list:
    """ Find the zero crossing of a column
    
    Parameters
    __________
    df: pd.DataFrame
        DataFrame to process
    col: str
        Column to find zero crossing of
    
    Returns
    __________
    list
        Mask with True on zero crossings, False elsewhere
    """
    mask = [False] * len(df)
    skip_next = False
    for i in range(1, len(df)):
        if df.iloc[i-1][col] * df.iloc[i][col] <= 0 and not (df.iloc[i-1][col] == 0 and df.iloc[i][col] == 0):
            if not skip_next:
                mask[i] = True
                skip_next = True
            else:
                skip_next = False
    return mask

def zero_crossings(df: DataFramePlus, col: str) -> tuple:
    """ Find the zero crossings of a column
    
    Parameters
    __________
    df: DataFramePlus
        DataFrame to process
    col: str
        Column to find zero crossings of
    
    Returns
    __________
    crossings_idx: list
        List of zero crossing indices
    crossings_t: list
        List of zero crossing times
    """
    crossings_idx = []
    crossings_t = []
    for i, row in df.iterrows():
        if i > df.index[0] and (i-1) in df.index and df.loc[i-1, col] * row[col] < 0:
            crossings_idx.append(i)
            crossings_t.append(row['t'])
    
    return crossings_idx, crossings_t


def plot(df: pd.DataFrame, type='acceleration') -> None:
    """ Plot the data
    
    Parameters
    __________
    df: pd.DataFrame
        DataFrame to plot
    type: str
        Type of data to plot
    
    Returns
    __________
    None
    """
    plt.plot(df['t'], df[f'{type[:3]}_mes'], label=f'Measured {type.capitalize()}', color='red')
    plt.plot(df['t'], df[f'{type[:3]}_cmd'], label=f'Commanded {type.capitalize()}', color='blue')
    plt.xlabel('Time [s]')
    plt.ylabel(f'{type.capitalize()} [$m/s^2$]')
    plt.legend()
    plt.grid(True)
    plt.show()