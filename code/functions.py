import os
import pandas as pd
import json
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from tools import DataFramePlus
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

def adjust_and_extend(comb_data, file_type):
    """ Adjust and extend the extracted data
    
    Parameters
    __________
    comb_data: list
        List of extracted data
    file_type: str
        Type of JSON file to extract data from
    
    Returns
    __________
    list
        List of extracted data
    """
    comb_data2 = extract_from_json(file_type)
    for i in range(len(comb_data2)):
        comb_data2[i][0] = comb_data[i][0] + comb_data[-1][0] + comb_data[-1][2]
    comb_data.extend(comb_data2)
    return comb_data

def time_stamps(file_type: str) -> list:
    """ Extract time stamps from a JSON file
    
    Parameters
    __________
    file_type: str
        Type of JSON file to extract time stamps from
    
    Returns
    __________
    list
        List of time stamps
    """

    if file_type == 'MULTI-SINE':
        comb_data = extract_from_json(f"{file_type}_1")
    elif file_type == 'AGARD-AR-144_B+E':
        comb_data = extract_from_json('AGARD-AR-144_B')
    else:
        comb_data = extract_from_json(file_type)

    if file_type == 'MULTI-SINE':
        comb_data = adjust_and_extend(comb_data, f"{file_type}_2")
        comb_data = adjust_and_extend(comb_data, f"{file_type}_3")
    elif file_type == 'AGARD-AR-144_B+E':
        comb_data = adjust_and_extend(comb_data, f"{file_type[:-3]}E")

    time_stamps = [[data[1] + data[0], data[2] + data[0] - data[1]] for data in comb_data]
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

def wavelength(df: pd.DataFrame, l_bound:int, idx_zeros: list, time_stamps: List[tuple]) -> Tuple[int, bool]:
    """ Determine if the column on a given interval is a wavelength
    
    Parameters
    __________
    df: pd.DataFrame
        DataFrame to process
    idx_zeros: list
        List of the indexes of zero crossings
    time_stamps: List[tuple]
        List of time stamps for pure signal
    
    Returns
    __________
    Tuple[int, bool]
        Tuple containing the new lower bound and a flag indicating if it is a wavelength
    """
    for i in range(len(idx_zeros) - 2):
        if idx_zeros[i] in df.index and idx_zeros[i+1] in df.index and idx_zeros[i+2] in df.index:
            start_t = df.loc[idx_zeros[i], 't']
            end_t = df.loc[idx_zeros[i+2], 't']
            for start, end in time_stamps:
                if start_t >= start and end_t <= end:
                    return idx_zeros[i], True
            return idx_zeros[i+1], False
            
    return l_bound, False

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