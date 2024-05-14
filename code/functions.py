import os
import pandas as pd
import numpy as np
import json
from typing import List, Tuple, Union, Optional
from scipy.signal import medfilt
from tools import DataFramePlus
from config import *

def to_seconds(df: pd.DataFrame, col_t: str, sampling_freq: Optional[float]=100) -> None:
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
    
    file_path = paths_json[file_type]
    
    with open(file_path, 'r') as f:
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

def adjust_and_extend(comb_data: list, file_type: str, offset: Optional[List[float]]=None) -> list:
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
    comb_data_new = extract_from_json(file_type)
    end_time = comb_data[-1][0] + comb_data[-1][2]
    
    if offset:
        end_time += offset
        
    for i in range(len(comb_data_new)):
        comb_data_new[i][0] += end_time
    comb_data.extend(comb_data_new)
    return comb_data

def time_stamps(file_type: str, dof: str, offset: Optional[List[float]]=None) -> list:
    """ Extract time stamps from a JSON file
    
    Parameters
    __________
    file_type: str
        Type of JSON file to extract time stamps from
    dof: str
        Degree of freedom to extract time stamps for
    offset: float
        Offset to add to the time stamps
    
    Returns
    __________
    list
        List of time stamps
    """

    dof_dir = {
        "x": [1, 0, 0, 0, 0, 0],
        "y": [0, 1, 0, 0, 0, 0],
        "z": [0, 0, 1, 0, 0, 0],
        "phi": [0, 0, 0, 1, 0, 0],
        "theta": [0, 0, 0, 0, 1, 0],
        "psi": [0, 0, 0, 0, 0, 1]
    }

    if file_type == 'MULTI-SINE':
        comb_data = extract_from_json(f"{file_type}_1")
    elif file_type == 'AGARD-AR-144_B+E':
        comb_data = extract_from_json('AGARD-AR-144_B')
    else:
        comb_data = extract_from_json(file_type)

    if file_type == 'MULTI-SINE':
        comb_data = adjust_and_extend(comb_data, f"{file_type}_2", offset[0])
        comb_data = adjust_and_extend(comb_data, f"{file_type}_3", offset[1])
        
    elif file_type == 'AGARD-AR-144_B+E':
        comb_data = adjust_and_extend(comb_data, f"{file_type[:-3]}E")
    time_stamps = [[data[1] + data[0], data[2] + data[0] - data[1]] for data in comb_data if data[6] == dof_dir[dof]]
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

def zero_crossings(df: DataFramePlus, col: str) -> Tuple[list, list]:
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

def bump_analysis(df: DataFramePlus, tol: float, cutoff: Optional[float]=25) -> Tuple[list, list]:
    """ Obtain bump magnitudes and corresponding input amplitudes
    
    Parameters
    __________
    df: DataFramePlus
        DataFrame containing the data
    tol: float
        Time tolerance for bump consideration
    cutoff: float
        Cutoff value for bump magnitude
    
    Returns
    __________
    Tuple[list, list]
        List of bump magnitudes and corresponding input amplitudes
    """
    top_sine = []
    bottom_sine = []
    
    idx_0, t_0 = zero_crossings(df, 'vel_cmd')
    df['diff'] = df['acc_mes'] - df['acc_cmd']
    
    for idx, t in zip(idx_0, t_0):
        interval = df[(df['t'] >= (t - tol)) & (df['t'] <= (t + tol))]
        bump_max = interval['diff'].max()
        bump_min = interval['diff'].min()
        if (((bump_max - bump_min)/abs(df.loc[idx, 'acc_cmd']))*100 <= cutoff) and abs(df.loc[idx, 'acc_cmd']) < .5:
            if df.loc[idx, 'acc_cmd'] < 0:
                bottom_sine.append([bump_max - bump_min, abs(df.loc[idx, 'acc_cmd'])])
            else:
                top_sine.append([bump_max - bump_min, df.loc[idx, 'acc_cmd']])

    return top_sine, bottom_sine


def wavelength(df: DataFramePlus, l_bound: int, idx_zeros: list, time_stamps: List[tuple], phase_threshold: float, col='acc_cmd') -> Tuple[int, bool]:
    """ Determine if the column on a given interval is a wavelength
    
    Parameters
    __________
    df: pd.DataFrame
        DataFrame to process
    l_bound: int
        Lower bound of the interval
    idx_zeros: list
        List of the indexes of zero crossings
    time_stamps: List[tuple]
        List of time stamps for pure signal
    phase_threshold: float
        Amplitude threshold for the phase
    col: str
        Column to process (default='acc_cmd')
    
    Returns
    __________
    Tuple[int, bool]
        Tuple containing the new lower bound and a flag indicating if it is a wavelength
    """
    for i in range(len(idx_zeros) - 2):
        idx_check = round(idx_zeros[i] + (idx_zeros[i+1] - idx_zeros[i]) / 2)
        if (idx_zeros[i] in df.index and idx_zeros[i+1] in df.index and idx_zeros[i+2] in df.index) and round(df.loc[idx_check, col], 3) != 0.0 and abs((df.loc[idx_zeros[i], col] - df.loc[idx_zeros[i], "acc_mes"])/df["acc_cmd"].max())*100 < phase_threshold:
            start_t = df.loc[idx_zeros[i], 't']
            end_t = df.loc[idx_zeros[i+2], 't']
            for start, end in time_stamps:
                if start_t >= start and end_t <= end:
                    return idx_zeros[i], True
            return idx_zeros[i+1], False
            
    return l_bound, False

def wl_multi_sine(df: DataFramePlus, time_stamps: List[tuple]) -> pd.Series:
    """ Generate mask with True for the time stamps, False otherwise
    
    Parameters
    __________
    df: pd.DataFrame
        DataFrame to process
    time_stamps: List[tuple]
        List of time stamps for pure signal
    
    Returns
    __________
    pd.Series
        Mask with True for the time stamps, False otherwise
    """
    mask = pd.Series(False, index = df.index)
    
    for start, end in time_stamps:
        mask |= (df['t'] >= start) & (df['t'] <= end)
    
    return mask

def bump_plus(df: DataFramePlus, spikes_min: Optional[float]=.75, t_min: Optional[int]=10) -> pd.Series:
    """ Create a mask for the DataFrame where rows between spikes are True and all other rows are False
    Parameters
    __________
    df: pd.DataFrame
        DataFrame to process
    spikes_min: float
        Minimum value for a spike to be considered
    t_min: int
        Minimum time between spikes for a section to be considered
    
    Returns
    __________
    pd.Series
        Boolean Series where True indicates the rows between spikes
    """

    spike_detected = False
    between_spikes = False
    mask = pd.Series(False, index=df.index)
    last_spike_end = None

    for i in range(len(df.index)):
        if abs(df.iloc[i]['acc_cmd']) > spikes_min:
            spike_detected = True
            if between_spikes and i - last_spike_end > t_min:
                mask[last_spike_end:i] = True
            between_spikes = False
        else:
            if spike_detected:
                spike_detected = False
                last_spike_end = i
                between_spikes = True

    return mask