import os
import pandas as pd
from scipy.signal import medfilt
from data_handling import DataFramePlus
from project_dir import *

def to_seconds(df: pd.DataFrame, col_t: str, sampling_freq=100) -> pd.DataFrame:
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
    pandas.Series
        Series converted to seconds
    """
    dt = df.loc[1, col_t] - df.loc[0, col_t]
    scale = (1 / sampling_freq) / dt 
    return df[col_t] * scale

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

def load(fname: str, dof: str, overwrite: bool) -> DataFramePlus:
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

    
    # Check if file has already been processed and load it
    if not overwrite and os.path.exists(f'data/processed/{fname}__{dof}.csv'):
        df = DataFramePlus.read_csv(f'data/processed/{fname}__{dof}.csv')
    else:
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
        