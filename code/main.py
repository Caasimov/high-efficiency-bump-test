import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as fn
from typing import List, Optional, Tuple, Union

import tools
from tools import DataFramePlus
from config import *

def preprocess(TARGET: str, DOF: str, overwrite: bool, prune: Optional[bool]=True) -> List[DataFramePlus]:
    """ Preprocess data for analysis
    
    Parameters
    __________
    TARGET: str
        Target pseudonym
    DOF: str
        Degree of freedom
    overwrite: bool
        Overwrite existing data
    prune: bool
        Prune data to remove fade-in and fade-out
    
    Returns
    __________
    List[DataFramePlus]
        List of preprocessed DataFrames
    """
    if not overwrite and os.path.exists(f"data/processed/{TARGET}__{DOF}.csv"):
        df = DataFramePlus()
        df.read_csv(f"data/processed/{TARGET}__{DOF}.csv", index_col=0)
    
    else:
        df = fn.load(TARGET, DOF)
        fn.to_seconds(df, 't')
        df.dydx('t', 'pos_mes', 'vel_mes')
        df.dydx('t', 'vel_mes', 'acc_mes')
        df['pos_mes'] += df._offset('pos_cmd', 'pos_mes')
        df.align(['pos_mes', 'vel_mes', 'acc_mes'], df._lag('acc_cmd', 'acc_mes'))
        fn.filter(df, 'acc_mes', 3)
        if prune:
            time_stamps = fn.time_stamps(TARGET)
            df = DataFramePlus(fn.no_fade(df, time_ints=time_stamps))
        df.smart_save(f"data/processed/{TARGET}__{DOF}.csv")
    return df

def bump_analysis(df: DataFramePlus, tol: float, cutoff: Optional[float]=35) -> Tuple[list, list]:
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
    
    idx_0, t_0 = fn.zero_crossings(df, 'vel_cmd')
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

def postprocess(type: str, trend_sep: List[list], trend_comb: list, save_check: Optional[bool]=True) -> None:
    """ Add generated data to csv output file
    
    Parameters
    __________
    type: str
        Type of data to save
    trend_sep: List[list]
        List of trend data for separated bumps (top/bottom)
    trend_comb: list
        List of trend data for combined bumps
    save_check: bool
        Check if the data should be saved
    
    Returns
    __________
    None"""
    df = DataFramePlus()
    if os.path.exists(f"{path_OUT}/OUTPUT.csv"):
        df.read_csv(f"{path_OUT}/OUTPUT.csv", index_col=0)
    else:
        df = DataFramePlus(columns=['OUT'])
        df['OUT'] = ['intercept_t', 'slope_t', 'R^2_t', 'intercept_b', 'slope_b', 'R^2_b', 'intercept_c', 'slope_c', 'R^2_c']
    
    if type in df.columns and save_check:
        verif = input(f"{type} column not empty. Overwrite? [y/n]: ")
        if verif.lower() == 'n':
            return
        
    df[type] = [
        trend_sep[0][0],
        trend_sep[0][1],
        trend_sep[0][2],
        trend_sep[1][0],
        trend_sep[1][1],
        trend_sep[1][2],
        trend_comb[0],
        trend_comb[1],
        trend_comb[2]
    ]
    df.to_csv(f"{path_OUT}/OUTPUT.csv")
    print(f"Data saved to {path_OUT}/OUTPUT.csv")
         

if __name__ == '__main__':
    ### DEFAULTS ###
    pd.options.mode.chained_assignment = None
    sampling_rate = 100 # Hz
    
    #~! INPUTS !~#
    TARGET = 'BUMP'
    DOF = 'z'
    
    if DOF == 'z':
        sep = True
    
    top_bumps, bottom_bumps = [], []
    dfs_fft = []
    
    df_main = preprocess(TARGET, DOF, overwrite=False, prune=False)
    idx_zeros, time_zeros = fn.zero_crossings(df_main, 'acc_cmd')
    
    tools.plot_signal(df_main, type='acceleration', save_check=True, fname=f"{TARGET}_{DOF}.png")
    
    if TARGET != 'BUMP':
        if TARGET == 'MULTI-SINE':
            time_stamps = fn.time_stamps(TARGET, DOF, [24.58, 19.05])
            wls = df_main.fragment_by_mask(fn.wl_multi_sine, time_stamps)
        else:
            time_stamps = fn.time_stamps(TARGET, DOF)
            wls = df_main.fragment_by_iteration(fn.wavelength, idx_zeros, time_stamps, phase_threshold=10.0)
            
        for wl in wls:
            top, bottom = bump_analysis(wl, 0.2)
            top_bumps.extend(top)
            bottom_bumps.extend(bottom)
            dfs_fft.append(wl.FFT(['acc_cmd', 'acc_mes'], sampling_rate))
    else:
        top_bumps, bottom_bumps = bump_analysis(df_main, 0.2, cutoff=100)
        print(top_bumps, bottom_bumps)
    
    x_t = [item[1] for item in top_bumps]
    y_t = [item[0] for item in top_bumps]
    x_b = [item[1] for item in bottom_bumps]
    y_b = [item[0] for item in bottom_bumps]
    

    x = x_t + x_b
    y = y_t + y_b
    
    trend_sep = tools.plot_IO(x_b, y_b, x_t, y_t, trend=True, save_check=True, fname=f"{TARGET}_{DOF}_sep.png")
    trend_comb = tools.plot_IO(x, y, trend=True, save_check=True, fname=f"{TARGET}_{DOF}_comb.png")
    tools.plot_deBode(dfs_fft, ['acc_cmd', 'acc_mes'], save_check=True, fname=f"{TARGET}_{DOF}.png", cutoff=1)
    
    postprocess(TARGET, trend_sep, trend_comb, save_check=True)