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
        if (((bump_max - bump_min)/abs(df.loc[idx, 'acc_cmd']))*100 < cutoff) and (abs(df.loc[idx, 'acc_cmd']) > 1e-3):
            if df.loc[idx, 'acc_cmd'] < 0:
                bottom_sine.append([bump_max - bump_min, abs(df.loc[idx, 'acc_cmd'])])
            else:
                top_sine.append([bump_max - bump_min, df.loc[idx, 'acc_cmd']])

    return top_sine, bottom_sine

def postprocess():
    pass

if __name__ == '__main__':
    ### DEFAULTS ###
    pd.options.mode.chained_assignment = None
    sampling_rate = 100 # Hz
    
    #~! INPUTS !~#
    TARGET = 'AGARD-AR-144_A'
    DOF = 'z'
    
    sep = False
    
    if DOF == 'z':
        sep = True
    
    top_bumps, bottom_bumps = [], []
    dfs_fft = []
    
    df_main = preprocess(TARGET, DOF, overwrite=False, prune=False)
    idx_zeros, time_zeros = fn.zero_crossings(df_main, 'acc_cmd')
    
    tools.plot_signal(df_main, type='acceleration', save_check=True, fname=f"{TARGET}_{DOF}.png")
    
    if TARGET != 'BUMP':
        if TARGET == 'MULTI-SINE':
            time_stamps = fn.time_stamps(TARGET, DOF, 22.5)
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
        top_bumps, bottom_bumps = bump_analysis(df_main, 0.2)
    
    x_t = [item[1] for item in top_bumps]
    y_t = [item[0] for item in top_bumps]
    x_b = [item[1] for item in bottom_bumps]
    y_b = [item[0] for item in bottom_bumps]
    
    if not sep:
        x = x_t + x_b
        y = y_t + y_b
    
    tools.plot_IO(x_b, y_b, x_t, y_t, trend=True, save_check=True, fname=f"{TARGET}_{DOF}.png")
    tools.plot_deBode(dfs_fft, ['acc_cmd', 'acc_mes'], save_check=True, fname=f"{TARGET}_{DOF}.png")