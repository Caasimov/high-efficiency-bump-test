import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as fn
from typing import List

from tools import DataFramePlus



### DEFAULTS ###
pd.options.mode.chained_assignment = None

def preprocess(TARGET: str, DOF: str, overwrite: bool, prune=True) -> List[DataFramePlus]:
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

def bump_analysis(df: DataFramePlus, tol: float, sep=True, cutoff=0.06) -> List[list]:
    """ Obtain bump magnitudes and corresponding input amplitudes
    
    Parameters
    __________
    df: DataFramePlus
        DataFrame containing the data
    tol: float
        Time tolerance for bump consideration
    sep: bool
        Separate bumps into bottom/top of sine wave
    cutoff: float
        Cutoff value for bump magnitude
    
    Returns
    __________
    List[list]
        List of bump magnitudes and corresponding input amplitudes
    """
    top_sine = []
    bottom_sine = []
    data = []
    
    idx_0, t_0 = fn.zero_crossings(df, 'vel_cmd')
    df['diff'] = df['acc_mes'] - df['acc_cmd']
    
    for idx, t in zip(idx_0, t_0):
        interval = df[(df['t'] >= (t - tol)) & (df['t'] <= (t + tol))]
        bump_max = interval['diff'].max()
        bump_min = interval['diff'].min()
        if (bump_max - bump_min < cutoff) and (abs(df.loc[idx, 'acc_cmd']) > 1e-3):
            if sep:
                if df.loc[idx, 'acc_cmd'] < 0:
                    bottom_sine.append([bump_max - bump_min, abs(df.loc[idx, 'acc_cmd'])])
                else:
                    top_sine.append([bump_max - bump_min, df.loc[idx, 'acc_cmd']])
            else:
                data.append([bump_max - bump_min, df.loc[idx, 'acc_cmd']])
    if sep:
        return top_sine, bottom_sine
    else:
        return data
        

if __name__ == '__main__':
    
    TARGET = 'AGARD-AR-144_A'
    DOF = 'z'
    
    top_bumps, bottom_bumps = [], []
    # df_main = preprocess(TARGET, DOF, overwrite=True, prune=False)
    # fn.plot(df_main)
    df_main = preprocess(TARGET, DOF, overwrite=False, prune=False)
    time_stamps = fn.time_stamps(TARGET)
    idx_zeros, time_zeros = fn.zero_crossings(df_main, 'acc_cmd')
    fn.plot(df_main)
    if TARGET != 'MULTI-SINE':
        wls = df_main.fragment(fn.wavelength, idx_zeros, time_stamps)
        for wl in wls:
            fn.plot(wl)
            top, bottom = bump_analysis(wl, 0.2)
            top_bumps.extend(top)
            bottom_bumps.extend(bottom)
    else:
        top_bumps, bottom_bumps = bump_analysis(df_main, 0.2)
    
    x_t = [item[1] for item in top_bumps]
    y_t = [item[0] for item in top_bumps]
    x_b = [item[1] for item in bottom_bumps]
    y_b = [item[0] for item in bottom_bumps]
    
    
    plt.scatter(x_t, y_t, color='blue', label='Positive Acceleration')
    plt.scatter(x_b, y_b, color='red', label='Negative Acceleration')
    plt.xlabel('Input Amplitude')
    plt.ylabel('Bump Magnitude')

    # Add dashed trendline
    z_t = np.polyfit(x_t, y_t, 1)
    p_t = np.poly1d(z_t)
    z_b = np.polyfit(x_b, y_b, 1)
    p_b = np.poly1d(z_b)

    plt.plot(x_t, p_t(x_t), 'b--')
    plt.plot(x_b, p_b(x_b), 'r--')
    plt.legend()

    plt.show()