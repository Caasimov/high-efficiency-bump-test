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
        print(f"{TARGET} centroid: {df['pos_mes'].mean()}")
        df['pos_mes'] += df._offset('pos_cmd', 'pos_mes')
        df.align(['pos_mes', 'vel_mes', 'acc_mes'], df._lag('acc_cmd', 'acc_mes'))
        fn.filter(df, 'acc_mes', 3)
        if prune:
            time_stamps = fn.time_stamps(TARGET)
            df = DataFramePlus(fn.no_fade(df, time_ints=time_stamps))
        df.smart_save(f"data/processed/{TARGET}__{DOF}.csv")
    return df

def process(TARGET: str, DOF: str, nsigma: Optional[int]=2) -> Union[Tuple[List[float], List[float], List[float], List[float], List[DataFramePlus]], List[Tuple[List[float], List[float]]]]:
    """ Process data for analysis
    
    Parameters
    __________
    TARGET: str
        Target pseudonym
    DOF: str
        Degree of freedom
    nsigma: int
        Number of standard deviations for bump analysis
    
    Returns
    __________
    Union[Tuple[List[float], List[float], List[float], List[float], List[DataFramePlus]], List[Tuple[List[float], List[float]]]
        Tuple of top bumps x, y and bottom bumps x, y, and list of DataFrames
        List of tuples of x, y data for BUMP+ analysis
    """
    top_bumps, bottom_bumps = [], []
    bump_plus = []
    dfs_fft = []

    df_main = preprocess(TARGET, DOF, overwrite=True, prune=False)
    idx_zeros, _ = fn.zero_crossings(df_main, 'acc_cmd')

    if TARGET not in ('BUMP', 'BUMP+'):
        if TARGET == 'MULTI-SINE':
            time_stamps = fn.time_stamps(TARGET, DOF, [24.58, 19.05])
            wls = df_main.fragment_by_mask(fn.wl_multi_sine, time_stamps)
        else:
            time_stamps = fn.time_stamps(TARGET, DOF)
            wls = df_main.fragment_by_iteration(fn.wavelength, idx_zeros, time_stamps)

        print(f"Analyzing {TARGET} bumps...")
        for wl in wls:
            top, bottom = fn.bump_analysis(wl, 0.2, cutoff=nsigma)
            top_bumps.extend(top)
            bottom_bumps.extend(bottom)
            dfs_fft.append(wl.FFT(['acc_cmd', 'acc_mes'], sampling_rate))

    elif TARGET == 'BUMP+':
        sections = df_main.fragment_by_mask(fn.bump_plus, spikes_min=.9)
        print(f"Analyzing {TARGET} bumps...")
        for section in sections[:-1]:
            if section['t'].max() - section['t'].min() > 20:
                top, bottom = fn.bump_analysis(section, 0.2, cutoff=nsigma)
                bump_plus.append([top, bottom])
    else:
        print(f"Analyzing {TARGET} bumps...")
        top_bumps, bottom_bumps = fn.bump_analysis(df_main, 0.2, cutoff=nsigma)

    print(f"# top bumps: {len(top_bumps)}\n# bottom bumps: {len(bottom_bumps)}")

    if TARGET != 'BUMP+':
        x_t = [item[1] for item in top_bumps]
        y_t = [item[0] for item in top_bumps]
        x_b = [item[1] for item in bottom_bumps]
        y_b = [item[0] for item in bottom_bumps]

        if TARGET == 'BUMP':
            return  x_t, y_t, x_b, y_b, df_main
        else:
            return  x_t, y_t, x_b, y_b, dfs_fft, df_main

    else:
        x_t, y_t, x_b, y_b = [], [], [], []
        for i in range(len(bump_plus)):
            x_top = [item[1] for item in bump_plus[i][0]]
            y_top = [item[0] for item in bump_plus[i][0]]
            x_bot = [item[1] for item in bump_plus[i][1]]
            y_bot = [item[0] for item in bump_plus[i][1]]
            x_t.append(x_top)
            y_t.append(y_top)
            x_b.append(x_bot)
            y_b.append(y_bot)
        return  x_t, y_t, x_b, y_b, df_main
        
def postprocess(type: str, trend_comb: list, trend_sep: Optional[List[list]]=None, save_check: Optional[bool]=True) -> None:
    """ Add generated data to csv output file
    
    Parameters
    __________
    type: str
        Type of data to save
    trend_comb: list
        List of trend data for combined bumps
    trend_sep: Optional[List[list]]
        List of trend data for separated bumps (top/bottom)
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
    
    if trend_sep is None:
        df[type] = [
            None,
            None,
            None,
            None,
            None,
            None,
            trend_comb[0],
            trend_comb[1],
            trend_comb[2]
        ]
    else:
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
    TARGETS = ['MULTI-SINE']
    DOF = 'z'
    fix_trend = False
    n_sigma = 2

    x_all, y_all = [], []

    for TARGET in TARGETS:
        #~! PROCESSING !~#
        if TARGET == 'AGARD-AR-144':
            print(f"Processing AGARD-AR-144 A...")
            x_t_A, y_t_A, x_b_A, y_b_A, dfs_fft_A, df_main_A = process('AGARD-AR-144_A', DOF, nsigma=n_sigma)
            print(f"Processing AGARD-AR-144 B&E...")
            x_t_B, y_t_B, x_b_B, y_b_B, dfs_fft_B, df_main_B = process('AGARD-AR-144_B+E', DOF, nsigma=n_sigma)

            x_A = x_t_A + x_b_A
            y_A = y_t_A + y_b_A
            x_B = x_t_B + x_b_B
            y_B = y_t_B + y_b_B

            x_t = x_t_A + x_t_B
            y_t = y_t_A + y_t_B
            x_b = x_b_A + x_b_B
            y_b = y_b_A + y_b_B

            x = x_A + x_B
            y = y_A + y_B

            x_all.append(x)
            y_all.append(y)

            # Plotting
            tools.plot_signal(df_main_A, save_check=True, fname=f"AGARD-AR-144_A_{DOF}")
            tools.plot_signal(df_main_B, save_check=True, fname=f"AGARD-AR-144_B+E_{DOF}")

            trend_sep_A = tools.plot_IO(x_b_A, y_b_A, x_t_A, y_t_A, freq_dep=False, fix_trend=fix_trend, save_check=True, fname=f"AGARD-AR-144_A_{DOF}_sep")
            trend_sep_B = tools.plot_IO(x_b_B, y_b_B, x_t_B, y_t_B, freq_dep=False, fix_trend=fix_trend, fname=f"AGARD-AR-144_B+E_{DOF}_sep")

            trend_comb_A = tools.plot_IO(x_A, y_A, freq_dep=False, fix_trend=fix_trend, save_check=True, fname=f"AGARD-AR-144_A_{DOF}_comb")
            trend_comb_B = tools.plot_IO(x_B, y_B, freq_dep=False, fix_trend=fix_trend, save_check=True, fname=f"AGARD-AR-144_B+E_{DOF}_comb")

            trend_sep = tools.plot_IO(x_b, y_b, x_t, y_t, freq_dep=False, fix_trend=fix_trend, save_check=True, fname=f"AGARD-AR-144_{DOF}_sep")
            trend_comb = tools.plot_IO(x, y, freq_dep=False, fix_trend=fix_trend, save_check=True, fname=f"AGARD-AR-144_{DOF}_comb")

            tools.plot_deBode(dfs_fft_A, ['acc_cmd', 'acc_mes'], save_check=True, fname=f"AGARD-AR-144_A_{DOF}")
            tools.plot_deBode(dfs_fft_B, ['acc_cmd', 'acc_mes'], save_check=True, fname=f"AGARD-AR-144_B+E_{DOF}")

            dfs_fft_A.extend(dfs_fft_B)
            tools.plot_deBode(dfs_fft_A, ['acc_cmd', 'acc_mes'], save_check=True, fname=f"AGARD-AR-144_{DOF}")

            postprocess('AGARD-AR-144 A', trend_comb_A, trend_sep_A, save_check=True)
            postprocess('AGARD-AR-144 B&E', trend_comb_B, trend_sep_B, save_check=True)
            postprocess('AGARD-AR-144', trend_comb, trend_sep, save_check=True)

        
        elif TARGET == 'MULTI-SINE':
            x_t, y_t, x_b, y_b, dfs_fft, df_main = process(TARGET, DOF, nsigma=n_sigma)
            x = x_t + x_b
            y = y_t + y_b

            # Plotting
            tools.plot_signal(df_main, save_check=True, fname=f"{TARGET}_{DOF}")

            trend_sep = tools.plot_IO(x_b, y_b, x_t, y_t, fix_trend=False, save_check=True, fname=f"{TARGET}_{DOF}_sep")
            trend_comb = tools.plot_IO(x, y, fix_trend=False, save_check=True, fname=f"{TARGET}_{DOF}_comb")

            tools.plot_deBode(dfs_fft, ['acc_cmd', 'acc_mes'], fname=f"{TARGET}_{DOF}")

            postprocess('MULTI-SINE', trend_comb, trend_sep, save_check=True)
        
        elif TARGET == 'BUMP':
            x_t, y_t, x_b, y_b, df_main = process(TARGET, DOF, nsigma=n_sigma)
            x = x_t[:-1] + x_b[:-1]
            y = y_t[:-1] + y_b[:-1]

            x_all.append(x)
            y_all.append(y)

            # Plotting
            tools.plot_signal(df_main, save_check=True, fname=f"{TARGET}_{DOF}")

            trend_sep = tools.plot_IO(x_b[:-1], y_b[:-1], x_t[:-1], y_t[:-1], fix_trend=fix_trend, save_check=True, fname=f"{TARGET}_{DOF}_sep")
            trend_comb = tools.plot_IO(x, y, fix_trend=fix_trend, save_check=True, fname=f"{TARGET}_{DOF}_comb")

            postprocess('BUMP', trend_comb, trend_sep, save_check=True)

        else:
            x_top, y_top, x_bot, y_bot, df_main = process(TARGET, DOF, nsigma=n_sigma)
            names = ['BUMP+_BOTTOM', 'BUMP+_TOP', 'BUMP+_ZERO']

            tools.plot_signal(df_main, save_check=True, fname=f"{TARGET}_{DOF}")

            for i, x_t, y_t, x_b, y_b in zip(range(len(x_top)), x_top, y_top, x_bot, y_bot):
                x = x_t + x_b
                y = y_t + y_b

                x_all.append(x)
                y_all.append(y)

                trend_sep = tools.plot_IO(x_b, y_b, x_t, y_t, fix_trend=fix_trend, save_check=True, fname=f"{names[i]}_{DOF}_sep")
                trend_comb = tools.plot_IO(x, y, fix_trend=fix_trend, save_check=True, fname=f"{names[i]}_{DOF}_comb")
                postprocess(names[i], trend_comb, trend_sep, save_check=True)
    
    plt_names = ['AGARD-AR-144', 'JLM ($z$ = 2.38 m)', 'JLM ($z$ = -0.50 m)', 'JLM ($z$ = 0.50 m)', 'JLM ($z$ = 0.00 m)']
    tools.plot_IO_full(x_all, y_all, plt_names=plt_names, freq_dep=False, bump_isol=True, save_check=True, fname=f"FULL_BUMPS_IO_{DOF}")
    tools.plot_IO_full(x_all, y_all, plt_names=plt_names, freq_dep=False, save_check=True, fname=f"FULL_IO_{DOF}")