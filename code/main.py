import numpy as np
import functions as fn
import os
from data_handling import DataFramePlus
import matplotlib.pyplot as plt
from typing import List

def preprocess(TARGET: str, DOF: str, overwrite: bool) -> List[DataFramePlus]:
    """ Preprocess data for analysis
    
    Parameters
    __________
    TARGET: str
        Target pseudonym
    DOF: str
        Degree of freedom
    
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
    
    TARGET = 'MULTI-SINE'
    DOF = 'z'
    
    top_bumps, bottom_bumps = [], []
    df_main = preprocess(TARGET, DOF, overwrite=False)      
    if TARGET != 'MULTI-SINE':

        wls = df_main.fragment(fn.wavelength, 'acc_cmd')[1: -1]
        for wl in wls:
            top, bottom = bump_analysis(wl, 0.2)
            top_bumps.extend(top)
            bottom_bumps.extend(bottom)
    else:
        top_bumps, bottom_bumps = bump_analysis(df_main, 0.2)
    
    x = [item[1] for item in bottom_bumps]
    y = [item[0] for item in bottom_bumps]
    plt.scatter(x, y)
    plt.xlabel('Input Amplitude')
    plt.ylabel('Bump Magnitude')
    plt.title('Top Bumps')
    
    # Add dashed trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), '--')
    
    plt.show()