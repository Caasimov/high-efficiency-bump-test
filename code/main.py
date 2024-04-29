import pandas as pd
import functions as fn
from data_handling import DataFramePlus
import matplotlib.pyplot as plt

TARGET = 'MULTI-SINE'
DOF = 'z'

df_main = fn.load(TARGET, DOF, False)

fn.to_seconds(df_main, 't')

df_main.dydx('t', 'pos_mes', 'vel_mes')
df_main.dydx('t', 'vel_mes', 'acc_mes')


df_main['pos_mes'] += df_main._offset('pos_cmd', 'pos_mes')

df_main.align(['pos_mes', 'vel_mes', 'acc_mes'], df_main._lag('acc_cmd', 'acc_mes'))

fn.filter(df_main, 'acc_mes', 3)

time_stamps = fn.time_stamps(TARGET)

df_main = DataFramePlus(fn.no_fade(df_main, time_ints=time_stamps))

wavelengths = df_main.fragment(fn.wavelength, 'acc_cmd')[1:-1]


fn.plot(wavelengths[5], 'acceleration')