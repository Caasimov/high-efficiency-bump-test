import os
import pandas as pd
import h5py
from typing import Callable, List, Any

class DataFramePlus(pd.DataFrame):
    """ Custom DataFrame class with additional functionality """
    def read_hdf5(self, fpath: str, colpaths: dict) -> pd.DataFrame:
        """ HDF5 file to pandas DataFrame
        
        Parameters
        __________
        fpath: str
            File path to the HDF5 file
        colpaths: dict
            Dictionary of column paths in the HDF5 file
        
        Returns
        __________
        pandas.DataFrame
            DataFrame with columns from the HDF5 file
        """
        
        with h5py.File(fpath, 'r') as f:
            for col, path in colpaths.items():
                self[col] = f[path][()]
    
    def smart_save(self, fpath: str, **kwargs) -> None:
        """ Save DataFrame to file and check for overwrite
        
        Parameters
        __________
        fpath: str
            File path to save the DataFrame
        kwargs: dict
            Additional keyword arguments for pd.DataFrame.to_csv
            
        Returns
        __________
        None
        """
        if os.path.exists(fpath):
            check = input(f'File {fpath} exists. Overwrite? [y/n]: ')
            if check.lower() == 'n':
                return
            
        self.to_csv(fpath, **kwargs)
    
    def fragment(self, func: Callable[[pd.DataFrame, Any], List[bool]], *args, **kwargs) -> List[pd.DataFrame]:
        """ Fragment DataFrame into a list of smaller DataFrames
        
        Parameters
        __________
        func: Callable
            Row-based logic to fragment the DataFrame, returns True/False list
            
        Returns
        __________
        List[pandas.DataFrame]
            List of DataFrames
        """
        
        mask = func(self, *args, **kwargs)
        fragments = []
        start = 0
        for i, val in enumerate(mask):
            if val:
                fragments.append(self.iloc[start:i])
                start = i
        fragments.append(self.iloc[start:])
        return fragments
    
    def dydx(self, xcol: str, ycol: str, deriv_name: str) -> None:
        """ Calculate the derivative of column y with respect to column x
        
        Parameters
        __________
        xcol: str
            Column name for x
        ycol: str
            Column name for y
        
        Returns
        __________
        None
        """
        shape0 = self.shape
        self[deriv_name] = (self[ycol].shift(-1) - self[ycol].shift(1)) / (self[xcol].shift(-1) - self[xcol].shift(1))
        self.dropna(inplace=True)
        self.reset_index(drop=True, inplace=True)
        
        print(f'Gradient calculated: DataFrame shape {shape0} -> {self.shape}')
    
    def align(self, col1: str, col2: str) -> None:
        """ Align two columns through cross-correlation
        
        Parameters
        __________
        col1: str
            Column name for the first column
        col2: str
            Column name for the second column
        
        Returns
        __________
        None
        
        """
        shape0 = self.shape
    
    def _lag(self):
        pass
    def _offset(self):
        pass