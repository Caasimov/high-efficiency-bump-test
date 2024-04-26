import os
import pandas as pd
import numpy as np
import h5py
from typing import Callable, List, Any

class DataFramePlus(pd.DataFrame):
    """ Custom DataFrame class with additional functionality """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def read_hdf5(self, fpath: str, colpaths: dict, colidx=None) -> None:
        """ HDF5 file to pandas DataFrame
        
        Parameters
        __________
        fpath: str
            File path to the HDF5 file
        colpaths: dict
            Dictionary of column paths in the HDF5 file
        colidx: int
            Index of the column to read (optional)
        
        Returns
        __________
        None
        """
        
        with h5py.File(fpath, 'r') as f:
            if colidx is None:
                for col, path in colpaths.items():
                    self[col] = f[path][()]
            else:
                for col, path in colpaths.items():
                    self[col] = f[path][()][:, colidx]
        
    def read_csv(self, fpath: str, **kwargs) -> None:
        """ CSV file to pandas DataFrame
        
        Parameters
        __________
        fpath: str
            File path to the CSV file
        
        Returns
        __________
        None
        """
        
        df = pd.read_csv(fpath, **kwargs)
        self.__init__(df)
                
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
    
    def clean(self) -> None:
        """ Drop NaN values and reset DataFrame index
        
        Parameters
        __________
        None
        
        Returns
        __________
        None
        """
        self.dropna(inplace=True)
        self.reset_index(drop=True, inplace=True)
    
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
        
        self.clean()
            
    def align(self, col1: str, col2: str) -> None:
        """ Align two columns through cross-correlation
        -> col2 will be shifted to match col1
        
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
        
        lag_idx = self._lag(col1, col2)
        self[col2] = self[col2].shift(-lag_idx)
        self.clean()
            
    def _lag(self, col1: str, col2: str) -> None:
        """ Calculate the lag between two columns
        
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
        cross_corr = np.correlate(self[col1], self[col2], mode='full')
        
        lag_idx = abs(cross_corr.argmax() - len(self[col2]))
        
        return lag_idx
        
    def _offset(self, col1: str, col2: str) -> None:
        """ Calculate the offset in mean between two columns
        
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
        return np.mean(self[col1]) - np.mean(self[col2])