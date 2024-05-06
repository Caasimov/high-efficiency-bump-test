### CONSTANTS ###

c1 = 'blue'
c2 = 'red'
marker = 'x'
linestyle1 = '--'
figsize_default = (8, 5)
figsize_deBode = (12, 8)

### FILEPATHS ###

paths_hdf5_main = {
    "AGARD-AR-144_A": "data/raw/hdf5/motionlog-20240301_133202.hdf5",
    "AGARD-AR-144_B+E": "data/raw/hdf5/motionlog-20240301_141239.hdf5",
    "MULTI-SINE": "data/raw/hdf5/motionlog-20240301_144109.hdf5",
    "BUMP": "data/raw/hdf5/motionlog-20240301_150040.hdf5",
    "BUMP+": "data/raw/hdf5/motionlog-20240425_091135.hdf5",
    "PMD": "data/raw/hdf5/motionlog-20240301_150320.hdf5"
    }

def paths_hdf5_cmd(dof: str) -> dict:
    """ Return dictionary of column paths for a degree of freedom
    
    Parameters
    __________
    dof: str
        Degree of freedom to return paths for
    
    Returns
    __________
    dict
        Dictionary of column paths
    """
    return {
        "t": "data/commanded/tick",
        "pos_cmd": f"data/commanded/data/{dof}",
        "vel_cmd": f"data/commanded/data/{dof}dot",
        "acc_cmd": f"data/commanded/data/{dof}dotdot",
    }

paths_hdf5_mes = {
    "t": "data/measured/tick",
    "pos_mes": "data/measured/data/actual_pos"
    }

paths_json = {
    "AGARD-AR-144_A": "data/raw/json/srs-agard144a.json",
    "AGARD-AR-144_B": "data/raw/json/srs-agard144b.json",
    "AGARD-AR-144_D": "data/raw/json/srs-agard144d.json",
    "AGARD-AR-144_E": "data/raw/json/srs-agard144e.json",
    "BUMP": "data/raw/json/srs-test-motion-bump.json",
    "BUMP+": "data/raw/json/srs-test-motion-bump-loc.json",
    "MULTI-SINE_1": "data/raw/json/srs-test-motion-sines1.json",
    "MULTI-SINE_2": "data/raw/json/srs-test-motion-sines2.json",
    "MULTI-SINE_3": "data/raw/json/srs-test-motion-sines3.json"
}
paths_plots = {
    "I/O": "plots/IO",
    "signal": "plots/signals",
    "deBode": "plots/deBode",
    "spectrum": "plots/spectra"
    }