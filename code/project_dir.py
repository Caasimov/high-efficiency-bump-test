paths_hdf5_main = {
    "AGARD-AR-144_A": "data/hdf5/motionlog-20240301_133202.hdf5",
    "AGARD-AR-144_B+E": "data/hdf5/motionlog-20240301_141239.hdf5",
    "MULTI-SINE": "data/hdf5/motionlog-20240301_144109.hdf5",
    "BUMP": "data/hdf5/motionlog-20240301_150040.hdf5",
    "PMD": "data/hdf5/motionlog-20240301_150320.hdf5"
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
    "AGARD-AR-144_A": 'data/json/srs-agard144a.json',
    "AGARD-AR-144_B": 'data/json/srs-agard144b.json',
    "AGARD-AR-144_D": 'data/json/srs-agard144d.json', 
    "AGARD-AR-144_E": 'data/json/srs-agard144e.json',
    "MULTI-SINE_1": 'data/json/srs-test-motion-sines1.json',
    "MULTI-SINE_2": 'data/json/srs-test-motion-sines2.json',
    "MULTI-SINE_3": 'data/json/srs-test-motion-sines3.json'
}
plot_dir = {}