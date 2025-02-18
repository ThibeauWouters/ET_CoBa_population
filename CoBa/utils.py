import numpy as np 
import matplotlib.pyplot as plt
import h5py
import copy
import bilby

# The population files -- use the uniform masses for now
BBH_POP_FILENAME = "./18321_1yrCatalogBBH.h5"
BNS_POP_FILENAME = "./18321_1yrCatalogBNS.h5"
BNS_GAUSS_POP_FILENAME = "./18321_1yrCatalogBNSmassGauss.h5"
FILENAMES_DICT = {"BBH": BBH_POP_FILENAME, 
                  "BNS": BNS_POP_FILENAME, 
                  "BNS_gauss": BNS_GAUSS_POP_FILENAME}

# Parameters that are in these files
ALL_BBH_KEYS = ['Mc', 'Phicoal', 'chi1', 'chi1x', 'chi1y', 'chi1z', 'chi2', 'chi2x', 'chi2y', 'chi2z', 'dL', 'dec', 'eta', 'iota', 'm1_source', 'm2_source', 'phi', 'phi12', 'phiJL', 'psi', 'ra', 'tGPS', 'tcoal', 'theta', 'thetaJN', 'tilt1', 'tilt2', 'z']
ALL_BNS_KEYS = copy.deepcopy(ALL_BBH_KEYS)
ALL_BNS_KEYS += ["Lambda1", "Lambda2"]

ALL_KEYS_DICT = {"BBH": ALL_BBH_KEYS, 
                 "BNS": ALL_BNS_KEYS, 
                 "BNS_gauss": ALL_BNS_KEYS}

def translate_CoBa_to_bilby(params: dict):
    """
    Translate some of the parameter names from the CoBa catalogue study to the bilby parameter names.
    """
    
    # Masses (convert chirp mass to the source frame)
    params["Mc_source"] = params["Mc"] / (1 + params["z"])
    params["mass_1_source"] = params["m1_source"]
    params["mass_2_source"] = params["m2_source"]
    
    params["mass_1"] = params["m1_source"] * (1 + params["z"])
    params["mass_2"] = params["m2_source"] * (1 + params["z"])
    
    params["chirp_mass"] = params["Mc"]
    
    # Spins
    params["a_1"] = params["chi1"]
    params["a_2"] = params["chi2"]
    params["tilt_1"] = params["tilt1"]
    params["tilt_2"] = params["tilt2"]
    
    # Extrinsic parameters
    params["dL"] = params["dL"] * 1000.0 # from Gpc to Mpc
    params["luminosity_distance"] = params["dL"]
    params["theta_jn"] = params["thetaJN"]
    params["phase"] = params["phi"]
    params["geocent_time"] = params["tGPS"]
    params["redshift"] = params["z"]
    
    return params

CoBa_events_dict = {}
CoBa_events_dict["BBH"] = {}
CoBa_events_dict["BNS"] = {}

def disable_transverse_spins(parameters: dict):
    """
    Disable the transverse spins in the dictionary.
    """
    parameters["chi1x"] = 0.0
    parameters["chi1y"] = 0.0
    parameters["chi1z"] = parameters["chi1"]
    
    parameters["chi2x"] = 0.0
    parameters["chi2y"] = 0.0
    parameters["chi2z"] = parameters["chi2"]
    
    parameters["phiJL"] = 0.0
    parameters["phi12"] = 0.0
    
    parameters["tilt_1"] = 0.0
    parameters["tilt_2"] = 0.0
    
    return parameters

# Initialize the dicts with the keys
print("Loading the CoBa catalogs...")
for pop_str in ["BBH", "BNS"]:
    for key in ALL_KEYS_DICT[pop_str]:
        CoBa_events_dict[pop_str][key] = []

for pop_str in ["BBH", "BNS"]:
    filename = FILENAMES_DICT[pop_str]
    with h5py.File(filename, "r") as file:
        for key in file.keys():
            CoBa_events_dict[pop_str][key] = np.array(file[key])
print("Loading the CoBa catalogs... DONE")

N_BBH_COBA = len(CoBa_events_dict["BBH"]["Mc"])
N_BNS_COBA = len(CoBa_events_dict["BNS"]["Mc"])

print(f"CoBa has {N_BBH_COBA} BBH and {N_BNS_COBA} signals.")

def get_CoBa_event(pop_str: str, idx: int):
    """
    Get a specific event from one of the CoBa catalogs
    """            
    example_signal = {}
    for key in CoBa_events_dict[pop_str].keys():
        example_signal[key] = CoBa_events_dict[pop_str][key][idx]
        
    return example_signal
