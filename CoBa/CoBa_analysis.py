import numpy as np 
import matplotlib.pyplot as plt
import h5py
import copy

import bilby

# The population files -- use the uniform masses for now
BBH_pop_filename = "./18321_1yrCatalogBBH.h5"
BNS_pop_filename = "./18321_1yrCatalogBNS.h5"
BNS_gauss_pop_filename = "./18321_1yrCatalogBNSmassGauss.h5"
FILENAMES_DICT = {"BBH": BBH_pop_filename, 
                  "BNS": BNS_pop_filename, 
                  "BNS_gauss": BNS_gauss_pop_filename}

# Parameters that are in these files
ALL_BBH_KEYS = ['Mc', 'Phicoal', 'chi1', 'chi1x', 'chi1y', 'chi1z', 'chi2', 'chi2x', 'chi2y', 'chi2z', 'dL', 'dec', 'eta', 'iota', 'm1_source', 'm2_source', 'phi', 'phi12', 'phiJL', 'psi', 'ra', 'tGPS', 'tcoal', 'theta', 'thetaJN', 'tilt1', 'tilt2', 'z']
ALL_BNS_KEYS = copy.deepcopy(ALL_BBH_KEYS)
ALL_BNS_KEYS += ["Lambda1", "Lambda2"]

ALL_KEYS_DICT = {"BBH": ALL_BBH_KEYS, 
                 "BNS": ALL_BNS_KEYS, 
                 "BNS_gauss": ALL_BNS_KEYS}

def translate_for_bilby(params: dict):
    """Simply add the required key names for bilby"""
    
    # Masses
    params["mass_1"] = params["m1_source"]
    params["mass_2"] = params["m2_source"]
    
    # Spins
    params["a_1"] = params["chi1"]
    params["a_2"] = params["chi2"]
    params["tilt_1"] = params["tilt1"]
    params["tilt_2"] = params["tilt2"]
    
    # Extrinsic parameters
    params["luminosity_distance"] = params["dL"] * 1000.0 # from Gpc to Mpc
    params["theta_jn"] = params["thetaJN"]
    params["phase"] = params["phi"]
    params["geocent_time"] = params["tGPS"]
    
    return params

CoBa_events_dict = {}
CoBa_events_dict["BBH"] = {}
CoBa_events_dict["BNS"] = {}

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

def get_CoBa_event(pop_str: str, idx: int):
    """
    Get a specific event from one of the CoBa catalogs
    """            
    example_signal = {}
    for key in CoBa_events_dict[pop_str].keys():
        example_signal[key] = CoBa_events_dict[pop_str][key][idx]

example_signal = get_CoBa_event("BBH", 1)
example_signal = translate_for_bilby(example_signal)
print(f"Example BBH signal: {example_signal}")

f_sampling = 4096.0
f_min = 20.0
duration = 24.0

waveform_arguments = {
    "waveform_approximant": "IMRPhenomPv2",
    "reference_frequency": f_min,
    "minimum_frequency": f_min
}

waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=f_sampling,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments
)

# Define the ET and CE detectors
ifos = bilby.gw.detector.InterferometerList(["ET", "CE"])

start_time = example_signal['tGPS'] - duration + 2

ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=f_sampling,
    duration=duration, 
    start_time=start_time)
ifos.inject_signal(waveform_generator=waveform_generator, parameters=example_signal)