import numpy as np 
import matplotlib.pyplot as plt
import h5py
import copy
import bilby

#################
### CONSTANTS ###
#################

# Population files -- we use the uniform masses for BNS for now, ignore Gaussian
BBH_POP_FILENAME = "./18321_1yrCatalogBBH.h5"
BNS_POP_FILENAME = "./18321_1yrCatalogBNS.h5"
BNS_GAUSS_POP_FILENAME = "./18321_1yrCatalogBNSmassGauss.h5"
FILENAMES_DICT = {"BBH": BBH_POP_FILENAME, 
                  "BNS": BNS_POP_FILENAME, 
                  "BNS_gauss": BNS_GAUSS_POP_FILENAME}

# Parameters that are in these files
ALL_BBH_KEYS = ['Mc', 'Phicoal', 'chi1', 'chi1x', 'chi1y', 'chi1z', 'chi2', 'chi2x', 'chi2y', 'chi2z', 'dL', 'dec', 'eta', 'iota', 'm1_source', 'm2_source', 'phi', 'phi12', 'phiJL', 'psi', 'ra', 'tGPS', 'tcoal', 'theta', 'thetaJN', 'tilt1', 'tilt2', 'z']
ALL_BNS_KEYS = copy.deepcopy(ALL_BBH_KEYS)
ALL_BNS_KEYS += ['Lambda1', 'Lambda2']

ALL_KEYS_DICT = {"BBH": ALL_BBH_KEYS, 
                 "BNS": ALL_BNS_KEYS, 
                 "BNS_gauss": ALL_BNS_KEYS}

### Reading/Loading the catalog
CoBa_events_dict = {}
CoBa_events_dict["BBH"] = {}
CoBa_events_dict["BNS"] = {}

for pop_str in ["BBH", "BNS"]:
    for key in ALL_KEYS_DICT[pop_str]:
        CoBa_events_dict[pop_str][key] = []

# Now read and save all 
print("Loading the CoBa catalogs...")
for pop_str in ["BBH", "BNS"]:
    filename = FILENAMES_DICT[pop_str]
    with h5py.File(filename, "r") as file:
        all_keys = list(file.keys())
        print(f"All keys listed for {pop_str} are {all_keys}")
        for key in all_keys:
            CoBa_events_dict[pop_str][key] = np.array(file[key])

N_BBH_COBA = len(CoBa_events_dict["BBH"]["Mc"])
N_BNS_COBA = len(CoBa_events_dict["BNS"]["Mc"])
print(f"CoBa has {N_BBH_COBA} BBH and {N_BNS_COBA} signals.")

# We overwrite the spins with their z-components
CoBa_events_dict["BNS"]["chi1"] = CoBa_events_dict["BNS"]["chi1z"]
CoBa_events_dict["BNS"]["chi2"] = CoBa_events_dict["BNS"]["chi2z"]

# For the BNS keys, pad the length with an array filled with zeroes (no precession for BNS)
for key in CoBa_events_dict["BNS"].keys():
    if len(CoBa_events_dict["BNS"][key]) < N_BNS_COBA:
        print(f"Padding {key} in BNS catalog")
        CoBa_events_dict["BNS"][key] = np.zeros(N_BNS_COBA)
print("Loading the CoBa catalogs... DONE")

# TODO: this is for debug to make sure everything is OK but remove later on
# Show the catalogs
print(f"BBH catalog shown:")
print(CoBa_events_dict["BBH"])

print(f"BNS catalog shown:")
print(CoBa_events_dict["BNS"])

#################
### UTILITIES ###
#################

def translate_CoBa_to_bilby(params: dict) -> dict:
    """
    Translate some of the parameter names from the CoBa catalogue study to the bilby parameter names.
    The idea is to just add the parameters to the dict, so we do not remove the old keys.
    
    Args:
        parameters (dict): The dictionary of parameters to modify.
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
    
    # For BNS, translate the tidal parameters
    if "Lambda1" in params.keys():
        params["lambda_1"] = params["Lambda1"]
        params["lambda_2"] = params["Lambda2"]
    
    return params

def disable_transverse_spins(parameters: dict) -> dict:
    """
    Disable the transverse spins in the parameters dictionary by setting the relevant parameters to zero.
    
    Args:
        parameters (dict): The dictionary of parameters to modify.
    """
    parameters["chi1x"] = 0.0
    parameters["chi1y"] = 0.0
    parameters["chi2x"] = 0.0
    parameters["chi2y"] = 0.0
    
    parameters["phiJL"] = 0.0
    parameters["phi12"] = 0.0
    parameters["thetaJN"] = 0.0
    parameters["tilt_1"] = 0.0
    parameters["tilt_2"] = 0.0
    
    parameters["chi1z"] = parameters["chi1"]
    parameters["chi2z"] = parameters["chi2"]
    
    return parameters

def get_CoBa_event(pop_str: str, idx: int) -> dict:
    """
    Get a specific event by specifying its index from the chosen CoBa catalogs.
    
    Args:
        pop_str (str): String, either BBH or BNS, to choose the catalog
        idx (int): Index number of the event for which we fetch the parameters.
    """
    if pop_str not in ["BBH", "BNS"]:
        raise ValueError(f"Population string {pop_str} not recognized, must be either BBH or BNS.")
    parameters = {}
    for key in ALL_KEYS_DICT[pop_str]:
        parameters[key] = CoBa_events_dict[pop_str][key][idx]
        
    return parameters


def inject_and_get_SNR(parameters: dict, 
                       f_min: float = 20.0,
                       f_sampling: float = 4096.0,
                       use_transverse_spins: bool = False,
                       is_tidal: bool = False) -> dict:
    """
    Inject a signal into the detectors and return the SNR.

    Args:
        parameters (dict): The parameters of the signal to be injected.
        f_min (float): The minimum frequency of the signal.
        f_sampling (float): The sampling frequency of the signal.
        use_transverse_spins (bool): Whether or not to use transverse spins (PhenomPv2 vs PhenomD). Put to False for BBH.
    """
    
    # TODO: if time permits, then check if disable transverse spins is needed for BNS or if this is OK after adding padding
    
    # We can choose to force aligned spins, and if BNS (tidal) waveform, there is no precession
    if not use_transverse_spins or is_tidal:
        parameters = disable_transverse_spins(parameters)
        approximant_str = "IMRPhenomD"
    else:
        approximant_str = "IMRPhenomPv2"
        
    if is_tidal:
        approximant_str += "_NRTidalv2"
        
    duration = bilby.gw.utils.calculate_time_to_merger(
            f_min,
            parameters['mass_1'],
            parameters['mass_2'],
            safety = 1.2)
    
    # Round to nearest above power of 2, making sure at least 4 seconds are used
    duration = np.ceil(duration + 4.0)
    
    waveform_arguments = {
        "waveform_approximant": approximant_str,
        "reference_frequency": f_min,
        "minimum_frequency": f_min,
        "maximum_frequency": 0.5 * f_sampling,
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
    
    # This is required since otherwise, the f_min is 5 Hz by default and durations will not match!
    for ifo in ifos:
        ifo.minimum_frequency = f_min
        ifo.maximum_frequency = 0.5 * f_sampling
    
    start_time = parameters['geocent_time'] - duration + 2.0
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=f_sampling,
        duration=duration, 
        start_time=start_time
        )
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=parameters)
    
    # Now fetch the SNR:
    snr_dict = {}
    for ifo in ifos:
        snr = ifo.meta_data['optimal_SNR']
        snr_dict[ifo.name] = snr
        
    # For ET, save the network SNR of the 3 ifos
    snr_dict["ET"] = np.sqrt(snr_dict["ET1"]**2 + snr_dict["ET2"]**2 + snr_dict["ET3"]**2)
    return snr_dict