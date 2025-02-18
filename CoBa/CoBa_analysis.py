import numpy as np 
import tqdm
import matplotlib.pyplot as plt

import bilby

import utils

# Get the catalog to show
bbh_catalog = utils.CoBa_events_dict["BBH"]

F_SAMPLING = 4096.0
F_MIN = 20.0

def inject_and_get_SNR(parameters: dict, 
                       f_min: float = F_MIN,
                       f_sampling: float = F_SAMPLING,
                       disable_transverse_spins: bool = True) -> dict:
    """
    Inject a signal into the detectors and return the SNR.

    Args:
        parameters (dict): The parameters of the signal to be injected.
        duration (float): Duration of the signal.
    """
    if disable_transverse_spins:
        parameters = utils.disable_transverse_spins(parameters)
        approximant_str = "IMRPhenomD"
    else:
        approximant_str = "IMRPhenomPv2"
    
    # print("parameters")
    # print(parameters)
    
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

# Loop over the events, make bilby silent from here on
bilby.core.utils.logger.setLevel("ERROR")
for idx in tqdm.tqdm(range(609, 610)):
    event = utils.get_CoBa_event("BBH", idx)
    event = utils.translate_CoBa_to_bilby(event)
    
    print("event")
    print(event)
    
    snr_dict = inject_and_get_SNR(event)
    
    if idx > 1_000:
        break