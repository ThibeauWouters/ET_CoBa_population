import numpy as np 
import tqdm
import bilby
import utils
import json

# Make bilby silent, otherwise will print for each injected signal
bilby.core.utils.logger.setLevel("ERROR")

for pop_str in ["BBH", "BNS"]:
    
    print(f"Now looping over the {pop_str} population...")
    
    # Get the correct number of events to loop over
    if pop_str == "BBH":
        N = utils.N_BBH_COBA
    else:
        N = utils.N_BNS_COBA
    
    # Toggle tidal effects to be added if we are looking at BNS
    is_tidal = pop_str == "BNS"
    
    # We will store the SNR in this dict for the entire population, then add it to the catalogue at the end
    all_snr_dict = {"ET1": np.zeros(N),
                    "ET2": np.zeros(N),
                    "ET3": np.zeros(N),
                    "ET": np.zeros(N),
                    "CE": np.zeros(N)}
    
    for idx in tqdm.tqdm(range(N)):
        event = utils.get_CoBa_event(pop_str, idx)
        event = utils.translate_CoBa_to_bilby(event)
        snr_dict = utils.inject_and_get_SNR(event)
        
        # Add the SNR to the complete list
        for key in list(all_snr_dict.keys()):
            all_snr_dict[key][idx] = snr_dict[key]
        
    # Now add all the SNRs to the CoBa catalogue dict
    for key in list(all_snr_dict.keys()):
        utils.CoBa_events_dict[pop_str][f"{key}_SNR"] = all_snr_dict[key]
        
    # NOTE: Numpy arrays are not compatible with JSON, so call tolist() before saving
    for key in list(utils.CoBa_events_dict[pop_str].keys()):
        utils.CoBa_events_dict[pop_str][key] = utils.CoBa_events_dict[pop_str][key].tolist()
        
    # Save it as JSON:
    filename = f"CoBa_events_{pop_str}.json"
    print(f"Saving the updated CoBa catalogue, with the SNRs, to {filename}")
    with open(filename, "w") as f:
        json.dump(utils.CoBa_events_dict[pop_str], f, indent = 4)