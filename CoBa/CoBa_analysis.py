import numpy as np 
import tqdm
import matplotlib.pyplot as plt
import bilby
import utils
import json

# Make bilby silent, otherwise will print for each injected signal
bilby.core.utils.logger.setLevel("ERROR")

for pop_str in ["BBH"]: # ["BBH", "BNS"]
    
    # Will store the SNR in this dict for the entire population, then add it to the catalogue at the end
    all_snr_dict = {"ET1": [], "ET2": [], "ET3": [], "ET": [], "CE": []}
    is_tidal = pop_str == "BNS"
    if pop_str == "BBH":
        N = utils.N_BBH_COBA
    else:
        N = utils.N_BNS_COBA
    
    for idx in tqdm.tqdm(range(N)):
        event = utils.get_CoBa_event("BNS", idx)
        event = utils.translate_CoBa_to_bilby(event)
        snr_dict = utils.inject_and_get_SNR(event)
        
        # Add the SNR to the complete list
        all_snr_dict["ET1"].append(snr_dict["ET1"])
        all_snr_dict["ET2"].append(snr_dict["ET2"])
        all_snr_dict["ET3"].append(snr_dict["ET3"])
        all_snr_dict["ET"].append(snr_dict["ET"])
        all_snr_dict["CE"].append(snr_dict["CE"])
        
    # Add the SNR to the CoBa catalogue
    utils.CoBa_events_dict[pop_str]["ET1_SNR"] = all_snr_dict["ET1"]
    utils.CoBa_events_dict[pop_str]["ET2_SNR"] = all_snr_dict["ET2"]
    utils.CoBa_events_dict[pop_str]["ET3_SNR"] = all_snr_dict["ET3"]
    utils.CoBa_events_dict[pop_str]["CE"] = all_snr_dict["CE"]
    
    # Save it as JSON:
    with open(f"CoBa_events_{pop_str}.json", "w") as f:
        json.dump(utils.CoBa_events_dict, f, indent = 4)