"""
Time to check out how the overlapping signals population looks like (difference in time of coalescence, etc.)
"""

import os
import json
import numpy as np 
import matplotlib.pyplot as plt
import corner
import scipy.stats as stats

from astropy.time import Time

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        density=True,
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

# This is where the postprocessed (= with SNR computed) catalogs live on the CIT cluster:
COBA_LOCATION = "/home/thibeau.wouters/projects/ET_CoBa_population/CoBa/"
BBH_CATALOG_FILENAME = os.path.join(COBA_LOCATION, "CoBa_events_BBH.json")
BBH_CATALOG_5HZ_FILENAME = os.path.join(COBA_LOCATION, "CoBa_events_BBH_5Hz.json")
BNS_CATALOG_FILENAME = os.path.join(COBA_LOCATION, "CoBa_events_BNS.json")

# Load catalogs just to have a look:
with open(BBH_CATALOG_FILENAME, "r") as f:
    BBH_CATALOG = json.load(f)
    
with open(BBH_CATALOG_5HZ_FILENAME, "r") as f:
    BBH_5HZ_CATALOG = json.load(f)
    
with open(BNS_CATALOG_FILENAME, "r") as f:
    BNS_CATALOG = json.load(f)
    
int_to_str_dict = {0: "BBH", 1: "BNS"}

def rescale_time(tGPS: np.array, resample: bool = True):
    """
    Because the CoBa paper sucks, the tGPS time is not in 1 year, but in 10 years. Rescale that to get in 1 year again.
    NOTE: since the tGPS is rounded to seconds, this is not so informative -- therefore, we can also resample our own tGPS.
    """
    # Define target range (full year 2030)
    t_start_2030 = 1577491181
    t_end_2030 = 1609027181
    
    t_min, t_max = np.min(tGPS), np.max(tGPS)

    if resample:
        print(f"Note: we are going to sample our own GPS time since the CoBa ones are bad")
        tGPS_rescaled = np.random.uniform(t_start_2030, t_end_2030, len(tGPS))
        tGPS_rescaled = np.sort(tGPS_rescaled)
    else:
        tGPS_rescaled = t_start_2030 + (tGPS - t_min) / (t_max - t_min) * (t_end_2030 - t_start_2030)
        
    return tGPS_rescaled
    
def make_dt_histogram(snr_cutoff: float = 8,
                      which_snr: str = 'network',
                      max_dt: float = 0.5):
    """
    Make a histogram of the time between consecutive events in the CoBa catalogs.
    
    Args:
        snr_cutoff: float, the SNR cutoff to use
        which_snr: str, the SNR to use ('network' or 'ET' or 'CE')
        max_dt: float, the maximum difference in time to consider for the plot
    """
    # Note: tcoal is better, since tGPS is rounded to seconds and not so informative
    keys_of_interest = ['tGPS', 'tcoal', 'ET_SNR', 'CE_SNR']
    time_key = 'tGPS'

    # Build a catalogue which is sorted by coalescence time 
    my_catalogue = {}
    for i, catalog in enumerate([BBH_CATALOG, BNS_CATALOG]):
        for key in keys_of_interest:
            if i == 0:
                my_catalogue[key] = np.array(catalog[key])
            else:
                # Append the BNS catalog to the BBH catalog
                my_catalogue[key] = np.concatenate((my_catalogue[key], np.array(catalog[key])))
    
    my_catalogue["network_SNR"] = np.sqrt(my_catalogue['ET_SNR']**2 + my_catalogue['CE_SNR']**2)
    my_catalogue["type"] = np.concatenate((np.zeros(len(BBH_CATALOG[time_key])), np.ones(len(BNS_CATALOG[time_key]))))

    # Sort the events based on the time of coalescence
    sorting = np.argsort(my_catalogue[time_key])
    for key, value in my_catalogue.items():
        my_catalogue[key] = value[sorting]

    mask = my_catalogue[f"{which_snr}_SNR"] > snr_cutoff
    kept = np.sum(mask) / len(mask)
    print(f"Keeping {kept:.5f} of the events with {which_snr} > {snr_cutoff}")
    
    for i, (key, value) in enumerate(my_catalogue.items()):
        my_catalogue[key] = value[mask]
        if i == 0:
            print("Length after the cut:")
            print(len(my_catalogue[key]))
        
    # Print the SNR range
    print(f"SNR range: {np.min(my_catalogue[f'{which_snr}_SNR']):.2f} - {np.max(my_catalogue[f'{which_snr}_SNR']):.2f}")
    
    # Fix the tGPS to be in 1 year, not 10 years
    my_catalogue['tGPS'] = rescale_time(my_catalogue['tGPS'])
        
    dt = np.diff(my_catalogue[time_key])
    
    tGPS_min = np.min(my_catalogue["tGPS"])
    tGPS_max = np.max(my_catalogue["tGPS"])
    
    time_min = Time(tGPS_min, format='gps')
    time_max = Time(tGPS_max, format='gps')
    
    print(f"First time: {time_min.iso}")
    print(f"Last time: {time_max.iso}")

    if time_key == "tcoal":
        print(f"Converting tcoal to seconds")
        dt = dt * 86400
    
    my_catalogue['dt'] = dt
    
    dt_dict = {"BBH+BBH": [],
               "BBH+BNS": [],
               "BNS+BBH": [],
               "BNS+BNS": []}
    
    # Iterate over the constructed events and check sort dt's based on type
    for i in range(len(my_catalogue['type']) - 1):
        this_event_type = my_catalogue['type'][i]
        next_event_type = my_catalogue['type'][i+1]
        
        this_event_time = my_catalogue[time_key][i]
        next_event_time = my_catalogue[time_key][i+1]
        
        dt = next_event_time - this_event_time
        if time_key == "tcoal":
            print(f"Converting tcoal to seconds")
            dt = dt * 86400 # convert to seconds
        
        dt_dict[f"{int_to_str_dict[this_event_type]}+{int_to_str_dict[next_event_type]}"].append(dt)

    any_negative = np.any(my_catalogue['dt'] < 0)
    print("Any negative dt in here?")
    print(any_negative)

    # Make a histogram of the dt:
    hist_kwargs = {
        "bins": 100,
        "histtype": "step",
        "linewidth": 2,
        "density": True
    }
    plt.figure(figsize = (12, 8))
    
    for key, col in zip(["BBH+BBH", "BBH+BNS", "BNS+BNS"], ["blue", "red", "green"]):
        print(f"Checking {key} . . .")
        samples = np.array(dt_dict[key])
        
        # Estimate lambda (rate parameter)
        lambda_hat = 1 / np.mean(samples)
        print(f"Estimated lambda: {lambda_hat:.4f}")
        
        # Perform Kolmogorov-Smirnov test
        d, p_value = stats.kstest(samples, 'expon', args=(0, 1/lambda_hat))
        print(f"KS test statistic: {d:.4f}, p-value: {p_value:.4f}")
        
        # Make the plot but mask up to given dt
        mask = np.array(samples) < max_dt
        kept = np.sum(mask) / len(mask)
        print(f"{kept*100:.2f}% of the events have dt below {max_dt}")
        plt.hist(samples[mask], label = key, color = col, **hist_kwargs)
        # x = np.linspace(0, np.max(samples), 1_000)
        # plt.plot(x, lambda_hat * np.exp(-lambda_hat * x), color = "red", linestyle = "--", lw=2, label='Fitted Exponential')
    
    plt.xlabel(r"$\Delta t$ [s]")
    plt.ylabel("Density")
    plt.title("Time between consecutive events")
    plt.legend()
    
    name = "./figures/dt_histogram.png"
    plt.savefig(name)
    plt.savefig(name.replace(".png", ".pdf"))
    plt.close()
    
    ### Considering only BBH events:
    bbh = my_catalogue['type'] == 0
    dt = np.diff(my_catalogue[time_key][bbh])
    if time_key == "tcoal":
        dt = dt * 86400 # convert to seconds
    
    # Estimate lambda (rate parameter)
    print(f"BBH only checking now")
    samples = dt
    lambda_hat = 1 / np.mean(samples)
    print(f"Estimated lambda: {lambda_hat:.4f}")
    
    # Perform Kolmogorov-Smirnov test
    d, p_value = stats.kstest(samples, 'expon', args=(0, 1/lambda_hat))
    print(f"KS test statistic: {d:.4f}, p-value: {p_value:.4f}")

    # Make a histogram of the dt:
    hist_kwargs = {
        "bins": 100,
        "histtype": "step",
        "linewidth": 2,
        "density": True
    }
    plt.figure(figsize = (12, 8))
    plt.hist(dt, color = "black", **hist_kwargs)
    
    plt.xlabel(r"$\Delta t$ [s]")
    plt.ylabel("Density")
    plt.title("Time between consecutive events")
    plt.legend()
    
    name = "./figures/dt_histogram_bbh_only.png"
    plt.savefig(name)
    plt.savefig(name.replace(".png", ".pdf"))
    plt.close()
    
    ### Considering only BNS events:
    bns = my_catalogue['type'] == 1
    dt = np.diff(my_catalogue[time_key][bns])
    if time_key == "tcoal":
        dt = dt * 86400 # convert to seconds
    
    # Estimate lambda (rate parameter)
    print(f"BNS only checking now")
    samples = dt
    lambda_hat = 1 / np.mean(samples)
    print(f"Estimated lambda: {lambda_hat:.4f}")
    
    # Perform Kolmogorov-Smirnov test
    d, p_value = stats.kstest(samples, 'expon', args=(0, 1/lambda_hat))
    print(f"KS test statistic: {d:.4f}, p-value: {p_value:.4f}")

    # Make a histogram of the dt:
    hist_kwargs = {
        "bins": 100,
        "histtype": "step",
        "linewidth": 2,
        "density": True
    }
    plt.figure(figsize = (12, 8))
    plt.hist(dt, color = "black", **hist_kwargs)
    
    plt.xlabel(r"$\Delta t$ [s]")
    plt.ylabel("Density")
    plt.title("Time between consecutive events")
    plt.legend()
    
    name = "./figures/dt_histogram_bns_only.png"
    plt.savefig(name)
    plt.savefig(name.replace(".png", ".pdf"))
    plt.close()
    
def main():
    make_dt_histogram()
    
if __name__ == "__main__":
    main()