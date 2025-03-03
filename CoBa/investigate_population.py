"""
Time to check out how the overlapping signals population looks like (difference in time of coalescence, etc.)
"""

import os
import json
import numpy as np 
np.random.seed(0) # fix the random seed so that the resampled catalogue tGPS is always the same
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
    else:
        tGPS_rescaled = t_start_2030 + (tGPS - t_min) / (t_max - t_min) * (t_end_2030 - t_start_2030)
        
    return tGPS_rescaled

# Build a catalogue which is sorted by coalescence time 
my_catalogue = {}
BBH_CATALOG["Lambda1"] = np.zeros_like(BBH_CATALOG["Mc"])
BBH_CATALOG["Lambda2"] = np.zeros_like(BBH_CATALOG["Mc"])

for i, catalog in enumerate([BBH_CATALOG, BNS_CATALOG]):
    print(f"Catalog keys: {catalog.keys()}")
    for key in catalog.keys():
        if i == 0:
            my_catalogue[key] = np.array(catalog[key])
        else:
            # Append the BNS catalog to the BBH catalog
            my_catalogue[key] = np.concatenate((my_catalogue[key], np.array(catalog[key])))

# Compute network SNR, and also save the type of event
my_catalogue["network_SNR"] = np.sqrt(my_catalogue['ET_SNR']**2 + my_catalogue['CE_SNR']**2)
my_catalogue["type"] = np.concatenate((np.zeros(len(BBH_CATALOG["tGPS"])), np.ones(len(BNS_CATALOG["tGPS"]))))

# Fix the tGPS to be in 1 year, not 10 years (typo in the CoBa catalogue?)
my_catalogue['tGPS'] = rescale_time(my_catalogue['tGPS'])
# Now sort based on tGPS
idx = np.argsort(my_catalogue['tGPS'])
for key, value in my_catalogue.items():
    my_catalogue[key] = value[idx]
    
print(f"Showing my catalogue:")
print(my_catalogue.keys())
print(my_catalogue)

LIMIT_TO_ONE_DAY = True

if LIMIT_TO_ONE_DAY:
    print("Limiting to one day of data . . .")
    tGPS_min = np.min(my_catalogue['tGPS'])
    tGP_max = tGPS_min + 86_400 # 1 day in seconds
    
    mask = (my_catalogue['tGPS'] >= tGPS_min) & (my_catalogue['tGPS'] <= tGP_max)

    for key, value in my_catalogue.items():
        my_catalogue[key] = value[mask]
    
def make_dt_histogram(snr_cutoff: float = 0.0,
                      which_snr: str = 'network',
                      max_dt: float = 2.0,
                      check_exponential: bool = False,
                      dump_injections: bool = True):
    """
    Make a histogram of the time between consecutive events in the CoBa catalogs.
    
    Args:
        snr_cutoff: float, the SNR cutoff to use
        which_snr: str, the SNR to use ('network' or 'ET' or 'CE')
        max_dt: float, the maximum difference in time to consider for the plot (the "dangerous" region)
        check_exponential: Estimate and print the exponential distribution parameters and p-value of KS test
    """

    # Only keep above a certain SNR cutoff
    mask = my_catalogue[f"{which_snr}_SNR"] > snr_cutoff
    kept = np.sum(mask) / len(mask)
    print(f"Keeping {kept:.5f} of the events (number: {np.sum(mask)}) with {which_snr} > {snr_cutoff}")
    
    for i, (key, value) in enumerate(my_catalogue.items()):
        my_catalogue[key] = value[mask]
        
    # Print number of BBH and BNS events
    print(f"Number of BBH: {np.sum(my_catalogue['type'] == 0)}")
    print(f"Number of BNS: {np.sum(my_catalogue['type'] == 1)}")
        
    # Get differences in time without taking into account different source types
    dt = np.diff(my_catalogue["tGPS"])
    print(dt)
    
    print(f"Any negative dt? {np.any(dt < 0)}")
    
    triples_counter = 0
    for i in range(len(dt) - 1):
        this_dt = dt[i]
        next_dt = dt[i+1]
        
        if this_dt < max_dt and next_dt < max_dt:
            triples_counter = 0
    
    print(f"We have {triples_counter} events with triple overlaps with dt < {max_dt}")
        
    # Loop and consider only the events with SNR above the cutoff
    dt_dict = {"BBH+BBH": [],
               "BBH+BNS": [],
               "BNS+BBH": [],
               "BNS+BNS": []}
    
    # Iterate over the constructed events and check sort dt's based on type
    for i in range(len(my_catalogue['type']) - 1):
        this_event_type = my_catalogue['type'][i]
        next_event_type = my_catalogue['type'][i+1]
        
        this_event_time = my_catalogue["tGPS"][i]
        next_event_time = my_catalogue["tGPS"][i+1]
        
        dt = next_event_time - this_event_time
        dt_dict[f"{int_to_str_dict[this_event_type]}+{int_to_str_dict[next_event_type]}"].append(dt)

    # Make a plot (bar chart) of the number of populations
    plt.figure(figsize = (12, 8))
    
    x = [0, 1, 2, 3]
    keys = ["BBH+BBH", "BBH+BNS", "BNS+BBH", "BNS+BNS"]
    y = [len(dt_dict[k]) for k in keys]
    
    # Sort based on the number:
    sort_idx = np.argsort(y)[::-1]
    y = np.array(y)[sort_idx]
    keys = np.array(keys)[sort_idx]
    
    plt.bar(x, y, color = "blue")
    plt.xticks(x, keys)
    plt.ylabel("Number of consecutive events type")
    
    name = "./figures/overlaps_population_histogram.png"
    plt.savefig(name)
    plt.savefig(name.replace(".png", ".pdf"))
    plt.close()
    
    # Same figure but now apply the max dt cut
    plt.figure(figsize = (12, 8))
    x = [0, 1, 2, 3]
    keys = ["BBH+BBH", "BBH+BNS", "BNS+BBH", "BNS+BNS"]
    dt_dict_cut = {}
    for key, value in dt_dict.items():
        dt_dict_cut[key] = [dt for dt in value if dt < max_dt]
    y = [len(dt_dict_cut[k]) for k in keys]
    
    sort_idx = np.argsort(y)[::-1]
    y = np.array(y)[sort_idx]
    keys = np.array(keys)[sort_idx]
    
    plt.bar(x, y, color = "blue")
    plt.xticks(x, keys)
    plt.ylabel("Number of consecutive events type")
    
    name = "./figures/overlaps_population_histogram_dt_cut.png"
    plt.savefig(name)
    plt.savefig(name.replace(".png", ".pdf"))
    plt.close()

    # Make a histogram of the dt:
    hist_kwargs = {
        "bins": 100,
        "histtype": "step",
        "linewidth": 2,
        "density": True
    }
    plt.figure(figsize = (12, 8))
    
    for key, col in zip(["BBH+BBH", "BBH+BNS", "BNS+BBH", "BNS+BNS"], ["blue", "red", "orange", "green"]):
        # Get the current set of samples
        samples = np.array(dt_dict[key])
        
        # Estimate exponential distribution parameters and perform Kolmogorov-Smirnov test
        if check_exponential:
            lambda_hat = 1 / np.mean(samples)
            d, p_value = stats.kstest(samples, 'expon', args=(0, 1/lambda_hat))
            print(f"Estimated lambda: {lambda_hat:.4f}")
            print(f"KS test statistic: {d:.4f}, p-value: {p_value:.4f}")
        
        # Make the plot but mask up to given dt
        mask = np.array(samples) < max_dt
        kept = np.sum(mask) / len(mask)
        print(f"{np.sum(mask)} of {key} events ({kept*100:.2f}%): below {max_dt}")
        plt.hist(samples[mask], label = key, color = col, **hist_kwargs)
    
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
    dt = np.diff(my_catalogue["tGPS"][bbh])
    
    # Estimate lambda (rate parameter)
    samples = dt
    
    if check_exponential:
        lambda_hat = 1 / np.mean(samples)
        d, p_value = stats.kstest(samples, 'expon', args=(0, 1/lambda_hat))
        print(f"Estimated lambda: {lambda_hat:.4f}")
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
    
    name = "./figures/dt_histogram_bbh_only.png"
    plt.savefig(name)
    plt.savefig(name.replace(".png", ".pdf"))
    plt.close()
    
    ### Considering only BNS events:
    bns = my_catalogue['type'] == 1
    dt = np.diff(my_catalogue["tGPS"][bns])
    # Estimate lambda (rate parameter)
    samples = dt
    
    if check_exponential:
        lambda_hat = 1 / np.mean(samples)
        d, p_value = stats.kstest(samples, 'expon', args=(0, 1/lambda_hat))
        print(f"Estimated lambda: {lambda_hat:.4f}")
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
    
    name = "./figures/dt_histogram_bns_only.png"
    plt.savefig(name)
    plt.savefig(name.replace(".png", ".pdf"))
    plt.close()
    
    # Dump the final set of injections to a JSON file:
    if dump_injections:
        filename = os.path.join(COBA_LOCATION, "CoBa_events_one_day.json")
        print(f"Dumping one day of data to {filename}")
        
        # Make all np arrays into lists for JSON compatibility:
        for key, value in my_catalogue.items():
            my_catalogue[key] = value.tolist()
        
        with open(filename, "w") as f:
            json.dump(my_catalogue, f)
    
def get_one_day(number_of_random_seeds: int = 1,
                network_snr_threshold: float = 12,
                dump_injections: bool = False):
    """Get the population properties for a single day of observance of ET and CE"""
    
    print(f"Creating the population to analyze for 1 day of ET and CE observance . . . ")
    
    for seed in range(number_of_random_seeds):
        
        # Set the random seed
        print(f" === Random seed {seed} ===")
        np.random.seed(seed)
        
        # Build a catalogue which is sorted by coalescence time but with all the keys
        one_day = {}

        for i, catalog in enumerate([BBH_CATALOG, BNS_CATALOG]):
            for key in list(BBH_CATALOG.keys()):
                if i == 0:
                    one_day[key] = np.array(catalog[key])
                else:
                    # Append the BNS catalog to the BBH catalog
                    one_day[key] = np.concatenate((one_day[key], np.array(catalog[key])))

        # Compute network SNR, and also save the type of event
        one_day["network_SNR"] = np.sqrt(one_day['ET_SNR']**2 + one_day['CE_SNR']**2)
        one_day["type"] = np.concatenate((np.zeros(len(BBH_CATALOG["tGPS"])), np.ones(len(BNS_CATALOG["tGPS"]))))

        # Fix the tGPS to be in 1 year, not 10 years (typo in the CoBa catalogue?)
        one_day['tGPS'] = rescale_time(one_day['tGPS'])
        
        # Now sort based on tGPS
        idx = np.argsort(one_day['tGPS'])
        for key, value in one_day.items():
            one_day[key] = value[idx]
            
        # Limit to one day:
        tGPS_start = np.min(one_day['tGPS'])
        tGPS_end = tGPS_start + 86_400 # 1 day in seconds
        
        mask = (one_day['tGPS'] >= tGPS_start) & (one_day['tGPS'] <= tGPS_end)
        
        for key, value in one_day.items():
            one_day[key] = value[mask]
            
        print(f"Number of events with SNR above {network_snr_threshold} in 1 day of ET-CE: {len(one_day['tGPS'])}")
        print(f"Number of BBH: {np.sum(one_day['type'] == 0)}")
        print(f"Number of BNS: {np.sum(one_day['type'] == 1)}")
        print("   ")
        
    # Dump the final set of injections to a JSON file:
    if dump_injections:
        filename = os.path.join(COBA_LOCATION, "CoBa_events_one_day.json")
        print(f"Dumping one day of data to {filename}")
        
        # Make all np arrays into lists for JSON compatibility:
        for key, value in one_day.items():
            one_day[key] = value.tolist()
        
        with open(filename, "w") as f:
            json.dump(one_day, f)
            
    
def main():
    make_dt_histogram(dump_injections = True)
    # get_one_day(dump_injections = True)
    
if __name__ == "__main__":
    main()