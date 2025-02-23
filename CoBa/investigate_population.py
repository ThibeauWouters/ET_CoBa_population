"""
Time to check out how the overlapping signals population looks like (difference in time of coalescence, etc.)
"""

import os
import json
import numpy as np 
import matplotlib.pyplot as plt
import corner

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
    
def make_dt_histogram(snr_cutoff: float = 8,
                      which_snr: str = 'network'):
    """
    Make a histogram of the time between consecutive events in the CoBa catalogs.
    """
    # Note: tcoal is better, since tGPS is rounded to seconds and not so informative
    keys_of_interest = ['tcoal', 'ET_SNR', 'CE_SNR']

    # Build a catalogue which is sorted by coalescence time 
    my_catalogue = {}
    for i, catalog in enumerate([BBH_CATALOG, BNS_CATALOG]):
        for key in keys_of_interest:
            if i == 0:
                my_catalogue[key] = np.array(catalog[key])
                my_catalogue["type"] = np.zeros(len(catalog[key]))
            else:
                # Append the BNS catalog to the BBH catalog
                my_catalogue[key] = np.concatenate((my_catalogue[key], np.array(catalog[key])))
                my_catalogue["type"] = np.concatenate((my_catalogue["type"], np.ones(len(catalog[key]))))

    my_catalogue["network_SNR"] = np.sqrt(my_catalogue['ET_SNR']**2 + my_catalogue['CE_SNR']**2)

    # Sort the events based on the time of coalescence
    sorting = np.argsort(my_catalogue['tcoal'])
    print(sorting)
    for key, value in my_catalogue.items():
        my_catalogue[key] = value[sorting]

    mask = my_catalogue[f"{which_snr}_SNR"] > snr_cutoff
    kept = np.sum(mask) / len(mask)
    print(f"Keeping {kept:.5f} of the events with {which_snr} > {snr_cutoff}")
    print(len(mask))

    for key, value in my_catalogue.items():
        print(key)
        print(value.shape)
        my_catalogue[key] = value[mask]
        
    dt = np.diff(my_catalogue['tcoal'])
    my_catalogue['dt'] = dt * 86400 # convert to seconds
    
    # Add dummy dt to account for different lengths
    my_catalogue['dt'] = np.concatenate(([0], my_catalogue['dt']))

    # Check out for sanity checking:
    max_number = 20
    print("Time of coalescence")
    print(my_catalogue['tcoal'][:max_number])

    print("dt")
    print(my_catalogue['dt'][:max_number])

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
    plt.figure()
    
    # All events together
    plt.hist(my_catalogue['dt'], color = "black", label = "All", **hist_kwargs)
    
    # Only BBH
    mask = my_catalogue['type'] == 0
    plt.hist(my_catalogue['dt'][mask], color = "blue", label = "BBH", **hist_kwargs)
    
    # Only BNS
    mask = my_catalogue['type'] == 1
    plt.hist(my_catalogue['dt'][mask], color = "red", label = "BNS", **hist_kwargs)
    plt.xlabel(r"$\Delta t$ [s]")
    plt.ylabel("Density")
    plt.title("Time between consecutive events")
    plt.legend()
    
    name = "./figures/dt_histogram.png"
    plt.savefig(name)
    plt.savefig(name.replace(".png", ".pdf"))
    plt.close()
    
def main():
    make_dt_histogram()
    
if __name__ == "__main__":
    main()