"""
Time to check out how the overlapping signals population looks like (difference in time of coalescence, etc.)
"""

import os
import json
import numpy as np 
np.random.seed(0) # fix the random seed so that the resampled catalogue tGPS is always the same
import matplotlib.pyplot as plt
import corner
import arviz
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


# Load the data from the one day ET-CE JSON file
json_file = "./CoBa_events_one_day.json"
with open(json_file, "r") as f:
    events = json.load(f)

# Convert all keys to numpy arrays
for key in events.keys():
    events[key] = np.array(events[key])
    
events["network_snr"] = np.sqrt(events["ET_SNR"]**2 + events["CE_SNR"]**2)

# BBH
mask = events["type"] == 0
print("mask")
print(mask)
nb_bbh_events = np.sum(mask)
bbh_events = {}
for key in events.keys():
    bbh_events[key] = events[key][mask]

# BNS
mask = events["type"] == 1
nb_bns_events = np.sum(mask)

bns_events = {}
for key in events.keys():
    bns_events[key] = events[key][mask]

# Show all the events in the catalogue    
print(f"All events: BBH = {nb_bbh_events}, BNS = {nb_bns_events}, total = {nb_bbh_events + nb_bns_events}")

# Show above certain SNR threshold
SNR_THRESHOLD = 12

# BBH
mask = (events["type"] == 0) & (events["network_snr"] > SNR_THRESHOLD)
nb_bbh_events = np.sum(mask)

# BNS
mask = (events["type"] == 1) & (events["network_snr"] > SNR_THRESHOLD)
nb_bns_events = np.sum(mask)
print(f"Above SNR {SNR_THRESHOLD}: BBH = {nb_bbh_events}, BNS = {nb_bns_events}, total = {nb_bbh_events + nb_bns_events}")


def make_plots(hdi_prob: float = 0.9):
    """
    Make plots of some parameters for one day, so we know what to expect in injections
    
    Args:
    hdi_prob: float, default = 0.9
        The probability for the highest density interval
    """
    
    figure_dir = "./figures/one_day/"
    
    # Make a histogram of the network SNR for BBH and BNS
    plt.figure(figsize=(8, 6))
    hist_kwargs = {"bins": 50, "histtype": "step", "linewidth": 2, "density": True}
    
    colors = ["blue", "red"]
    labels = ["BBH", "BNS"]
    catalog_list = [bbh_events, bns_events]
    
    for catalog, lab, col in zip(catalog_list, labels, colors):
        snr_values = catalog["network_snr"]
        med = np.median(snr_values)
        low, high = arviz.hdi(snr_values, hdi_prob=hdi_prob)
        high = high - med
        low = med - low
        
        print(f"For catalog {lab}, the 90% confidence interval is {med:.2f}-{low:.2f}+{high:.2f}")
        
        plt.hist(snr_values, color=col, label=lab, **hist_kwargs)
    
    plt.xlabel("Network SNR")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(figure_dir, "network_snr.pdf"))
    plt.close()
    
    # Make a plot of the time of coalescence differences
    tcoal = events["tGPS"]
    dt = np.diff(tcoal)
    
    plt.figure(figsize=(8, 6))
    hist_kwargs = {"bins": 50, "histtype": "step", "linewidth": 2, "density": True}
    plt.hist(dt, color="blue", **hist_kwargs)
    danger_threshold = 2.0
    danger_number = int(np.sum(dt < danger_threshold))
    very_danger_threshold = 0.5
    very_danger_number = int(np.sum(dt < very_danger_threshold))
    plt.xlabel("Time of coalescence difference (s)")
    plt.ylabel("Density")
    plt.title(f"Number dt below {danger_threshold:.2f} s = {danger_number}, below {very_danger_threshold:.2f} s = {very_danger_number}")
    plt.savefig(os.path.join(figure_dir, "dt.pdf"))
    plt.close()
    
    
def main():
    make_plots()
    
if __name__ == "__main__":
    main()