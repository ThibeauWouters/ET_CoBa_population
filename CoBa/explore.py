"""
Simple script to load the CoBa catalogs (with the SNRs computed) and have a look at some SNR distributions as sanity checks.
"""

import os
import copy
import arviz
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

# Print the number of events in each catalog
key = "ET_SNR"
print(f"Number of BBH events: {len(BBH_CATALOG[key])}")
print(f"Number of BNS events: {len(BNS_CATALOG[key])}")

def plot_snr_comparison(which: str = "network",
                        max_snr: float = 200):
    """
    Make a plot showing 20 Hz SNR vs 5 Hz
    """
    
    # First just say out loud how many events are above SNR of 8
    detected = np.sqrt(np.array(BBH_CATALOG["ET_SNR"]) ** 2 + np.array(BBH_CATALOG["CE_SNR"]) ** 2) > 8
    print(f"Fraction of events with SNR > 8: {np.sum(detected) / len(detected)}")
    
    detected = np.sqrt(np.array(BBH_5HZ_CATALOG["ET_SNR"]) ** 2 + np.array(BBH_5HZ_CATALOG["CE_SNR"]) ** 2) > 8
    print(f"Fraction of events with SNR > 8: {np.sum(detected) / len(detected)}")
    
    if which.lower() not in ["et", "ce", "network"]:
        raise ValueError("which must be one of 'ET', 'CE' or 'network'")
        
    # 20 Hz:
    et_snr = np.array(BBH_CATALOG["ET_SNR"])
    ce_snr = np.array(BBH_CATALOG["CE_SNR"])
    network_snr = np.sqrt(et_snr ** 2 + ce_snr ** 2)
    
    if which.lower() == "et":
        x = np.array(et_snr)
    elif which.lower() == "ce":
        x = np.array(ce_snr)
    else:
        x = np.array(network_snr)
    
    # 5 Hz:
    et_snr = np.array(BBH_5HZ_CATALOG["ET_SNR"])
    ce_snr = np.array(BBH_5HZ_CATALOG["CE_SNR"])
    network_snr = np.sqrt(et_snr ** 2 + ce_snr ** 2)
    
    if which.lower() == "et":
        y = np.array(et_snr)
    elif which.lower() == "ce":
        y = np.array(ce_snr)
    else:
        y = np.array(network_snr)
        
    # NOTE: There seems to be one faulty datapoints, this always has to be done
    mask = y < 4_000
    x = x[mask]
    y = y[mask]
    
    mask = y < max_snr 
    x = x[mask]
    y = y[mask]

    plt.figure()
    plt.plot(x, y, "o", label="BBH", alpha = 0.25)
    # Draw a straight line for reference
    x_ = np.linspace(0.0, np.max(y), 1_000)
    plt.plot(x_, x_, color = "black", linestyle = "--", alpha = 0.75)
    plt.xlabel(f"{which} SNR at 20 Hz")
    plt.ylabel(f"{which} SNR at 5 Hz")
    plt.legend()
    
    name = "./figures/SNR_comparison.png"
    plt.savefig(name)
    plt.savefig(name.replace(".png", ".pdf"))
    plt.close()

def compare_flow_snr(snr_cutoff: float = 600,
                     detectable_snr: float = 13.85) -> None:
    """
    Make a plot comparings the SNR distribution for BBH SNRs for 20 Hz vs 5 Hz

    Args:
        snr_cutoff (float): Only take events with an SNR below this value, to avoid outliers giving a large tail in the plot
        verbose (bool): Whether or not to plot a few extra things for debugging/checking.
        detectable_snr (float): SNR above which the event is detectable by the network (default is 13.85: see arXiv:2102.07544v1)
    """
    # Start exploring the setup: load the SNR
    labels_list = ["20 Hz", "5 Hz"]
    catalogs_list = [BBH_CATALOG, BBH_5HZ_CATALOG]
    
    results_dict = {}

    for label, catalog in zip(labels_list, catalogs_list):
        print(f"Making SNR histogram for {label} catalog SNR below {snr_cutoff} . . . ")
        
        et_snr = np.array(catalog["ET_SNR"])
        ce_snr = np.array(catalog["CE_SNR"])
        network_snr = np.sqrt(et_snr ** 2 + ce_snr ** 2)
        
        if detectable_snr > 0:
            print(f"Only events with SNR above {detectable_snr} are detectable by the network.")
            detected = network_snr > detectable_snr
            
            et_snr = et_snr[detected]
            ce_snr = ce_snr[detected]
            network_snr = network_snr[detected]
        
        # Limit SNR to be below a given threshold
        mask = network_snr < snr_cutoff
        fraction_kept = np.sum(mask) / len(mask)
        print("fraction_kept")
        print(fraction_kept)
        
        et_snr = et_snr[mask]
        ce_snr = ce_snr[mask]
        network_snr = network_snr[mask]
        
        results_dict[label] = {"et_snr": et_snr,
                               "ce_snr": ce_snr,
                               "network_snr": network_snr}
        
    # Make a histogram of the SNRs
    plt.figure()
    hist_kwargs = {"bins": 100, 
                    "histtype": "step", 
                    "density": True,
                    "linewidth": 3
                }
    
    for label, col in zip(labels_list, ["blue", "red"]):
        # plt.hist(results_dict["et_snr"][mask], label="ET", color = "blue", **hist_kwargs)
        # plt.hist(results_dict["ce_snr"][mask], label="CE", color = "red", **hist_kwargs)
        
        network_snr = results_dict[label]["network_snr"]
        plt.hist(results_dict[label]["network_snr"], label=label, color = col, **hist_kwargs)
        
        # Print median SNR and 90% quantile
        med = np.median(network_snr)
        low, high = arviz.hdi(network_snr, hdi_prob=0.9)
        low = med - low
        high = high - med
    
        print(label)
        print(f"Median SNR: {med:.2f} with 90% CI: [{low:.2f}, {high:.2f}]")
    
    plt.legend()
    
    name = f"./figures/SNR_histogram_comparison_flow.png"
    
    plt.xlabel("SNR")
    plt.ylabel("density")
    
    plt.savefig(name)
    plt.savefig(name.replace(".png", ".pdf"))
    plt.close()
        
    print("DONE")
    
def make_snr_histograms(snr_cutoff_dict: dict,
                        verbose: bool = False,
                        detectable_snr: float = 13.85) -> None:
    """
    Make a plot showing the SNR distribution for ET, CE and the combined network SNR.

    Args:
        snr_cutoff (float): Only take events with an SNR below this value, to avoid outliers giving a large tail in the plot
        verbose (bool): Whether or not to plot a few extra things for debugging/checking.
        detectable_snr (float): SNR above which the event is detectable by the network (default is 13.85: see arXiv:2102.07544v1)
    """
    # Start exploring the setup: load the SNR
    pop_str_list = ["BBH", "BNS"]
    catalogs_list = [BBH_CATALOG, BNS_CATALOG]

    for pop_str, catalog in zip(pop_str_list, catalogs_list):
        snr_cutoff = snr_cutoff_dict[pop_str]
        print(f"Making SNR histogram for {pop_str} catalog SNR below {snr_cutoff} . . . ")
        
        # # For individual ET ifos SNRs:
        # et1_snr = np.array(catalog["ET1_SNR"])
        
        if verbose:
            print(catalog["ET_SNR"][:10])
            print(catalog["CE_SNR"][:10])
        
        et_snr = np.array(catalog["ET_SNR"])
        ce_snr = np.array(catalog["CE_SNR"])
        network_snr = np.sqrt(et_snr ** 2 + ce_snr ** 2)
        
        if detectable_snr > 0:
            print(f"Only events with SNR above {detectable_snr} are detectable by the network.")
            detected = network_snr > detectable_snr
            
            et_snr = et_snr[detected]
            ce_snr = ce_snr[detected]
            network_snr = network_snr[detected]
        
        # # Show a couple of the highest SNR points
        if verbose:
            max_idx = 20
            
            # Sort them from high SNR to low SNR
            sort_idx_et = np.argsort(et_snr)[::-1]
            sort_idx_ce = np.argsort(ce_snr)[::-1]
            sort_idx_network = np.argsort(network_snr)[::-1]
            
            et_snr_sorted = et_snr[sort_idx_et]
            ce_snr_sorted = ce_snr[sort_idx_ce]
            network_snr_sorted = network_snr[sort_idx_network]
            print(f"Highest ET SNRs: {et_snr_sorted[:max_idx]}")
            print(f"Highest CE SNRs: {ce_snr_sorted[:max_idx]}")
            print(f"Highest network SNRs: {network_snr_sorted[:max_idx]}")
            
        # Limit SNR to be below a given threshold
        mask = network_snr < snr_cutoff
        fraction_kept = np.sum(mask) / len(mask)
        
        # Make a histogram of the SNRs
        plt.figure()
        hist_kwargs = {"bins": 100, 
                       "histtype": "step", 
                       "density": True,
                       "linewidth": 3
                    }
        plt.hist(et_snr[mask], label="ET", color = "blue", **hist_kwargs)
        plt.hist(ce_snr[mask], label="CE", color = "red", **hist_kwargs)
        plt.hist(network_snr[mask], label="Network", color = "purple", **hist_kwargs)
        
        # Print median SNR and 90% quantile
        med = np.median(network_snr[mask])
        low, high = arviz.hdi(network_snr[mask], hdi_prob=0.9)
        low = med - low
        high = high - med
        
        print(f"Median SNR: {med:.2f} with 90% CI: [{low:.2f}, {high:.2f}]")
        
        plt.legend()
        plt.title(f"{pop_str} (SNR below {snr_cutoff}, fraction {fraction_kept:.5f} kept)")
        
        name = f"./figures/SNR_histogram_{pop_str}.png"
        
        plt.xlabel("SNR")
        plt.ylabel("density")
        
        plt.savefig(name)
        plt.savefig(name.replace(".png", ".pdf"))
        plt.close()
        
    print("DONE")
    
def main():
    # snr_cutoff_dict = {"BBH": 600, "BNS": 150}
    # make_snr_histograms(snr_cutoff_dict)
    
    # compare_flow_snr()
    
    plot_snr_comparison()
    
if __name__ == "__main__":
    main()