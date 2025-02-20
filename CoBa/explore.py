import os
import copy
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
BNS_CATALOG_FILENAME = os.path.join(COBA_LOCATION, "CoBa_events_BNS.json")

# Load catalogs just to have a look:
with open(BBH_CATALOG_FILENAME, "r") as f:
    BBH_CATALOG = json.load(f)
    
with open(BNS_CATALOG_FILENAME, "r") as f:
    BNS_CATALOG = json.load(f)


def make_snr_histograms(snr_cutoff: float,
                        verbose: bool = False) -> None:
    """
    Make a plot showing the SNR distribution for ET, CE and the combined network SNR.

    Args:
        snr_cutoff (float): Only take events with an SNR below this value, to avoid outliers giving a large tail in the plot
        verbose (bool): Whether or not to plot a few extra things for debugging/checking.
    """
    # Start exploring the setup: load the SNR
    pop_str_list = ["BBH", "BNS"]
    catalogs_list = [BBH_CATALOG, BNS_CATALOG]

    for pop_str, catalog in zip(pop_str_list, catalogs_list):
        print(f"Making SNR histogram for {pop_str} catalog SNR below {snr_cutoff} . . . ")
        
        # # For individual ET ifos SNRs:
        # et1_snr = np.array(catalog["ET1_SNR"])
        
        if verbose:
            print(catalog["ET_SNR"][:10])
            print(catalog["CE_SNR"][:10])
        
        et_snr = np.array(catalog["ET_SNR"])
        ce_snr = np.array(catalog["CE_SNR"])
        network_snr = np.sqrt(et_snr ** 2 + ce_snr ** 2)
        
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
                       "density": True
                    }
        plt.hist(et_snr[mask], label="ET", color = "blue", **hist_kwargs)
        plt.hist(ce_snr[mask], label="CE", color = "red", **hist_kwargs)
        plt.hist(network_snr[mask], label="Network", color = "purple", **hist_kwargs)
        
        plt.legend()
        plt.title(f"{pop_str} (SNR below {snr_cutoff}, fraction {fraction_kept:.5f} kept)")
        
        plt.savefig(f"./figures/SNR_histogram_{pop_str}.png")
        plt.close()
        
        # Also make a histogram of the masses
        mass_1 = np.array(catalog["m1_source"])
        mass_2 = np.array(catalog["m2_source"])
        
        plt.figure()
        plt.hist(mass_1[mask], label="Mass 1", color = "blue", **hist_kwargs)
        plt.hist(mass_2[mask], label="Mass 2", color = "red", **hist_kwargs)
        plt.legend()
        # plt.title(f"{pop_str} (SNR below {snr_cutoff}, fraction {fraction_kept:.5f} kept)")
        plt.savefig(f"./figures/mass_histogram_{pop_str}.png")
        plt.close()
        
    print("DONE")
    
def investigate_mass_distributions():
    """
    Just a short investigation that shows that the mass distribution is sampled uniformly in the triangle m1>m2.
    Note: this is actually in the README of (ET-0084A-23) -- still good sanity check
    """
    
    catalog = BNS_CATALOG
    
    # Get the samples
    mass_1 = np.array(catalog["m1_source"])
    mass_2 = np.array(catalog["m2_source"])
    mass_samples = np.vstack([mass_1, mass_2]).T
    
    corner_kwargs = copy.deepcopy(default_corner_kwargs)
    hist_kwargs = {"density": True,
                   "color": "blue"
                }
    
    # Make the first cornerplot
    corner_kwargs["hist_kwargs"] = hist_kwargs
    fig = corner.corner(mass_samples, labels=["Mass 1", "Mass 2"], **corner_kwargs)
    
    # Sample the masses uniformly
    m1_tmp = np.random.uniform(1.1, 2.5, size=len(mass_1))
    m2_tmp = np.random.uniform(1.1, 2.5, size=len(mass_2))
    
    m1 = np.where(m1_tmp > m2_tmp, m1_tmp, m2_tmp)
    m2 = np.where(m1_tmp > m2_tmp, m2_tmp, m1_tmp)
    
    mass_samples = np.vstack([m1, m2]).T
    hist_kwargs["color"] = "red"
    corner_kwargs["hist_kwargs"] = hist_kwargs
    corner_kwargs["color"] = "red"
    
    corner.corner(mass_samples, labels=["Mass 1", "Mass 2"], fig = fig, **corner_kwargs)
    
    # Second cornerplot
    hist_kwargs
    corner_kwargs["hist_kwargs"] = hist_kwargs
    plt.savefig(f"./figures/mass_cornerplot_BNS.png")
    plt.close()
        
    
def main():
    # make_snr_histograms(snr_cutoff=100)
    investigate_mass_distributions()
    
if __name__ == "__main__":
    main()