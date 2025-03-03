import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

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
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

filename = "list_etmdc1_snr.txt"

# Load with pandas 
df = pd.read_csv(filename, sep=" ")

bbh = df[df["type"] == 3]
nsbh = df[df["type"] == 2]
bns = df[df["type"] == 1]

SNR_THRESHOLD = 12
bbh_cutoff = bbh[bbh["SNR"] > SNR_THRESHOLD]
nsbh_cutoff = nsbh[nsbh["SNR"] > SNR_THRESHOLD]
bns_cutoff = bns[bns["SNR"] > SNR_THRESHOLD]

# Print the total number to the screen
print("Total number of events: ", len(df))
print("Number of BBH events: ", len(bbh))
print("Number of NSBH events: ", len(nsbh))
print("Number of BNS events: ", len(bns))
print("\n")

print("bbh")
print(bbh)

t_c = bbh["t_c"]

# TODO: finish this up

# print("Number of BBH events above SNR threshold: ", len(bbh_cutoff))
# print("Number of NSBH events above SNR threshold: ", len(nsbh_cutoff))
# print("Number of BNS events above SNR threshold: ", len(bns_cutoff))

# plt.figure(figsize=(10, 8))