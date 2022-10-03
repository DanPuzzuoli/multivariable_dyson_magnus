#%%
# Plot wide plots containing every Dyson and Magnus configuration on CPU
from matplotlib.offsetbox import AnchoredText
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import sqlite3
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import factorial

mpl.rcParams["figure.dpi"] = 300
#%%
PARTIAL_DATA=False
plot_folder = "plot_folder"
connection = sqlite3.connect("data/cpu_data.sqlite")
df = pd.read_sql("SELECT * FROM benchmarks", con=connection)
#%%
# Define some helper functions for calculating the 'term count` of each perturbative solver configuration
def choose(n, k):
    # n choose k:
    return factorial(n) / (factorial(k) * factorial(n - k))


def term_count(n, c):

    return int(choose(((4 * (c + 1)) + n), n) - 1)



# %%
# Label the rows of the dataframe with solver, gpu usage, and number of vmapped inputs
def labeler(row):
    label = ""
    solver = row["solver"]
    try:
        num_inputs = row["num_inputs"]
    except KeyError:
        num_inputs = 0

    if num_inputs == np.nan:
        num_inputs = 0

    if row["gpus"] == 1:
        cores = "gpu"
    else:
        cores = row["cpus"]

    if num_inputs != 0:
        label = f"{solver} {cores} with {num_inputs} inputs"
    else:
        label = f"{solver} {cores}"

    if not row["vmap"]:
        label = label + "_serial"
    return label


def new_labeler(row):
    if row['solver'] == 'ODE Solver':
        return "ODE Solver"
    else:
        return f"{row['solver']} ({row['cheb_order']}, {row['exp_order']})"

#%%
color_palette = "tab10"

df_plot = df.copy()

df_plot["label"] = df_plot.apply(labeler, axis=1)
df_plot["total_run_time"] = df_plot["ave_run_time"].map(lambda x: x * 100)
df_plot["total_grad_run_time"] = df_plot["ave_grad_run_time"].map(lambda x: x * 100)
df_plot["solver"] = df_plot["solver"].map(
    lambda x: {"dense": "ODE Solver", "dyson": "Dyson", "magnus": "Magnus"}[x]
)
df_plot = df_plot[df_plot["gpus"] == 0]

df_plot["num_terms"] = df_plot.apply(
    lambda row: term_count(c=row["cheb_order"], n=row["exp_order"]), axis=1
)
#%%
# Plot average distance vs total run time for 100 inputs, for each solver

df_plot = df_plot[(df_plot['num_inputs'] == 100) | (df_plot['solver'].isin(['Magnus', 'Dyson']))]
df_plot['label'] = df_plot.apply(new_labeler, axis=1)
df_plot = df_plot.sort_values(by="label")

df_plot.dropna(inplace=True)
#%%
def setup_plot(ylabel, ax, grad=False):
    ax.set(xscale="log", yscale="log")
    ax.set_title(None)
    ax.set_xlabel("Average Distance")
    ax.set_ylabel(ylabel)
    if grad:
        ax.set_ylim(top=1e5, bottom=1e1)
    else:
        ax.set_ylim(top=1e4, bottom=1e1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_xlim(left=1e-11, right=1e-1)
    ax.yaxis.grid(True)
#%%
# Plot Magnus and Dyson configurations run time stacked on top of each other
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1 = axs[0]
ax2 = axs[1]
data_dyson = df_plot[df_plot['solver'].isin(['Dyson', 'ODE Solver'])]
data_magnus = df_plot[df_plot['solver'].isin(['Magnus', 'ODE Solver'])]
palette=(sns.color_palette('tab20', n_colors=data_dyson.label.nunique()))
palette2=(sns.color_palette('tab20', n_colors=data_magnus.label.nunique()))

grid1 = sns.scatterplot(
    y="total_run_time",
    x="ave_distance",
    ax=ax1,
    data=data_dyson,
    palette = palette,
    hue="label",
)

grid2 = sns.scatterplot(
    y="total_run_time",
    x="ave_distance",
    ax=ax2,
    data=data_magnus,
    palette = palette2,
    hue="label",
)


setup_plot('Total Run Time (s)', ax1)
setup_plot('Total Run Time (s)', ax2)
anchored_text = AnchoredText("A", loc=1)
ax1.add_artist(anchored_text)
anchored_text2 = AnchoredText("B", loc=1)
ax2.add_artist(anchored_text2)

plt.savefig(
    os.path.join(plot_folder, "cpu_distance_wide_stacked.png"),
    facecolor="white",
    bbox_inches="tight",
)

# %%
# Plot Magnus and Dyson configurations grad run time stacked on top of each other
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax3 = axs[0]
ax4 = axs[1]

grid3 = sns.scatterplot(
    y="total_grad_run_time",
    x="ave_distance",
    ax=ax3,
    data=data_dyson,
    palette = palette,
    hue="label",
)
grid4 = sns.scatterplot(
    y="total_grad_run_time",
    x="ave_distance",
    ax=ax4,
    data=data_magnus,
    palette = palette2,
    hue="label",
)
setup_plot('Total Grad Run Time (s)', ax3, grad=True)
setup_plot('Total Grad Run Time (s)', ax4, grad=True)

anchored_text3 = AnchoredText("A", loc=1)
ax3.add_artist(anchored_text3)
anchored_text4 = AnchoredText("B", loc=1)
ax4.add_artist(anchored_text4)

plt.savefig(
    os.path.join(plot_folder, "cpu_distance_wide_stacked_grad.png"),
    facecolor="white",
    bbox_inches="tight",
)
