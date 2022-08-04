#%%
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

# connection = sqlite3.connect("cpu_data.sqlite")
# # df = pd.read_sql("SELECT * FROM benchmarks", con=connection)
# df.to_sql("benchmarks", con=connection, if_exists="replace")

plot_folder = "new_plots"
connection = sqlite3.connect("paper_gpu.sqlite")
df = pd.read_sql("SELECT * FROM benchmarks", con=connection)
plot_folder = "new_plots"
#%%
df = df[df["gpus"] == 1]
connection = sqlite3.connect("gpu_data.sqlite")
df.to_sql("benchmarks", con=connection, if_exists="replace")
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


#%%
color_palette = "tab10"

df_plot = df.copy()
df_plot["label"] = df_plot.apply(labeler, axis=1)
df_plot["total_run_time"] = df_plot["ave_run_time"].map(lambda x: x * 100)
df_plot["total_grad_run_time"] = df_plot["ave_grad_run_time"].map(lambda x: x * 100)
df_plot["solver"] = df_plot["solver"].map(
    lambda x: {"dense": "ODE Solver", "dyson": "Dyson", "magnus": "Magnus"}[x]
)
df_plot = df_plot.sort_values(by="label")
df_plot = df_plot[df_plot["gpus"] == 1]

# %%
# Plot average distance vs total gradient run time for 100 inputs, for each solver
fig, ax = plt.subplots(figsize=(4, 4))
n_colors = df_plot.solver.nunique()
df_plot = df_plot.sort_values(by="solver")
grid = sns.scatterplot(
    x="total_grad_run_time",
    y="ave_distance",
    ax=ax,
    data=df_plot,
    palette=sns.color_palette(color_palette, n_colors=n_colors),
    hue="solver",
    style="solver",
    markers=["o", "X", "s"],
)

ax.set(xscale="log", yscale="log")
ax.set_title("Distance vs grad run time for 100 inputs on gpu")
ax.set_ylabel("Average Distance")
ax.set_xlabel("Total Grad Run Time (s)")

plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig(
    os.path.join(plot_folder, "distance_v_grad_time.png"),
    facecolor="white",
    bbox_inches="tight",
)

plt.show()


#%%
# Plot average distance vs total run time for 100 inputs, for each solver, showing the number of inputs used for vmap
fig, ax = plt.subplots(figsize=(4, 4))
df_plot = df_plot.sort_values(by="label")
n_colors = df_plot.label.nunique()
grid = sns.scatterplot(
    x="total_run_time",
    y="ave_distance",
    ax=ax,
    data=df_plot,
    palette=sns.color_palette(color_palette, n_colors=n_colors),
    hue="label",
    style="label",
    markers=["o", "o", "o", "X", "X", "X", "s",],
)

ax.set(xscale="log", yscale="log")
ax.set_title("Distance vs run time for 100 inputs on gpu")
ax.set_ylabel("Average Distance")
ax.set_xlabel("Total Run Time (s)")

plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig(
    os.path.join(plot_folder, "distance_v_time_inputs.png"),
    facecolor="white",
    bbox_inches="tight",
)

plt.show()
#%%
# Find the fastest solver configuration from df2 that is at least as accurate as each solver configuration in df1
# In other words: for each row1 in dataframe1, find the row2 of subset(dataframe2, distance <= distance1) with the least total_run_time
def find_nearest_below(row1, df2, label="total_run_time"):
    df = df2[df2["ave_distance"] <= row1["ave_distance"]]
    if len(df) == 0:
        return 0
    speedup = row1[label] / min(df[label])
    idx = df[label].idxmin()
    solver_type = df.loc[idx, :]["solver"]

    return speedup, solver_type, min(df[label])


#%%
# Plot the speedup ratios of the perturbative solvers vs the ODE solver for normal and gradient run time
df1 = df_plot[df_plot["solver"] == "ODE Solver"]
df1 = df1.sort_values(by="total_run_time")
df1.drop_duplicates(subset="tol", keep="first", inplace=True)
df4 = df_plot[df_plot["solver"] != "ODE Solver"]
#%%
df1[["Perturbative Speedup", "Solver", "pert_speed"]] = df1.apply(
    find_nearest_below, df2=df4, axis=1, result_type="expand", label="total_run_time"
)
df1[["Grad Perturbative Speedup", "Solver", "pert_speed"]] = df1.apply(
    find_nearest_below,
    df2=df4,
    axis=1,
    result_type="expand",
    label="total_grad_run_time",
)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
grid2 = sns.scatterplot(
    ax=axs[1], y="Grad Perturbative Speedup", x="ave_distance", data=df1
)
grid = sns.scatterplot(y="Perturbative Speedup", x="ave_distance", data=df1, ax=axs[0])
axs[0].set(xscale="log")
axs[0].set_title("Perturbative solver speedup on GPU")
axs[0].set_xlabel("Average Distance")
axs[0].set_ylim(bottom=0)

axs[1].set(xscale="log")
axs[1].set_title("Perturbative solver gradient speedup on GPU")
axs[1].set_xlabel("Average Distance")
axs[1].set_ylim(bottom=0)

plt.savefig(
    os.path.join(plot_folder, "speedup_plots.png"),
    facecolor="white",
    bbox_inches="tight",
)

plt.show()
#%%
# Define some helper functions for calculating the 'term count` of each perturbative solver configuration
def choose(n, k):
    # n choose k:
    return factorial(n) / (factorial(k) * factorial(n - k))


def term_count(n, c):

    return choose(((4 * (c + 1)) + n), n) - 1


#%%
# Plot the term count of each perturbative solver configuration against the average distance, separated by step count
color_palette = "magma"

df_plot1 = df_plot.copy()
df_plot1["num_terms"] = df_plot.apply(
    lambda row: term_count(c=row["cheb_order"], n=row["exp_order"]), axis=1
)
df_plot1["step_count"] = df_plot1["step_count"].map(lambda x: int(x))
df_plot1.rename(
    columns={
        "step_count": "Step Count",
        "num_terms": "Number of Terms",
        "ave_distance": "Average Distance",
        "solver": "Solver",
    },
    inplace=True,
)
df_plot1 = df_plot1[df_plot1["Solver"] != "ODE Solver"]

fig, ax = plt.subplots(figsize=(4, 4))
ax.set(xscale="log", yscale="log")
n_colors = df_plot1["Step Count"].nunique()
ax.set_title("Distance vs number of terms for perturbative solvers")
grid = sns.scatterplot(
    x="Number of Terms",
    y="Average Distance",
    ax=ax,
    data=df_plot1,
    palette=sns.color_palette(color_palette, n_colors=n_colors),
    hue="Step Count",
    style="Solver",
)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig(
    os.path.join(plot_folder, "terms_v_distance.png"),
    facecolor="white",
    bbox_inches="tight",
)

plt.show()
