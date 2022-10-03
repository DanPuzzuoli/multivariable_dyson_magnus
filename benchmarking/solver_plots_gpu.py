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
plot_folder = "plot_folder"
connection = sqlite3.connect("data/gpu_data.sqlite")
df = pd.read_sql("SELECT * FROM benchmarks", con=connection)
#%%
# # DELETE THIS
# df = df[df["gpus"] == 1]
# connection = sqlite3.connect("gpu_data.sqlite")
# df.to_sql("benchmarks", con=connection, if_exists="replace")
# df.to_csv("gpu_data.csv")
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

df_plot["num_terms"] = df_plot.apply(
    lambda row: term_count(c=row["cheb_order"], n=row["exp_order"]), axis=1
)

# Plot average distance vs total run time for 100 inputs, for each solver
fig, ax = plt.subplots(figsize=(4, 4))
n_colors = df_plot.solver.nunique()
df_plot = df_plot.sort_values(by="solver")

df_plot = df_plot[
    (df_plot["num_inputs"] == 100) | (df_plot["solver"].isin(["Magnus", "Dyson"]))
]

dfD = df_plot[df_plot["solver"] == "Dyson"]
dfM = df_plot[df_plot["solver"] == "Magnus"]
dfO = df_plot[df_plot["solver"] == "ODE Solver"]

dfD1 = dfD[(dfD["cheb_order"] == 2) & (dfD["exp_order"] == 5)]
dfD1["label"] = f'Dyson with term count {dfD1.iloc[0]["num_terms"]}'
dfD2 = dfD[(dfD["cheb_order"] == 0) & (dfD["exp_order"] == 4)]
dfD2["label"] = f'Dyson with term count {dfD2.iloc[0]["num_terms"]}'
dfD3 = dfD[(dfD["cheb_order"] == 0) & (dfD["exp_order"] == 2)]
dfD3["label"] = f'Dyson with term count {dfD3.iloc[0]["num_terms"]}'
dfD4 = dfD[(dfD["cheb_order"] == 0) & (dfD["exp_order"] == 3)]
dfD4["label"] = f'Dyson with term count {dfD4.iloc[0]["num_terms"]}'
# dfD = pd.concat([dfD1, dfD2,dfD4])

dfM1 = dfM[(dfM["cheb_order"] == 1) & (dfM["exp_order"] == 3)]
dfM1["label"] = f'Magnus with term count {dfM1.iloc[0]["num_terms"]}'
dfM2 = dfM[(dfM["cheb_order"] == 0) & (dfM["exp_order"] == 2)]
dfM2["label"] = f'Magnus with term count {dfM2.iloc[0]["num_terms"]}'
dfM3 = dfM[(dfM["cheb_order"] == 2) & (dfM["exp_order"] == 4)]
dfM3["label"] = f'Magnus with term count {dfM3.iloc[0]["num_terms"]}'
dfM4 = dfM[(dfM["cheb_order"] == 2) & (dfM["exp_order"] == 5)]
dfM4["label"] = f'Magnus with term count {dfM4.iloc[0]["num_terms"]}'


dfD = pd.concat([dfD1, dfD2, dfD3, dfD4])
dfD = dfD.sort_values("num_terms")
dfM = pd.concat([dfM1, dfM2, dfM3, dfM4])
dfM = dfM.sort_values("num_terms")
# dfM = pd.concat([dfM1, dfM2, dfM3] )

# df_plotD = dfD.groupby(pd.cut(np.log(dfD['ave_distance']), 30)).min()
# df_plotM = dfM.groupby(pd.cut(np.log(dfM['ave_distance']), 30)).min()


# df_plotD = dfD.loc[dfD['ave_distance']==dfD['ave_distance'].min()]
# df_plotM = dfM.loc[dfM['ave_distance']==dfM['ave_distance'].min()]
# df_plotO = dfO.loc[dfO['ave_distance']==dfO['ave_distance'].min()]

# df_plot = pd.concat([df_plotD, df_plotM, dfO])


def new_labeler(row):
    if row["solver"] == "ODE Solver":
        return "ODE Solver"
    else:
        return f"{row['solver']} ({row['cheb_order']}, {row['exp_order']})"


dfO["label"] = "ODE Solver"
df_plot = pd.concat([dfD, dfM, dfO])
df_plot["label"] = df_plot.apply(new_labeler, axis=1)

df_plot.dropna(inplace=True)

grid = sns.scatterplot(
    y="total_run_time",
    x="ave_distance",
    ax=ax,
    data=df_plot,
    palette=sns.color_palette(color_palette, n_colors=df_plot["label"].nunique()),
    hue="label",
    style="label",
    markers=["o", "o", "o", "o", "X", "X", "X", "X", "s"],
)

ax.set(xscale="log", yscale="log")
ax.set_title("Distance vs run time for 100 inputs on gpu")
ax.set_xlabel("Average Distance")
ax.set_ylabel("Total Run Time (s)")

plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig(
    os.path.join(plot_folder, "distance_v_time.png"),
    facecolor="white",
    bbox_inches="tight",
)

plt.show()

#%%
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
    markers=["o", "o", "o", "o", "X", "X", "X", "X", "s",],
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
    ax=axs[1], x="Grad Perturbative Speedup", y="ave_distance", data=df1
)
grid = sns.scatterplot(x="Perturbative Speedup", y="ave_distance", data=df1, ax=axs[0])
axs[0].set(yscale="log")
axs[0].set_title("Perturbative solver speedup on GPU")
axs[0].set_ylabel("Average Distance")
axs[0].set_xlim(left=0)

axs[1].set(yscale="log")
axs[1].set_title("Perturbative solver gradient speedup on GPU")
axs[1].set_ylabel("Average Distance")
axs[1].set_xlim(left=0)

plt.savefig(
    os.path.join(plot_folder, "speedup_plots.png"),
    facecolor="white",
    bbox_inches="tight",
)

plt.show()
#%%
# Plot the term count of each perturbative solver configuration against the average distance, separated by step count
color_palette = "magma"

df_plot1 = df_plot.copy()
df_plot1 = df_plot[df_plot["step_count"].isin([10000, 50000])]
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

# ax.set(yscale='log')
n_colors = df_plot1["Step Count"].nunique()
ax.set_title("Distance vs number of terms for perturbative solvers")
grid = sns.lineplot(
    x="Number of Terms",
    y="Average Distance",
    ax=ax,
    data=df_plot1,
    palette=sns.color_palette(color_palette, n_colors=n_colors),
    hue="Step Count",
    style="Solver",
    markers=["o", "X",],
)
ax.set(xscale="log", yscale="log")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig(
    os.path.join(plot_folder, "terms_v_distance.png"),
    facecolor="white",
    bbox_inches="tight",
)
plt.show()
# %%

#%%
# calc the trendline
z = np.polyfit(df_plot1["Number of Terms"], df_plot1["Average Distance"], 3)
p = np.poly1d(z)
# pylab.plot(x,p(x),"r--")
# the line equation:
print("y=%.6fx+(%.6f)" % (z[0], z[1]))


# plt.show()

# %%
#%%
# Plot the term count of each perturbative solver configuration against the average distance, separated by step count
color_palette = "tab10"

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
        "total_run_time": "Total Run Time (s)",
    },
    inplace=True,
)
df_plot1 = df_plot1[df_plot1["Solver"] != "ODE Solver"]

fig, ax = plt.subplots(figsize=(4, 4))
ax.set(xscale="log", yscale="log")
n_colors = df_plot1["Number of Terms"].nunique()
ax.set_title("Distance vs Run Time for perturbative solvers")
grid = sns.scatterplot(
    y="Total Run Time (s)",
    x="Average Distance",
    ax=ax,
    data=df_plot1,
    palette=sns.color_palette(color_palette, n_colors=n_colors),
    hue="Number of Terms",
    style="Solver",
)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig(
    os.path.join(plot_folder, "terms_time_v_distance.png"),
    facecolor="white",
    bbox_inches="tight",
)

plt.show()
# %%