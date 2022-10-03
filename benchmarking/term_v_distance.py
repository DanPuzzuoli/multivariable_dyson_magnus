#%%
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
connection = sqlite3.connect("data/gpu_data.sqlite")
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

df_plot = df_plot[(df_plot['num_inputs'] == 100) | (df_plot['solver'].isin(['Magnus', 'Dyson']))]


def new_labeler(row):
    if row['solver'] == 'ODE Solver':
        return "ODE Solver"
    else:
        return f"{row['solver']} ({row['cheb_order']}, {row['exp_order']})"

df_plot['label'] = df_plot.apply(new_labeler, axis=1)

df_plot.dropna(inplace=True)

grid = sns.scatterplot(
    y="total_run_time",
    x="ave_distance",
    ax=ax,
    data=df_plot,
    palette=sns.color_palette(color_palette, n_colors=df_plot['label'].nunique()),
    hue="label",
    style="label",
    # markers=["o", "o", "o", "o", "X", "X", "X", "X", "s"],
)

ax.set(xscale="log", yscale="log")
# ax.set_title("Distance vs run time for 100 inputs on gpu")
ax.set_title(None)
ax.set_xlabel("Average Distance")
ax.set_ylabel("Total Run Time (s)")

plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.savefig(
#     os.path.join(plot_folder, "distance_v_time.png"),
#     facecolor="white",
#     bbox_inches="tight",
# )

plt.show()

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
#%%
# Plot average distance vs total run time for 100 inputs, for each solver
n_colors = df_plot.solver.nunique()
df_plot = df_plot.sort_values(by="solver")

df_plot = df_plot[(df_plot['num_inputs'] == 100) | (df_plot['solver'].isin(['Magnus', 'Dyson']))]
df_plot['label'] = df_plot.apply(new_labeler, axis=1)

df_plot.dropna(inplace=True)
#%%
fig, ax = plt.subplots(figsize=(4, 4))

dfD = df_plot[df_plot['solver'] == 'Dyson']
dfM = df_plot[df_plot['solver'] == 'Magnus']
dfO = df_plot[df_plot['solver'] == 'ODE Solver']

dfD1 = dfD[(dfD['cheb_order'] == 2) & (dfD['exp_order'] == 5)]
dfD1['label'] = f'Dyson with term count {dfD1.iloc[0]["num_terms"]}'
dfD2 = dfD[(dfD['cheb_order'] == 0) & (dfD['exp_order'] == 4)]
dfD2['label'] = f'Dyson with term count {dfD2.iloc[0]["num_terms"]}'
dfD3 = dfD[(dfD['cheb_order'] == 0) & (dfD['exp_order'] == 2)]
dfD3['label'] = f'Dyson with term count {dfD3.iloc[0]["num_terms"]}'
dfD4 = dfD[(dfD['cheb_order'] == 0) & (dfD['exp_order'] == 3)]
dfD4['label'] = f'Dyson with term count {dfD4.iloc[0]["num_terms"]}'
# dfD = pd.concat([dfD1, dfD2,dfD4])

dfM1 = dfM[(dfM['cheb_order'] == 1) & (dfM['exp_order'] == 3)]
dfM1['label'] = f'Magnus with term count {dfM1.iloc[0]["num_terms"]}'
dfM2 = dfM[(dfM['cheb_order'] == 0) & (dfM['exp_order'] == 2)]
dfM2['label'] = f'Magnus with term count {dfM2.iloc[0]["num_terms"]}'
dfM3 = dfM[(dfM['cheb_order'] == 2) & (dfM['exp_order'] == 4)]
dfM3['label'] = f'Magnus with term count {dfM3.iloc[0]["num_terms"]}'
dfM4 = dfM[(dfM['cheb_order'] == 2) & (dfM['exp_order'] == 5)]
dfM4['label'] = f'Magnus with term count {dfM4.iloc[0]["num_terms"]}'


dfD = pd.concat([dfD1, dfD2, dfD3, dfD4])
dfD = dfD.sort_values('num_terms')
dfM = pd.concat([dfM1, dfM2, dfM3, dfM4])
dfM = dfM.sort_values('num_terms')
# dfM = pd.concat([dfM1, dfM2, dfM3] )

# df_plotD = dfD.groupby(pd.cut(np.log(dfD['ave_distance']), 30)).min()
# df_plotM = dfM.groupby(pd.cut(np.log(dfM['ave_distance']), 30)).min()


# df_plotD = dfD.loc[dfD['ave_distance']==dfD['ave_distance'].min()]
# df_plotM = dfM.loc[dfM['ave_distance']==dfM['ave_distance'].min()]
# df_plotO = dfO.loc[dfO['ave_distance']==dfO['ave_distance'].min()]

# df_plot = pd.concat([df_plotD, df_plotM, dfO])

def new_labeler(row):
    if row['solver'] == 'ODE Solver':
        return "ODE Solver"
    else:
        return f"{row['solver']} ({row['cheb_order']}, {row['exp_order']})"

dfO['label'] = 'ODE Solver'
df_plot1 = pd.concat([dfD, dfM, dfO])
df_plot1['label'] = df_plot1.apply(new_labeler, axis=1)

df_plot1.dropna(inplace=True)

grid = sns.scatterplot(
    y="total_grad_run_time",
    x="ave_distance",
    ax=ax,
    data=df_plot1,
    palette=sns.color_palette(color_palette, n_colors=df_plot1['label'].nunique()),
    hue="label",
    style="label",
    # markers=["o", "o", "o", "o", "X", "X", "X", "X", "s"],
)

ax.set(xscale="log", yscale="log")
# ax.set_title("Distance vs grad run time for 100 inputs on gpu")
ax.set_title(None)
ax.set_xlabel("Average Distance")
ax.set_ylabel("Total Grad Run Time (s)")

plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.savefig(
#     os.path.join(plot_folder, "distance_v_grad_time.png"),
#     facecolor="white",
#     bbox_inches="tight",
# )

plt.show()

if PARTIAL_DATA:
    df_plot = df_plot1

#%%
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
ax1 = axs[0]
ax2 = axs[1]


grid1 = sns.scatterplot(
    y="total_run_time",
    x="ave_distance",
    ax=ax1,
    data=df_plot[df_plot['solver'].isin(['Dyson', 'ODE Solver'])],
    # palette=sns.color_palette(color_palette, n_colors=df_plot['label'].nunique()),
    hue="label",
    style="label",
    # markers=["o", "o", "o", "o", "X", "X", "X", "X", "s"],
    # legend=None
)

grid2 = sns.scatterplot(
    y="total_run_time",
    x="ave_distance",
    ax=ax2,
    data=df_plot[df_plot['solver'].isin(['Magnus', 'ODE Solver'])],
    # palette=sns.color_palette(color_palette, n_colors=df_plot['label'].nunique()),
    hue="label",
    style="label",
    # markers=["o", "o", "o", "o", "X", "X", "X", "X", "s"],
)
ax1.set(xscale="log", yscale="log")
# ax1.set_title("Run Time ")
ax1.set_title(None)
ax1.set_xlabel("Average Distance")
ax1.set_ylabel("Total Run Time (s)")
ax1.set_ylim(top=1e3, bottom=1e-1)
# ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper right")
# ax1.set_xlim(left=1e-11)
ax1.set_xlim(left=1e-11, right=1e-1)
ax1.yaxis.grid(True)

ax2.set(xscale="log", yscale="log")
# ax2.set_title("Mag run time")
ax2.set_title(None)
ax2.set_xlabel("Average Distance")
ax2.set_ylabel(None)
ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
ax2.set_ylim(top=1e3, bottom=1e-1)
ax2.set_xlim(left=1e-11, right=1e-1)
ax2.yaxis.grid(True)

# plt.savefig(
#     os.path.join(plot_folder, "distance_mag_dyson.png"),
#     facecolor="white",
#     bbox_inches="tight",
# )


# %%

# color_palette = "tab10"

fig, axs = plt.subplots(figsize=(10, 4))
ax1 = axs

data1=df_plot[df_plot['solver'].isin(['Dyson', 'ODE Solver'])]
data1 = data1.sort_values(by="label")
# color_palette = sns.color_palette("Paired", data1.label.nunique())
color_palette = sns.color_palette("tab20", data1['label'].nunique())

grid1 = sns.scatterplot(
    y="total_run_time",
    x="ave_distance",
    ax=ax1,
    data=data1,
    palette=color_palette,
    hue="label",
)

ax1.set(xscale="log", yscale="log")
# ax1.set_title("Run Time ")
ax1.set_title(None)
ax1.set_xlabel("Average Distance")
ax1.set_ylabel("Total Run Time (s)")
ax1.set_ylim(top=1e3, bottom=1e-1)
# ax1.legend(bbox_to_anchor=(1.20, 1), loc="upper right")
# ax1.set_xlim(left=1e-11)
ax1.set_xlim(left=1e-11, right=1e-1)
ax1.yaxis.grid(True)

# plt.savefig(
#     os.path.join(plot_folder, "distance_wide_dyson_run.png"),
#     facecolor="white",
#     bbox_inches="tight",
# )

plt.show()
# %%

# color_palette = "tab10"

fig, axs = plt.subplots(figsize=(10, 4))
ax1 = axs

data1=df_plot[df_plot['solver'].isin(['Magnus', 'ODE Solver'])]
data1 = data1.sort_values(by="label")
# color_palette = sns.color_palette("Paired", data1.label.nunique())
color_palette = sns.color_palette("tab20", data1['label'].nunique())

grid1 = sns.scatterplot(
    y="total_run_time",
    x="ave_distance",
    ax=ax1,
    data=data1,
    palette=color_palette,
    hue="label",
    # style="label",
    # markers=["o", "o", "o", "o", "X", "X", "X", "X", "s", "s", "s", "s", "v"],
    # legend=None
)

ax1.set(xscale="log", yscale="log")
# ax1.set_title("Run Time ")
ax1.set_title(None)
ax1.set_xlabel("Average Distance")
ax1.set_ylabel("Total Run Time (s)")
ax1.set_ylim(top=1e3, bottom=1e-1)
# ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper right")
# ax1.legend(bbox_to_anchor=(1.22, 1), loc="upper right")
# ax1.set_xlim(left=1e-11)
ax1.set_xlim(left=1e-11, right=1e-1)
ax1.yaxis.grid(True)

# plt.savefig(
#     os.path.join(plot_folder, "distance_wide_magnus_run.png"),
#     facecolor="white",
#     bbox_inches="tight",
# )
# %%

# color_palette = "tab10"

fig, axs = plt.subplots(figsize=(10, 4))
ax1 = axs

data1 = df_plot[df_plot['solver'].isin(['Dyson', 'ODE Solver'])]
data1 = data1.sort_values(by="label")
# color_palette = sns.color_palette("Paired", data1.label.nunique())
color_palette = sns.color_palette("tab20", data1['label'].nunique())

grid1 = sns.scatterplot(
    y="total_grad_run_time",
    x="ave_distance",
    ax=ax1,
    data=data1,
    palette=color_palette,
    hue="label",
    # style="label",
    # markers=["o", "o", "o", "o", "X", "X", "X", "X", "s", "s", "s", "s", "v"],
    # legend=None
)

ax1.set(xscale="log", yscale="log")
# ax1.set_title("Grad Run Time ")
ax1.set_title(None)
ax1.set_xlabel("Average Distance")
ax1.set_ylabel("Total Grad Run Time (s)")
ax1.set_ylim(top=1e3, bottom=1e-1)
ax1.legend(bbox_to_anchor=(1.20, 1), loc="upper right")
# ax1.set_xlim(left=1e-11)
ax1.set_xlim(left=1e-11, right=1e-1)
ax1.yaxis.grid(True)

# plt.savefig(
#     os.path.join(plot_folder, "distance_wide_dyson_grad.png"),
#     facecolor="white",
#     bbox_inches="tight",
# )
# %%

# color_palette = "tab10"

fig, axs = plt.subplots(figsize=(10, 4))
ax1 = axs

data1=df_plot[df_plot['solver'].isin(['Magnus', 'ODE Solver'])]
data1 = data1.sort_values(by="label")
# color_palette = sns.color_palette("Paired", data1.label.nunique())
color_palette = sns.color_palette("tab20", data1['label'].nunique())

grid1 = sns.scatterplot(
    y="total_grad_run_time",
    x="ave_distance",
    ax=ax1,
    data=data1,
    palette=color_palette,
    hue="label",
    # style="label",
    # markers=["o", "o", "o", "o", "X", "X", "X", "X", "s", "s", "s", "s", "v"],
    # legend=None
)

ax1.set(xscale="log", yscale="log")
# ax1.set_title("Grad Run Time")
ax1.set_title(None)
ax1.set_xlabel("Average Distance")
ax1.set_ylabel("Total Grad Run Time (s)")
ax1.set_ylim(top=1e3, bottom=1e-1)
ax1.legend(bbox_to_anchor=(1.22, 1), loc="upper right")
# ax1.set_xlim(left=1e-11)
ax1.set_xlim(left=1e-11, right=1e-1)
ax1.yaxis.grid(True)

# plt.savefig(
#     os.path.join(plot_folder, "distance_wide_magnus_grad.png"),
#     facecolor="white",
#     bbox_inches="tight",
# )
# %%
# Plot the term count of each perturbative solver configuration against the average distance, separated by step count
color_palette = "magma"

df_plot1 = df_plot.copy()
df_plot1 = df_plot[df_plot['step_count'].isin([10000,50000])]
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
df_plot1 = df_plot1[df_plot1['Step Count'] == 10000]

fig, ax = plt.subplots(figsize=(4, 4))

# ax.set(yscale='log')
n_colors = df_plot1["Solver"].nunique()
# ax.set_title("Distance vs number of terms for perturbative solvers")
ax.set_title(None)
grid = sns.lineplot(
    x="Number of Terms",
    y="Average Distance",
    ax=ax,
    data=df_plot1,
    palette=sns.color_palette(color_palette, n_colors=n_colors),
    # hue="Step Count",
    # style="Solver",
    hue="Solver",
    # style="Step Count",
    marker= 'o'
)
ax.set(xscale="log", yscale="log")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.savefig(
#     os.path.join(plot_folder, "terms_v_distance_all.png"),
#     facecolor="white",
#     bbox_inches="tight",
# )
# %%
# Plot the term count of each perturbative solver configuration against the average distance, separated by step count
color_palette = "magma"

df_plot1 = df_plot.copy()
df_plot1 = df_plot[df_plot['step_count'].isin([10000,50000])]
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
df_plot2 = df_plot1.copy()
df_plot1 = df_plot1[df_plot1['Step Count'] == 10000]
df_plot2 = df_plot2[df_plot2['Step Count'] == 50000]

# fig, ax = plt.subplots(figsize=(4, 4))
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ax1 = axs[0]
ax2 = axs[1]

# ax.set(yscale='log')
n_colors = df_plot1["Solver"].nunique()
# ax.set_title("Distance vs number of terms for perturbative solvers")
grid = sns.lineplot(
    x="Number of Terms",
    y="Average Distance",
    ax=ax1,
    data=df_plot1,
    palette=sns.color_palette(color_palette, n_colors=n_colors),
    # hue="Step Count",
    # style="Solver",
    hue="Solver",
    # style="Step Count",
    legend=None,
    marker= 'o'
)
ax1.set(xscale="log", yscale="log")
ax1.set_xlim(left=1e1, right=1e4)
ax1.set_ylim(bottom=1e-11, top=1e-1)
ax1.yaxis.grid(True)
grid = sns.lineplot(
    x="Number of Terms",
    y="Average Distance",
    ax=ax2,
    data=df_plot2,
    palette=sns.color_palette(color_palette, n_colors=n_colors),
    # hue="Step Count",
    # style="Solver",
    hue="Solver",
    # style="Step Count",
    marker= 'o'
)
ax2.set(xscale="log", yscale="log")
ax2.set_xlim(left=1e1, right=1e4)
ax2.set_ylim(bottom=1e-11, top=1e-1)
ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
ax2.set_ylabel(None)
ax2.yaxis.grid(True)

anchored_text = AnchoredText("A", loc=1)
ax1.add_artist(anchored_text)
anchored_text2 = AnchoredText("B", loc=1)
ax2.add_artist(anchored_text2)

plt.savefig(
    os.path.join(plot_folder, "terms_v_distance_all.png"),
    facecolor="white",
    bbox_inches="tight",
)