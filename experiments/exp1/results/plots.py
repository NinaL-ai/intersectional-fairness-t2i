import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import seaborn as sns



def create_table(outfile):
    # diversity values table
    folder = "fairness"
    all_data = []

    for file in os.listdir(folder):
        if file.startswith("fair_df_") and file.endswith(".csv"):
            # Extract model name
            model = file.replace("fair_df_", "").replace(".csv", "")

            # Read CSV (index = metrics)
            df = pd.read_csv(os.path.join(folder, file), index_col=0)

            # Remove possible unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # Transpose: professions become rows
            df_t = df.T.reset_index().rename(columns={"index": "profession"})

            # Add model column
            df_t["model"] = model

            # Keep only required metrics
            df_t = df_t[["profession", "model", "Fcov", "Fcov_marginal", "max_dev", "jsd_dev"]]

            all_data.append(df_t)

    # Combine all models
    final_df = pd.concat(all_data, ignore_index=True)

    # Capitalize profession names
    final_df["profession"] = final_df["profession"].str.capitalize()

    # Pivot: columns = (model, metric)
    pivot_df = final_df.pivot(
        index="profession",
        columns="model",
        values=["Fcov_marginal", "Fcov", "max_dev", "jsd_dev"]
    )

    # Reorder to (model → metrics)
    pivot_df = pivot_df.swaplevel(0, 1, axis=1)
    pivot_df = pivot_df.sort_index(axis=1, level=0)

    # Enforce metric order
    metric_order = ["Fcov_marginal", "Fcov", "max_dev", "jsd_dev"]
    models = pivot_df.columns.levels[0]

    pivot_df = pivot_df.reindex(
        columns=pd.MultiIndex.from_product([models, metric_order])
    )

    # Create LaTeX table
    latex_table = pivot_df.to_latex(
        float_format="%.3f",
        caption="Intersectional diversity across models and professions. The highest diversity score for each specialty is highlighted in bold.",
        label="tab:representation",
        multicolumn=True,
        multirow=True,
        bold_rows=True,
        column_format="l" + "ccc" * len(pivot_df.columns.levels[0])
    )

    # Save to file
    with open(outfile, "w") as f:
        f.write(latex_table)

    return pivot_df

def plot_heatmap(df):
    # ----------------------------
    # 1. Extract metrics
    # ----------------------------
    model_order = ["Stablexl1.0", "Stable3.5-medium", "Flux-dev"]
    fcov = df.xs('Fcov', level=1, axis=1)[model_order]
    max_dev = df.xs('max_dev', level=1, axis=1)[model_order]
    jsd_dev = df.xs('jsd_dev', level=1, axis=1)[model_order]
    fcov_marg = df.xs('Fcov_marginal', level=1, axis=1)[model_order]

    # ----------------------------
    # 2. Consistent ordering
    # ----------------------------
    order = fcov.mean(axis=1).sort_values(ascending=False).index

    fcov = fcov.loc[order]
    fcov = fcov.rename(columns={"Stable3.5-medium": "SD3.5-M", "Stablexl1.0": "SDXL", "Flux-dev": "FLUX1"})
    fcov = fcov.rename(index={"Ceo": "CEO"})

    max_dev = max_dev.loc[order]
    max_dev = max_dev.rename(columns={"Stable3.5-medium": "SD3.5-M", "Stablexl1.0": "SDXL", "Flux-dev": "FLUX1"})
    max_dev = max_dev.rename(index={"Ceo": "CEO"})

    jsd_dev = jsd_dev.loc[order]
    jsd_dev = jsd_dev.rename(columns={"Stable3.5-medium": "SD3.5-M", "Stablexl1.0": "SDXL", "Flux-dev": "FLUX1"})
    jsd_dev = jsd_dev.rename(index={"Ceo": "CEO"})

    fcov_marg = fcov_marg.loc[order]
    fcov_marg = fcov_marg.rename(columns={"Stable3.5-medium": "SD3.5-M", "Stablexl1.0": "SDXL", "Flux-dev": "FLUX1"})
    fcov_marg = fcov_marg.rename(index={"Ceo": "CEO"})

    # ----------------------------
    # 3. Heatmap function
    # ----------------------------
    def plot(data, filename, show_yticks=True, show_cbar=True):
        plt.figure(figsize=(6, 8))

        ax = sns.heatmap(
            data,
            cmap="Reds",
            vmin=0,
            vmax=1,
            linewidths=0.3,
            linecolor="lightgray",
            square=True,
            cbar=show_cbar,
            cbar_kws={"shrink": 0.7, "pad": 0.03, "aspect": 30} if show_cbar else None
        )

        ax.set_xlabel("")
        ax.set_ylabel("")  # remove "Profession"

        # y-axis labels only for fcov
        if show_yticks:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
        else:
            ax.set_yticks([])

        # ----------------------------
        # Colorbar customization
        # ----------------------------
        if show_cbar:
            cbar = ax.collections[0].colorbar

            # ticks every 0.1
            ticks = np.arange(0, 1.01, 0.1)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
            cbar.ax.tick_params(labelsize=12)

            # add border around colorbar
            cbar.outline.set_visible(True)
            cbar.outline.set_linewidth(1.0)

        plt.xticks(rotation=70, ha="center", fontsize=13)
        plt.tight_layout()

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


    # ----------------------------
    # 4. Save plots
    # ----------------------------

    # Fcov marginal: keep profession labels, no colorbar
    plot(
        fcov_marg,
        "plots/heatmap_fcov_marg.pdf",
        show_yticks=True,
        show_cbar=False
    )

    # Fcov: keep profession labels, no colorbar
    plot(
        fcov,
        "plots/heatmap_fcov.pdf",
        show_yticks=False,
        show_cbar=False
    )

    # max_dev: no profession labels, no colorbar
    plot(
        max_dev,
        "plots/heatmap_max_dev.pdf",
        show_yticks=False,
        show_cbar=False
    )

    # jsd_dev: no profession labels, WITH colorbar
    plot(
        jsd_dev,
        "plots/heatmap_jsd_dev.pdf",
        show_yticks=False,
        show_cbar=True
    )



def plot_bar(model, profession):
    # final stacked bar plot code
    base_dir = "analysis_percentages"
    groups = ["gender", "age", "race"]
    age_order = ["young", "middle aged", "elderly"]

    bar_width = 0.35
    gap = 0.05

    group_colors = {
        "gender": ["#FFFFFF", "#000000"],  # ["#084582", "#4fa3f7"],
        "age": ["#c6dbef", "#6baed6", "#08519c"],  #["#6e6d6d", "#bab8b8", "#000000"],
        # "race": ["#FFCCCC", "#FF9999", "#FF6666", "#FF3333", "#CC0000", "#990000"]
        "race": ["#fee5d9", "#FF9999", "#FF3333", "#990000"] #"#FF3333"
    }

    FONT_SIZE = 20
    AX_HEIGHT = 5
    WIDTH_PER_BAR = 1.2

    model_path = os.path.join(base_dir, model, profession)

    plot_data = {}
    group_lookup = {}

    # --- Load data ---
    for file in os.listdir(model_path):
        if file.endswith("_percentages.csv") and "neutral" not in file:
            label = file.replace("_percentages.csv", "")
            df = pd.read_csv(os.path.join(model_path, file))

            if label in ["male", "female"]:
                group = "gender"
            elif label in ["young", "middle aged", "elderly"]:
                group = "age"
            else:
                group = "race"

            if group == "race":
                df = df[df["category"] != "neutral"]
                if df.empty:
                    continue

            df["category"] = df["category"].str.capitalize()
            plot_data[label] = df
            group_lookup[label] = group

    # --- Create subplots ---
    # Compute number of bars per group
    group_sizes = []
    for group in groups:
        labels = [l for l in plot_data if group_lookup[l] == group]
        group_sizes.append(len(labels))

    fig = plt.figure(figsize=(sum(group_sizes) * 1.5, AX_HEIGHT))
    gs = fig.add_gridspec(1, 3, width_ratios=group_sizes)

    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    for ax, group in zip(axes, groups):

        #labels = sorted([l for l in plot_data if group_lookup[l] == group])
        labels = [
            l for l in age_order
            if l in plot_data and group_lookup[l] == "age"
        ] if group == "age" else sorted(
            [l for l in plot_data if group_lookup[l] == group]
        )
        x_labels = [lbl.capitalize() for lbl in labels]
        x = np.arange(len(x_labels)) * 1.1

        for i, label in enumerate(labels):
            # print(label)
            df = plot_data[label]
            other_groups = [g for g in groups if g != group]
            offsets = [-bar_width/2 - gap/2, bar_width/2 + gap/2]

            for idx, g in enumerate(other_groups):
                bottoms = 0
                categories = sorted(df[df["group"] == g]["category"].unique())
                palette = group_colors[g]

                for j, category in enumerate(categories):
                    subset = df[
                        (df["group"] == g) &
                        (df["category"].str.lower() == category.lower())
                        ]
                    value = subset["percentage"].values[0] if not subset.empty else 0

                    ax.bar(
                        x[i] + offsets[idx],
                        value,
                        bar_width,
                        bottom=bottoms,
                        color=palette[j % len(palette)],
                        edgecolor="black",
                        linewidth=1.5
                        )
                    bottoms += value

        # --- Grid ---
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)

        # --- Same y-axis for all ---
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 25))

        # --- Axis labels ---
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=FONT_SIZE)

        if group == "gender":
            ax.set_ylabel("Proportion [%]", fontsize=FONT_SIZE)

        #ax.set_title(group.capitalize(), fontsize=24)
        ax.tick_params(axis='y', labelsize=FONT_SIZE)

    # --- Create GLOBAL legend ---
    legend_elements = []

    # Gender
    for i, label in enumerate(["Male", "Female"]):
        x = ["#000000", "#FFFFFF"]
        legend_elements.append(Patch(
            facecolor=x[i],
            label=label,
            edgecolor="black"
        ))

    # Age
    for i, label in enumerate(["Young", "Middle aged", "Elderly"]):
        c = ["#08519c", "#c6dbef", "#6baed6"]
        legend_elements.append(Patch(
            facecolor=c[i],
            label=label,
            edgecolor="black"
        ))

    # Race
    race_categories = set()
    for df in plot_data.values():
        if "race" in df["group"].values:
            race_categories.update(df[df["group"] == "race"]["category"].unique())

    race_categories = sorted(race_categories)
    race_categories.remove("Indian")
    print(race_categories)

    for i, label in enumerate(race_categories):
        # if label != "Indian":
        x = ["#fee5d9", "#FF9999", "#990000"]
        legend_elements.append(Patch(
            facecolor=x[i],
            label=label,
            edgecolor="black",
            linewidth=1
        ))

    # --- Place legend BELOW ---
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        fontsize=FONT_SIZE
    )

    # --- Layout ---
    plt.subplots_adjust(wspace=0.3, bottom=0.4)

    # --- Save ---
    os.makedirs("plots/percentage_bar_plot", exist_ok=True)
    plt.savefig("plots/percentage_bar_plot/" + model + "_" + profession + "_combined.pdf", bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    # df = create_table("diversity_table.tex")
    # plot_heatmap(df)
    plot_bar("Flux-dev", "teacher")
