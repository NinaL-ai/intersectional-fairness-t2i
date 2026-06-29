import itertools
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from itertools import combinations
from scipy.stats import entropy
from itertools import product


class Metrics:
    def __init__(self, base_dir=None, attributes=None, target=None, out_path="results/fairness/",
                 jsd_path="results/dependency_matrix/"):
        self.base_dir = base_dir
        self.groups = {
            "gender": ["female", "male"],
            "age": ["young", "middle aged", "elderly"],
            "race": ["asian", "indian", "black", "white"]
        }
        self.n_groups = len(list(product(*self.groups.values())))
        self.attributes = ["age", "gender", "race"]
        self.target = target if target != None else 1 / self.n_groups
        self.jsd_path = jsd_path
        self.out_path = out_path


    def calc_pairwise_js(self, ):
        # find specialities
        specialties = []
        for model in os.listdir(self.base_dir):
            model_path = os.path.join(self.base_dir, model)
            for specialty in os.listdir(model_path):
                specialties.append(specialty)

        # main loop: compute pairwise dependencies
        for specialty in specialties:
            records = []
            for model in os.listdir(self.base_dir):
                model_path = os.path.join(self.base_dir, model, specialty)
                if not os.path.isdir(model_path):
                    continue

                # dependency matrix (pairwise JS)
                dep_matrix = pd.DataFrame(index=self.attributes, columns=self.attributes, dtype=float)
                for source_attr in self.attributes:
                    attribute_groups = self.groups[source_attr]

                    for target_attr in self.attributes:
                        if source_attr == target_attr:
                            dep_matrix.loc[source_attr, target_attr] = np.nan
                            continue

                        dists = []
                        for p in attribute_groups:
                            file_path = os.path.join(model_path, f"{p}_percentages.csv")
                            if not os.path.exists(file_path):
                                continue

                            df = pd.read_csv(file_path)
                            dist = self._extract_distribution(df, target_attr)
                            dists.append(dist)

                        if len(dists) > 1:
                            dep = self._compute_dependency(dists)
                        else:
                            dep = np.nan

                        dep_matrix.loc[source_attr, target_attr] = dep

                dep_matrix["model"] = model
                records.append(dep_matrix.reset_index().rename(columns={"index": "source"}))

                # save results
                df = pd.concat(records, ignore_index=True)
                os.makedirs(self.jsd_path, exist_ok=True)
                df.to_csv(self.jsd_path + f"{specialty}.csv", index=False)


    def _extract_distribution(self, df, target_attr):
        inter = df[df["group"] == "intersection"].copy()
        parts = inter["category"].str.split("_", expand=True)
        for i in range(len(self.attributes)):
            inter[self.attributes[i]] = parts[i]

        dist = inter.groupby(target_attr)["percentage"].sum()
        if dist.sum() > 0:
            dist = dist / dist.sum()
        return dist


    def _compute_dependency(self, dists):
        """
        Normalized JSD-based independence score in [0,1]
        """
        # align distributions
        all_keys = sorted(set().union(*[d.index for d in dists]))
        aligned = []
        for d in dists:
            vec = np.array([d.get(k, 0) for k in all_keys], dtype=float)
            if vec.sum() > 0:
                vec /= vec.sum()
            aligned.append(vec)
        aligned = np.array(aligned)

        # marginal reference distribution
        marginal = np.mean(aligned, axis=0)
        marginal = np.round(marginal, 15)

        # JS distances to marginal
        js_vals = [
            jensenshannon(p, marginal, base=2)**2
            for p in aligned
        ]
        max_js = max(js_vals)

        # maximum possible js distance for normalization
        k = len(all_keys)
        if k == 1:
            return 1.0

        uniform = np.ones(k) / k
        delta = np.zeros(k)
        delta[0] = 1.0
        JS_MAX = jensenshannon(delta, uniform, base=2)**2

        dep_norm = max_js / JS_MAX
        dep_norm = np.clip(dep_norm, 0.0, 1.0)

        return dep_norm

    def coverage(self, df_neutral):
        df_inter = df_neutral[df_neutral["group"] == "intersection"].copy()
        df_inter["proportion"] = df_inter["percentage"] / 100
        observed = df_inter[df_inter["percentage"] > 0].copy()
        return len(observed) / len(df_inter)

    def marginal_coverage(self, df_neutral):
        #F_cov^marginal for single axis coverage
        df_single = df_neutral[df_neutral["group"] != "intersection"].copy()
        observed_single = df_single[df_single["percentage"] > 0].copy()
        return len(observed_single) / len(df_single)

    def intersectional_dependency(self, pair_dependency, model):
        Find = pair_dependency[pair_dependency["model"] == model][self.attributes].max().max()
        return Find

    def max_deviation(self, df_neutral):
        df_inter = df_neutral[df_neutral["group"] == "intersection"].copy()
        df_inter["proportion"] = df_inter["percentage"] / 100
        print(df_inter["proportion"])
        return np.max(np.abs(df_inter["proportion"] - self.target))

    def jsd_deviation(self, df_neutral):
        df_inter = df_neutral[df_neutral["group"] == "intersection"].copy()
        df_inter["proportion"] = df_inter["percentage"] / 100
        if type(self.target) == float:
            target = [self.target for i in range(self.n_groups)]
        return jensenshannon(df_inter["proportion"], target, base=2)**2


    def calc_intersectional_fairness(self,):
        """
        Calculate Intersectional Fairness metrics for all models:
        - Fcov: coverage (fraction of observed intersectional groups)
        - Fbal: balance over observed groups using KL divergence
        - Fdiv: combined diversity (coverage * balance)
        - Find: pairwise independence (precomputed)
        """
        self.calc_pairwise_js()

        records = {}
        for model in os.listdir(self.base_dir):
            model_path = os.path.join(self.base_dir, model)
            if not os.path.isdir(model_path):
                continue

            print(model)

            for specialty in os.listdir(model_path):
                results = {}
                specialty_path = os.path.join(model_path, specialty)
                pairjs = pd.read_csv(self.jsd_path + f"{specialty}.csv")

                # Fairness metrics
                results["Find"] = self.intersectional_dependency(pairjs, model)
                neutral_file = os.path.join(specialty_path, "neutral_percentages.csv")
                df_neutral = pd.read_csv(neutral_file)
                results["Fcov"] = self.coverage(df_neutral)
                results["Fcov_marginal"] = self.marginal_coverage(df_neutral)
                results["max_dev"] = self.max_deviation(df_neutral)
                results["jsd_dev"] = self.jsd_deviation(df_neutral)
                records[specialty] = results

            # save results
            df = pd.DataFrame(records)
            os.makedirs(self.out_path, exist_ok=True)
            df.to_csv(self.out_path + f"fair_df_{model}.csv")


    def plot_pairwise_js(self, ):
        # load data
        attr_labels = ["Age", "Gender", "Race"]
        model_names = {
            "Stable3.5-medium": "SD3.5-M",
            "Stablexl1.0": "SDXL",
            "Flux-dev": "FLUX1"
        }

        # Fixed size per matrix
        matrix_width = 3
        matrix_height = 3

        # Font sizes
        fontsize = 22

        # Output folder
        os.makedirs(self.jsd_path + "plots/", exist_ok=True)

        for file in os.listdir(self.jsd_path):
            if file.endswith(".csv"):
                specialty = file.replace(".csv", "")

                df = pd.read_csv(os.path.join(self.jsd_path, file))
                df["source"] = pd.Categorical(df["source"], categories=self.attributes, ordered=True)

                models = sorted(df["model"].unique())
                n_models = len(models)

                # Figure height = 3 matrices stacked
                fig_height = n_models * matrix_height + 0.5
                fig_width = matrix_width + 0.5 + (0.5 if specialty.lower() == "surgeon" else 0)
                fig = plt.figure(figsize=(fig_width, fig_height))

                for idx, model in enumerate(models):
                    df_model = df[df["model"] == model]
                    mat_df = df_model.set_index("source")[self.attributes].loc[self.attributes, self.attributes]
                    mat = mat_df.values.astype(float)

                    # Compute axes position
                    bottom = 0.05 + (n_models - idx - 1) * (matrix_height / fig_height + 0.1)
                    width = matrix_width / fig_width
                    height = matrix_height / fig_height
                    left = 0.1  # fixed for all matrices

                    ax = fig.add_axes([left, bottom, width, height])

                    # Plot heatmap
                    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1)

                    # Ticks
                    ax.set_xticks(np.arange(len(self.attributes)))
                    ax.set_yticks(np.arange(len(self.attributes)))

                    # Y labels only for cardiologist
                    if specialty.lower() == "baker":
                        ax.set_yticklabels(attr_labels, fontsize=fontsize)
                    else:
                        ax.set_yticklabels([])

                    # ax.set_xticklabels(attr_labels, fontsize=fontsize, rotation=45, ha='right')
                    if model == "Stablexl1.0":
                        ax.set_xticklabels(
                            attr_labels,
                            fontsize=fontsize,
                            rotation=70,
                            ha="right"
                        )
                    else:
                        ax.set_xticklabels([])

                    # Annotate values
                    for i in range(len(self.attributes)):
                        for j in range(len(self.attributes)):
                            val = mat[i, j]
                            text_color = "white" if val > 0.5 else "black"

                            txt = f"{val:.2f}" if not np.isnan(val) else "-"
                            ax.text(j, i, txt, ha="center", va="center", fontsize=24, color=text_color)

                    # Black border
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.5)
                        spine.set_color("black")

                    # Add model header
                    # ax.set_title(model, fontsize=fontsize, pad=12)
                    ax.set_title(
                        model_names.get(model, model),
                        fontsize=fontsize,
                        pad=12
                    )

                    # Remove ticks outside matrix
                    ax.tick_params(top=False, bottom=False, left=False, right=False)

                    # Add colorbar manually only for last matrix if surgeon
                    if specialty.lower() == "professional athlete" and idx == n_models - 1:
                        cax = fig.add_axes([left + width + 0.05, bottom, 0.05, height])
                        cbar = fig.colorbar(im, cax=cax)
                        cbar.ax.tick_params(labelsize=fontsize)

                # Save PDF
                filename = self.jsd_path + f"plots/heatmap_{specialty}.pdf"
                plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
                plt.close()


