import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

# Determine paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DEP_DIR = os.path.join(SCRIPT_DIR, "experiments/exp1", "results")
MODELS = ["Flux-dev", "Stable3.5-medium", "Stablexl1.0"]
PROFESSIONS = [
    "doctor", "nurse", "surgeon", "psychologist",  # Healthcare
    "singer", "artist", "writer",  # Creative
    "police officer", "judge",  # Law / authority
    "scientist", "software engineer",  # Tech / science
    "teacher",  # Education
    "CEO", "journalist", "politician", "office worker",  # Business / white collar
    "chef", "waiter", "flight attendant", "receptionist", # Service / hospitality  #
    "mechanic", "cashier", "baker",  # Manual labor
    "professional athlete", "basketball player", "fitness instructor"  # Sports
]

def load_fairness_data():
    data = {}
    for m in MODELS:
        path = os.path.join(RESULTS_DEP_DIR, "fairness", f"fair_df_{m}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.rename(columns={df.columns[0]: "Metric"}, inplace=True)
            data[m] = df
    return data

def get_combined_fairness_df(model_filter, specialty_filter):
    rows = []
    for m in MODELS:
        path = os.path.join(RESULTS_DEP_DIR, "fairness", f"fair_df_{m}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df.rename(columns={df.columns[0]: "Metric"}, inplace=True)
        # Melt the dataframe so we have: Metric, Specialty, Score
        df_melt = df.melt(id_vars=["Metric"], var_name="Profession", value_name="Score")
        df_melt["Model"] = m
        rows.append(df_melt)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    # Reorder columns
    combined = combined[["Model", "Profession", "Metric", "Score"]]
    # combined = combined.replace({"Find": r"$F_{dep}$"})
    # print(combined)

    # Apply filters
    if model_filter != "All":
        combined = combined[combined["Model"] == model_filter]
    if specialty_filter != "All":
        combined = combined[combined["Profession"] == specialty_filter]

    if combined.empty:
        return pd.DataFrame()

    # Pivot for readability: index = [Model, Specialty], columns = Metric
    pivoted = combined.pivot_table(index=["Model", "Profession"], columns="Metric", values="Score").reset_index()

    # Round scores for readability
    for col in pivoted.columns:
        if col not in ["Model", "Profession"]:
            pivoted[col] = pivoted[col].astype(float).round(4)

    # Sort by model and specialty
    pivoted = pivoted.sort_values(["Model", "Profession"])

    pivoted = pivoted.sort_values(["Model", "Profession"])
    print(pivoted.columns)
    pivoted.columns.name = None

    # Rename the column
    pivoted = pivoted[["Model", "Profession", "Fcov_marginal", "Fcov", "max_dev", "jsd_dev", "Find"]]
    pivoted["Profession"] = pivoted["Profession"].str.capitalize()

    pivoted = pivoted.rename(columns={
        "Find": "Fdep",
        "Fcov": "Fcov",
        "Fcov_marginal": "Fcov marg",
        "jsd_dev": "ΔJSD",
        "max_dev": "Δmax"

    })
    return pivoted

def load_dependency_matrix(specialty, model):
    path = os.path.join(RESULTS_DEP_DIR, "js_matrix", f"dependency_matrix_{specialty}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Filter by model
    df_model = df[df["model"].str.lower().str.startswith(model.lower())]
    if df_model.empty:
        return None
    # Pivot or extract the matrix columns: gender, age, race
    attributes = ["gender", "age", "race"]
    df_model = df_model.set_index("source")[attributes].loc[attributes, attributes]
    return df_model.astype(float)

def plot_dependency_heatmap(specialty, model):
    matrix_df = load_dependency_matrix(specialty, model)
    if matrix_df is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, f"No data available for model {model} & specialty {specialty}", ha="center", va="center")
        plt.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(6, 5))
    # Plot heatmap, leaving diagonal elements (NaNs) blank
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar=True,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 12}
    )
    ax.set_title(f"Pairwise JS Dependence Score\n({model} - {specialty})", fontsize=12, pad=15)
    plt.tight_layout()
    return fig

def get_raw_dependency_df(specialty, model):
    matrix_df = load_dependency_matrix(specialty, model)
    if matrix_df is None:
        return pd.DataFrame()

    matrix_df = matrix_df.reset_index().rename(columns={"source": "Attribute"})
    numeric_cols = matrix_df.select_dtypes(include=["number"]).columns
    matrix_df[numeric_cols] = matrix_df[numeric_cols].round(4)

    return matrix_df

def load_percentages(model, specialty, group):
    filename = f"{group}_percentages.csv"
    path = os.path.join(RESULTS_DEP_DIR, "analysis_percentages", model, specialty, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def plot_demographics(model, specialty, group):
    df = load_percentages(model, specialty, group)
    if df is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No demographic data available for:\nModel: {model}, Profession: {specialty}, Group: {group}",
                ha="center", va="center")
        plt.tight_layout()
        return fig, pd.DataFrame()

    fig = plt.figure(figsize=(14, 8))

    # 1. Marginal Age (top left)
    ax_age = plt.subplot2grid((2, 3), (0, 0))
    df_age = df[df["group"] == "age"].sort_values("category")
    if not df_age.empty:
        sns.barplot(data=df_age, x="category", y="percentage", ax=ax_age, hue="category", legend=False, palette="Blues_d")
        ax_age.set_title("Age Distribution (%)", fontweight="bold")
        ax_age.set_ylim(0, 110)
        ax_age.set_ylabel("Percentage")
        ax_age.set_xlabel("")
        for p in ax_age.patches:
            height = p.get_height()
            if height > 0:
                ax_age.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 2),
                                ha='center', va='center', fontsize=9, color='black', xytext=(0, 2),
                                textcoords='offset points')

    # 2. Marginal Gender (top middle)
    ax_gen = plt.subplot2grid((2, 3), (0, 1))
    df_gen = df[df["group"] == "gender"].sort_values("category")
    if not df_gen.empty:
        sns.barplot(data=df_gen, x="category", y="percentage", ax=ax_gen, hue="category", legend=False,
                    palette="Purples_d", width=0.5)
        ax_gen.set_title("Gender Distribution (%)", fontweight="bold")
        ax_gen.set_ylim(0, 110)
        ax_gen.set_ylabel("")
        ax_gen.set_xlabel("")
        for p in ax_gen.patches:
            height = p.get_height()
            if height > 0:
                ax_gen.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 2),
                                ha='center', va='center', fontsize=9, color='black', xytext=(0, 2),
                                textcoords='offset points')

    # 3. Marginal Race (top right)
    ax_race = plt.subplot2grid((2, 3), (0, 2))
    df_race = df[df["group"] == "race"].sort_values("category")
    if not df_race.empty:
        sns.barplot(data=df_race, x="category", y="percentage", ax=ax_race, hue="category", legend=False, palette="Oranges_d")
        ax_race.set_title("Race Distribution (%)", fontweight="bold")
        ax_race.set_ylim(0, 110)
        ax_race.set_ylabel("")
        ax_race.set_xlabel("")
        ax_race.tick_params(axis='x', rotation=45)
        for p in ax_race.patches:
            height = p.get_height()
            if height > 0:
                ax_race.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 2),
                                 ha='center', va='center', fontsize=8, color='black', xytext=(0, 2),
                                 textcoords='offset points')

    # 4. Intersectional (bottom span)
    #
    df_inter = df[df["group"] == "intersection"].copy()
    # Filter out 0% categories and sort descending
    df_inter = df_inter[df_inter["percentage"] > 0].sort_values("percentage", ascending=False)
    df_inter_top = df_inter.head(12)
    if len(df_inter_top) < 5:
        colspan = 1
    elif len(df_inter_top) < 9:
        colspan = 2
    else:
        colspan = 3

    ax_inter = plt.subplot2grid((2, 3), (1, 0), colspan=colspan)

    if not df_inter.empty:
        # bar_width = min(0.8, 6 / len(df_inter_top))
        sns.barplot(data=df_inter_top, x="category", y="percentage", ax=ax_inter, hue="category", legend=False,
                    palette="viridis")  # width=bar_width
        ax_inter.set_ylim(0, 110)
        ax_inter.set_title("Intersectional Distribution - Top Groups with >0% (%)", fontweight="bold")
        ax_inter.set_ylabel("Percentage")
        ax_inter.set_xlabel("Intersection (Age_Gender_Race)")
        ax_inter.tick_params(axis='x', rotation=20)
        for p in ax_inter.patches:
            height = p.get_height()
            if height > 0:
                ax_inter.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 1),
                                  ha='center', va='center', fontsize=9, color='black', xytext=(0, 2),
                                  textcoords='offset points')
    else:
        ax_inter.text(0.5, 0.5, "No intersectional groups detected with >0%", ha="center", va="center")

    plt.tight_layout()

    # Format a nice df for rendering
    df_display = df[df["percentage"] > 0].copy().sort_values(["group", "percentage"], ascending=[True, False])
    df_display = df_display.rename(columns={"group": "Group Type", "category": "Category Label", "percentage": "Percentage (%)"})
    df_display["Percentage (%)"] = df_display["Percentage (%)"].round(4)

    return fig, df_display


# Build the Gradio App Layout
with gr.Blocks(
        title="Intersectional Fairness in Text-To-Image Models",
        css="""
        .math-symbol {
            display: inline-block;
            line-height: 1;
        }
    
        .math-symbol sup {
            font-size: 0.75em;
            vertical-align: 0.6em;
            margin-left: -0.1em;
        }
    
        .math-symbol sub {
            font-size: 0.75em;
            vertical-align: 0em;
            margin-left: -2.4em;
        }
        """
) as demo:
    gr.Markdown(
        """
        # 📊 Intersectional Fairness in Text-To-Image Models
        This dashboard allows you to explore demographic distributions, intersectional fairness metrics, 
        and pairwise dependency scores across different Generative AI Text-to-Image models.
        """
    )

    with gr.Tabs():
        with gr.Tab("⚖️ Model Fairness Scores"):
            gr.Markdown(
                """
                ### Intersectional Fairness Scores
                Below are the pre-calculated fairness metrics for each model.
                - **<span class='math-symbol'>F<sup>marg</sup><sub>cov</sub></span>**: Marginal group coverage (higher is better)
                - **F<sub>cov</sub>**: Intersectional group coverage (higher is better)
                - **Δ<sub>max</sub>**: Maximum deviation of group representation to the target (lower is better)
                - **Δ<sub>JSD</sub>**: Distributional deviation of group representation to the target (lower is better)
                - **F<sub>dep</sub>**: Maximum pairwise dependency between attributes (lower is better)
                """
            )
            with gr.Row():
                model_sel = gr.Dropdown(choices=["All"] + MODELS, value="All", label="Filter by Model")
                specialty_sel = gr.Dropdown(choices=["All"] + PROFESSIONS, value="All", label="Filter by Profession")

            fairness_table = gr.DataFrame(interactive=False, show_search="search")
            # fairness_table = gr.HTML(interactive=False)

            # Hook filters up
            for component in [model_sel, specialty_sel]:
                component.change(
                    get_combined_fairness_df,
                    inputs=[model_sel, specialty_sel],
                    outputs=fairness_table,
                )

        with gr.Tab("🗺️ Pairwise Dependence"):
            gr.Markdown(
                """
                ### Pairwise JS Dependency Matrices
                This matrix shows the Jensen-Shannon (JS) divergence dependency score (Dep) between demographic attributes.
                A **lower** score indicates that the attributes (e.g., race, age, gender) are more independent 
                in the generated images (representing fairer, less correlated outcomes).
                """
            )
            with gr.Row():
                dep_model = gr.Dropdown(choices=MODELS, value="Flux-dev", label="Model")
                dep_specialty = gr.Dropdown(choices=PROFESSIONS, value="teacher", label="Profession")

            with gr.Row():
                with gr.Column(scale=4):
                    dep_plot = gr.Plot(label="Dep")
                with gr.Column(scale=5):
                    dep_data = gr.DataFrame(label="Raw Dependency Matrix", interactive=False)

            def update_dependency_tab(spec, mod):
                fig = plot_dependency_heatmap(spec, mod)
                df = get_raw_dependency_df(spec, mod)
                return fig, df

            dep_model.change(update_dependency_tab, inputs=[dep_specialty, dep_model], outputs=[dep_plot, dep_data])
            dep_specialty.change(update_dependency_tab, inputs=[dep_specialty, dep_model], outputs=[dep_plot, dep_data])

        with gr.Tab("📊 Demographic Distribution"):
            gr.Markdown(
                """
                ### Demographic Percentages
                Explore the distribution of generated faces along age, gender, race, and their intersections 
                based on the prompting group.
                """
            )
            with gr.Row():
                demo_model = gr.Dropdown(choices=MODELS, value="Flux-dev", label="Model")
                demo_specialty = gr.Dropdown(choices=PROFESSIONS, value="teacher", label="Profession")
                demo_group = gr.Dropdown(
                    choices=[
                        "neutral", "female", "male",
                        "young", "elderly", "middle aged",
                        "white", "black", "asian", "indian"
                    ],
                    value="neutral",
                    label="Prompt Group"
                )

            with gr.Row():
                demo_plot = gr.Plot(label=" ")

            with gr.Row():
                demo_table = gr.DataFrame(label="Category Percentages (>0%)", interactive=False)

            demo_model.change(plot_demographics, inputs=[demo_model, demo_specialty, demo_group],
                              outputs=[demo_plot, demo_table])
            demo_specialty.change(plot_demographics, inputs=[demo_model, demo_specialty, demo_group], outputs=[demo_plot, demo_table])
            demo_group.change(plot_demographics, inputs=[demo_model, demo_specialty, demo_group], outputs=[demo_plot, demo_table])

    # Initial loading functions
    demo.load(
        get_combined_fairness_df,
        inputs=[model_sel, specialty_sel],
        outputs=fairness_table,
    )
    demo.load(update_dependency_tab, inputs=[dep_specialty, dep_model], outputs=[dep_plot, dep_data])
    demo.load(plot_demographics, inputs=[demo_model, demo_specialty, demo_group], outputs=[demo_plot, demo_table])

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),

    )
