import os
import json
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT_DIR = '' # location of folder downloaded from Zenodo
METRICS_ROOT_DIR = f'{DATA_ROOT_DIR}/metrics'
casps = ['casp14', 'casp15']
bb_types = ['sequence', 'af2', 'af3']
combination_name = "combination"
tools_to_latex = {
    'af2': r'AlphaFold2',
    'af3': r'AlphaFold3',
    combination_name: "Weighted Combination",
    'flowpacker_cluster_conf': r'FlowPacker',
    'pippack_ensembled': r'PIPPack',
    'diffpack_confidence': r'DiffPack',
    'attnpacker': r'AttnPacker',
    'dlpacker_score': r'DLPacker',
    'faspr': r'FASPR',
    'pyrosetta_packer': r'PyRosetta Packer',
    'scwrl4': r'SCWRL4',
}
metrics = {
    ("all", "rmsd"): (r"$\Delta RMSD \downarrow$", (-0.1, 0.1)),
    # ("core", "rmsd"),
    # ("surface", "rmsd"),
    ("all", "mean_mae", 0): (r"$\Delta \chi_1\text{-}MAE \downarrow$", (-3, 3)),
    ("all", "mean_mae", 1): (r"$\Delta \chi_2\text{-}MAE \downarrow$", (-4, 4)),
    ("all", "mean_mae", 2): (r"$\Delta \chi_3\text{-}MAE \downarrow$", (-5, 5)),
    ("all", "mean_mae", 3): (r"$\Delta \chi_4\text{-}MAE \downarrow$", (-10, 10)),
    ("all", "mae_sr"): (r"$\Delta RR \uparrow$", (-4, 4)),
    ("all", "clash_info", "100", "num_clashes"): (r"$\Delta clashes_{100\%vdW} \downarrow$", (-50, 50)),
    # ("all", "clash_info", "90", "num_clashes"),
    # ("all", "clash_info", "80", "num_clashes"),
}

def get_dict():
    overall_dict = {}
    for casp in casps:
        overall_dict[casp] = {}
        for bb_type in bb_types:
            overall_dict[casp][bb_type] = {}
            for tool in tools_to_latex:
                if tool in ('af2', 'af3') and bb_type != 'sequence':
                    continue
                if bb_type == 'sequence' and tool not in ('af2', 'af3'):
                    continue
                if tool == combination_name and casp != "casp15":
                    continue
                # print(f"casp = {casp}, bb type = {bb_type}, tool = {tool}")

                metrics_dir = f'{METRICS_ROOT_DIR}/{casp}/{casp}_{tool}_predictions' \
                    if tool in ('af2', 'af3') \
                    else f'{METRICS_ROOT_DIR}/{casp}/repacking_{bb_type}_bb/{tool}_predictions'
                metrics_dir_id = f"{casp}_repacking{bb_type}_{tool}" \
                    if tool != combination_name \
                    else f"{casp}_repacking{bb_type}"
                stats_file_name = f"{metrics_dir_id}.json"
                stats_file_name = os.path.join(metrics_dir, stats_file_name)
                with open(stats_file_name, "r") as file:
                    overall_dict[casp][bb_type][tool] = json.load(file)

    return overall_dict

def get_nested_value(dictionary, path, index):
    if index == len(path):
        return dictionary
    else:
        return get_nested_value(dictionary[path[index]], path, index + 1)

def delta_plots(overall_dict):
    casp = "casp15"
    bb_type = "af3"

    # Creates the figure and axes
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 6), sharey=True)
    for ax, metric in zip(axes, metrics):
        deltas = []
        colors = []
        for tool in tools_to_latex:
            if tool in ("af2", "af3"):
                continue

            # Gets the metric values
            from_tool = get_nested_value(overall_dict[casp][bb_type][tool], metric, 0)
            original_alphafold = get_nested_value(overall_dict[casp]['sequence'][bb_type], metric, 0)
            delta = from_tool - original_alphafold
            if metric == ("all", "mae_sr"):
                delta *= 100
            deltas.append(delta)

            # Retrieves the p-value for the color
            p_values_file = os.path.join(
                METRICS_ROOT_DIR,
                casp,
                f"repacking_{bb_type}_bb",
                f"{tool}_predictions",
                "p_values.json" \
                    if tool != combination_name \
                    else f"{casp}_repacking{bb_type}_p_values.json"
            )
            with open(p_values_file, "r") as f:
                p_values = json.load(f)

            # Decides the color based on whether the change was statistically significant or not
            metric_name = ""
            for key in metric:
                if metric_name:
                    metric_name += "_"
                if isinstance(key, int):
                    metric_name += str(key+1)
                else:
                    metric_name += key
            p_value = p_values[metric_name]
            color = (0, 0, 1, 1) if p_value < 0.05 \
                else (0.5, 0.5, 0.9, 1)
            colors.append(color)            
        y_positions = range(len(deltas))

        # Scatter plot for the current metric
        min_x, max_x = metrics[metric][1]
        ax.set_xlim(min_x, max_x)
        ax.axvline(0, linestyle=':', color='gray', linewidth=1)
        for delta, y, color in zip(deltas, y_positions, colors):
            if delta > max_x:
                ax.annotate(
                    "▶", xy=(max_x, y), xytext=(max_x - 0.1 * max_x, y),
                    ha="right", va="center", fontsize=8, color=color
                )
            elif delta < min_x:
                ax.annotate(
                    "◀", xy=(min_x, y), xytext=(min_x + 0.1 * max_x, y),
                    ha="left", va="center", fontsize=8, color=color
                )
            else:
                ax.scatter(delta, y, c=[color], s=100, alpha=0.7)

        # Adds labels and gridlines
        ax.set_title(metrics[metric][0])
        ax.set_yticks(y_positions)
        ax.set_yticklabels(list(tools_to_latex.values())[2:])

    # Adjusts spacing between subplots
    plt.tight_layout()
    plot_file_name = "delta_plots.png"
    plt.savefig(plot_file_name, dpi=300)
    print(f"Saved to {plot_file_name} .")

    pass

if __name__ == "__main__":
    overall_dict = get_dict()
    delta_plots(overall_dict)
