import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

METRICS_ROOT_DIR = "/home/common/proj/side_chain_packing/data/FINAL/metrics"

def gather_data():
    casp = "casp15"
    data = {casp: {}}
    alphafolds = {"af2": "AlphaFold2", "af3": "AlphaFold3"}
    tools = {
        "combination": "Weighted Combination",
        "flowpacker_cluster_conf": "FlowPacker",
        "pippack_ensembled": "PIPPack",
        "diffpack_confidence": "DiffPack",
        "attnpacker": "AttnPacker",
        "dlpacker_score": "DLPacker",
        "faspr": "FASPR",
        "pyrosetta_packer": "PyRosetta Packer",
        "scwrl4": "SCWRL4",
    }

    for af in alphafolds:
        data[casp][af] = {}

        # Retrieves AlphaFold metrics
        alphafold_name = "Original AlphaFold Side-Chains"
        data[casp][af][alphafold_name] = []
        metrics_dir = os.path.join(METRICS_ROOT_DIR, casp, f"{casp}_{af}_predictions")
        for target_metrics_file in sorted(os.listdir(metrics_dir)):
            prefix = f"{casp}_repackingsequence_{af}_T"
            if not target_metrics_file.startswith(prefix):
                continue
            file_path = os.path.join(metrics_dir, target_metrics_file)
            with open(file_path, "r") as f:
                target_metrics = json.load(f)
                recovery_rate = target_metrics["all"]["mae_sr"]
                data[casp][af][alphafold_name].append(recovery_rate)

        # Retrieves metrics for the side-chain packing tools
        # and our energy-based tool that combines them
        for tool in tools:
            data[casp][af][tools[tool]] = []
            metrics_dir = os.path.join(METRICS_ROOT_DIR, casp, f"repacking_{af}_bb", f"{tool}_predictions")
            for target_metrics_file in sorted(os.listdir(metrics_dir)):
                prefix = f"{casp}_repacking{af}_{tool}_T" if tool != "combination" \
                    else f"{casp}_repacking{af}_T"
                if not target_metrics_file.startswith(prefix):
                    continue
                file_path = os.path.join(metrics_dir, target_metrics_file)
                with open(file_path, "r") as f:
                    target_metrics = json.load(f)
                    recovery_rate = target_metrics["all"]["mae_sr"]
                    data[casp][af][tools[tool]].append(recovery_rate)

    # Saves the dictionary with aggregated results
    output_filename = f"{casp}_recovery_rates.json"
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {output_filename}.")

def make_boxplots():
    # Constructs dataframes (only for CASP15)
    casp = "casp15"
    afs = ["af2", "af3"]
    with open(f"{casp}_recovery_rates.json") as f:
        data = json.load(f)
    for af in afs:
        for tool_name in data[casp][af]:
            for i in range(len(data[casp][af][tool_name])):
                data[casp][af][tool_name][i] *= 100

    backbone_type_name = "Backbone Type"
    af2_df = pd.DataFrame(data=data[casp]["af2"]).assign(**{backbone_type_name: "AlphaFold2-Generated"})
    af3_df = pd.DataFrame(data=data[casp]["af3"]).assign(**{backbone_type_name: "AlphaFold3-Generated"})

    # Combines the dataframes
    rr_column_name = "Recovery Rate (%)"
    tool_word = "Tool"
    cdf = pd.concat([af2_df, af3_df])
    mdf = pd.melt(cdf, id_vars=[backbone_type_name], var_name=tool_word)
    mdf = mdf.rename(columns={"value": rr_column_name})

    # Makes the box plots
    meanlineprops = dict(
        # linestyle='--',
        linewidth=1.0,
        color='yellow'
    )
    ax = sns.boxplot(
        x=backbone_type_name,
        y=rr_column_name,
        hue=tool_word,
        data=mdf,
        showmeans=True,
        meanline=True,
        meanprops=meanlineprops
    )
    ax.set_ylim(0, 100)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title=tool_word)

    # Saves the boxplots
    filename = f"{casp}_rr_boxplots.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"Saved to {filename}.")

if __name__ == "__main__":
    gather_data()
    make_boxplots()
