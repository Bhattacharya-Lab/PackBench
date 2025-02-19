import os
from scipy.stats import wilcoxon
import json

DATA_ROOT_DIR = '' # location of folder downloaded from Zenodo
METRICS_ROOT_DIR = f'{DATA_ROOT_DIR}/metrics'
casps = ['casp14', 'casp15']
bb_types = ['af2', 'af3']
tools = [
    # 'af2', # only for sequence-based prediction
    # 'af3', # only for sequence-based prediction
    'flowpacker_bc40_noconf',
    'flowpacker_bc40_conf',
    'flowpacker_cluster_noconf',
    'flowpacker_cluster_conf',
    'pippack',
    'pippack_rs',
    'pippack_ensembled',
    'pippack_ensembled_rs',
    'diffpack',
    'diffpack_confidence',
    'attnpacker',
    'attnpacker_postproc',
    'dlpacker_sequence',
    'dlpacker_natoms',
    'dlpacker_score',
    'faspr',
    'pyrosetta_packer',
    'scwrl4',
]
metrics = [
    "all_rmsd",
    "core_rmsd",
    "surface_rmsd",
    "all_mean_mae_1",
    "all_mean_mae_2",
    "all_mean_mae_3",
    "all_mean_mae_4",
    "all_mae_sr",
    "all_clash_info_100_num_clashes",
    "all_clash_info_90_num_clashes",
    "all_clash_info_80_num_clashes"
]
num_targets = {
    "casp14": 66,
    "casp15": 71, # Was 72 before removing T1169.pdb
}

def traverse(src, dest, key_stack):
    if key_stack in metrics:
        dest[key_stack].append(src)
        return

    if isinstance(src, dict):
        for key in src:
            new_key_stack = key_stack if not key_stack else key_stack + "_"
            new_key_stack += key
            traverse(src[key], dest, new_key_stack)
    
    if isinstance(src, list):
        for i in range(len(src)):
            new_key_stack = key_stack if not key_stack else key_stack + "_"
            new_key_stack += str(i + 1)
            traverse(src[i], dest, new_key_stack)

def run_wilcoxon(before_metrics_dir, before_id, after_metrics_dir, after_id, output_filename):
    # Retrives the .json files
    before_files, after_files = {}, {}
    for filename in os.listdir(before_metrics_dir):
        if filename.startswith(before_id + "_T") and filename.endswith("json"):
            target = os.path.splitext(filename)[0][len(before_id)+1:]
            path = os.path.join(before_metrics_dir, filename)
            with open(path, "r") as file:
                loaded_dict = json.load(file)
            before_files[target] = loaded_dict
    for filename in os.listdir(after_metrics_dir):
        if filename.startswith(after_id + "_T") and filename.endswith("json"):
            target = os.path.splitext(filename)[0][len(after_id)+1:]
            path = os.path.join(after_metrics_dir, filename)
            with open(path, "r") as file:
                loaded_dict = json.load(file)
            after_files[target] = loaded_dict
    assert set(before_files.keys()) == set(after_files.keys())

    # Obtains the metrics in a simpler format
    before, after = {}, {}
    for metric in metrics:
        before[metric] = []
        after[metric] = []
    for target in before_files:
        traverse(before_files[target], before, "")
        traverse(after_files[target], after, "")

    # Gets the rounded diffs
    diffs = {}
    for metric in metrics:
        diffs[metric] = []
        assert len(before[metric]) == len(after[metric])
        for i in range(len(before[metric])):
            diff = after[metric][i] - before[metric][i]
            diff = int(diff * 1000) / 1000.0
            diffs[metric].append(diff)

    # Runs the Wilcoxon signed-rank test for each metric
    p_values = {}
    for metric in metrics:
        direction = "greater" if metric == "all_mae_sr" else "less"
        stat, p = wilcoxon(diffs[metric], alternative=direction)
        p_values[metric] = p

    # Saves the p-values
    with open(output_filename, "w") as f:
        json.dump(p_values, f, indent=4)
    print(f"Saved to {output_filename}.")

    return p_values

def generate_p_values():
    for casp in casps:
        for af in bb_types:
            print(f"Running tests for tools that repacked {af.upper()}'s predictions of {casp.upper()}")
            for tool in tools:
                nonnative_metrics_dir = os.path.join(METRICS_ROOT_DIR, casp, f"{casp}_{af}_predictions")
                nonnative_id = f"{casp}_repackingsequence_{af}"
                repacked_metrics_dir = os.path.join(METRICS_ROOT_DIR, casp,
                    f"repacking_{af}_bb", f"{tool}_predictions")
                repacked_id = f"{casp}_repacking{af}_{tool}"

                p_values_filename = os.path.join(repacked_metrics_dir, "p_values.json")
                run_wilcoxon(nonnative_metrics_dir, nonnative_id,
                             repacked_metrics_dir, repacked_id,
                             p_values_filename)
        #         break
        #     break
        # break

def main():
    generate_p_values()

if __name__ == "__main__":
    main()
