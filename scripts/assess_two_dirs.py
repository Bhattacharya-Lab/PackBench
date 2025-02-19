from protein_learning.assessment.sidechain import assess_sidechains, summarize, debug
import pprint
import os
import json
import torch
from tqdm import tqdm
import glob

# Calculates the metrics for a pair of (target, prediction).
def results_per_target(targets_dir, predictions_dir, stats_dir=None, output_per_residue_metrics=False):
    target_stats_list = []
    target_files = glob.glob(os.path.join(targets_dir, "*.pdb"))
    target_files = set([os.path.basename(filename) for filename in target_files])
    prediction_files = glob.glob(os.path.join(predictions_dir, "*.pdb"))
    prediction_files = set([os.path.basename(filename) for filename in prediction_files])
    print(f"target_files - prediction_files = {target_files - prediction_files}")
    print(f"prediction_files - target_files = {prediction_files - target_files}")
    assert set(target_files) == set(prediction_files)

    for target in tqdm(target_files):
        predicted_pdb = os.path.join(predictions_dir, target)
        target_pdb = os.path.join(targets_dir, target)
        res_level_stats = assess_sidechains(target_pdb_path=target_pdb, decoy_pdb_path=predicted_pdb, steric_tol_fracs = [1,0.9,0.8])
        target_level_stats = summarize(res_level_stats)
        target_stats_list.append(target_level_stats)

        if output_per_residue_metrics:
            assert stats_dir is not None, "Pass in a desired directory for per-residue metrics"
            converted = tensor_to_python(res_level_stats)
            os.makedirs(stats_dir, exist_ok=True)
            target_name = os.path.splitext(os.path.basename(target_pdb))[0]
            stats_file_name = f"{target_name}.json"
            stats_file_name = os.path.join(stats_dir, stats_file_name)
            with open(stats_file_name, "w") as file:
                json.dump(converted, file, indent=4)

    return target_stats_list

# For a given test dataset, backbone type, and tool, this function just aggregates the data
# from all of the targets together.
def aggregate_dataset_stats(target_stats_list):
    overall_stats = {}
    first_target_stats = target_stats_list[0]

    for centrality in ["all", "core", "surface"]:
        overall_stats[centrality] = dict()

        tensor_keys = ['rmsd', 'mae_sr', 'mean_mae']
        for key in tensor_keys:
            stacked = torch.stack([target[centrality][key] for target in target_stats_list]).float()
            overall_stats[centrality][key] = torch.mean(stacked, dim=0)
        
        overall_stats[centrality]['dihedral_counts'] = torch.sum(
            torch.stack([target[centrality]['dihedral_counts'] for target in target_stats_list]),
            dim=0
        )

        overall_stats[centrality]['num_sc'] = torch.sum(torch.tensor([target[centrality]['num_sc'] for target in target_stats_list]))
        overall_stats[centrality]['mean_seq_len'] = torch.mean(
            torch.tensor([target[centrality]['seq_len'] for target in target_stats_list]).float()
        )

    # These metrics are only given across all residues.
    overall_stats["all"]["ca_rmsd"] = torch.mean(
        torch.stack([target["all"]["ca_rmsd"] for target in target_stats_list]).float(),
        dim=0
    )
    overall_stats["all"]['clash_info'] = {
        threshold: {
            'energy': torch.mean(
                torch.stack([target["all"]['clash_info'][threshold]['energy'] for target in target_stats_list]).float(),
                dim=0
            ),
            'num_atom_pairs': torch.mean(
                torch.tensor([target["all"]['clash_info'][threshold]['num_atom_pairs'] for target in target_stats_list]).float()
            ),
            'num_clashes': torch.mean(
                torch.tensor([target["all"]['clash_info'][threshold]['num_clashes'] for target in target_stats_list]).float()
            ),
        } for threshold in first_target_stats["all"]['clash_info'].keys()
    }
    
    # print(f'overall_stats = {pprint.pformat(overall_stats)}')
    return overall_stats

# Makes the dictionary serializable in JSON.
def tensor_to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.ndim > 0 else obj.item()
    if isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    
    return obj

if __name__ == "__main__":
    targets_dir = '' # location of folder with native structures
    predictions_dir = '' # location of folder holding structures with packed side-chains

    per_target = results_per_target(targets_dir=targets_dir, predictions_dir=predictions_dir)
    across_all_targets = aggregate_dataset_stats(target_stats_list=per_target)

    stats_dir = './assessment_results' # location of folder in which outputted results will be placed
    stats_file_name = 'all_targets.json'
    converted = tensor_to_python(across_all_targets)
    os.makedirs(stats_dir, exist_ok=True)
    stats_file_name = os.path.join(stats_dir, stats_file_name)
    with open(stats_file_name, "w") as file:
        json.dump(converted, file, indent=4)
    print(f"Saved to {stats_file_name} .")

    pass
