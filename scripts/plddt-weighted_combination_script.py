import os
import random
import math
from tqdm import tqdm
import multiprocessing
import time
import matplotlib.pyplot as plt

import pyrosetta
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.scoring import fa_rep
from pyrosetta.rosetta.core.pose import Pose

# Defines constants
DATA_ROOT_DIR = f"/home/common/proj/side_chain_packing/data/FINAL"
OUTPUT_ROOT_DIR = f"./packbench_reproducibility"
casps = ['casp14', 'casp15']
bb_types = ['af2', 'af3']
all_tools = [
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
random.seed(1048596)

def pyrosetta_init():
    init('-out:levels core.conformation.Conformation:error '
        'core.pack.pack_missing_sidechains:error '
        'core.pack.dunbrack.RotamerLibrary:error '
        'core.scoring.etable:error '
        '-ex1 -ex2 -ex3 -ex4 '
        '-multi_cool_annealer 5 '
        '-no_his_his_pairE '
        '-linmem_ig 1 '
        '-ignore_unrecognized_res 1 '
        '-detect_disulf 0 '
        '-mute all')
    
def load_library_structures(target, repacked_dirs):
    library_structures = {}
    for dir in tqdm(repacked_dirs, desc=f"Loading library structures for target {target}"):
        filepath = os.path.join(repacked_dirs[dir], target)
        library_structures[dir] = pose_from_pdb(filepath)
    return library_structures

def save_graph(energies, filepath):
    x = list(range(len(energies)))

    plt.plot(x, energies, marker='o', linestyle='-', color='b', label='Energy Values')
    plt.xlabel("Iteration Number")
    plt.ylabel("Energy of Current Pose")
    plt.title("Energy Over Time")
    plt.grid(True)
    plt.legend()

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def pack_target(
        target, dataset_name, backbone_tool, nonnative_dir, repacked_dirs, output_dir, length):
    # Loads the necessary poses
    print(f"Now packing target {target}")
    library_structures = load_library_structures(target, repacked_dirs)
    original_pdb_path = os.path.join(nonnative_dir, target)
    current_pose = pose_from_pdb(original_pdb_path)

    # Gets the confidence value for each residue
    pose_info = current_pose.pdb_info()
    confidence_per_residue = {}
    for i in range(1, current_pose.total_residue() + 1):
        residue = current_pose.residue(i)
        confidences = []
        backbone_names = ["N", "CA", "C", "O"]
        num_backbone_counted = 0
        for j in range(1, residue.natoms() + 1):
            if residue.atom_name(j).strip() in backbone_names:
                confidences.append(float(pose_info.bfactor(res=i, atom_index=j)) / 100.0)
                num_backbone_counted += 1
        assert num_backbone_counted == 4
        avg_confidence = sum(confidences) / float(len(confidences))
        confidence_per_residue[i] = avg_confidence

    # Initializes the energy function
    scorefxn = pyrosetta.create_score_function("ref2015")

    # Runs simulated annealing (in this case, greedy optimization)
    energies = [scorefxn(current_pose)]
    attempted, accepted, rejected = 0, 0, 0
    num_iterations = length
    for iter in tqdm(range(num_iterations), desc=f"Mutating residues for {target}"):
        # Selects the residue(s) and tool
        residue_idx = random.randint(1, current_pose.total_residue())
        residue = current_pose.residue(residue_idx)
        selected_tool = random.choice(list(library_structures.keys()))

        # Mutates the angles and gets the new energy score
        if residue_idx > library_structures[selected_tool].total_residue() \
                or residue.nchi() > library_structures[selected_tool].residue(residue_idx).nchi() \
                or residue.nchi() == 0:
            continue
        chi_num = random.randint(1, residue.nchi())
        old_angle = current_pose.chi(chi_num, residue_idx)

        nonnative_prob = confidence_per_residue[residue_idx]
        repacked_prob = 1 - nonnative_prob
        nonnative_angle = old_angle
        repacked_angle = library_structures[selected_tool].chi(chi_num, residue_idx)

        setting_to = nonnative_prob * nonnative_angle + repacked_prob * repacked_angle
        current_pose.set_chi(chi_num, residue_idx, setting_to)
        mutated_energy = scorefxn(current_pose)

        # Chooses to accept or reject the mutation
        delta_energy = mutated_energy - energies[-1]    
        attempted += 1
        if delta_energy < 0:
            accepted += 1
            energies.append(mutated_energy)
            # print(f"Accepting {selected_tool} over {backbone_tool} for residue {residue_idx}")
        else:
            rejected += 1
            energies.append(energies[-1])

            current_pose.set_chi(chi_num, residue_idx, old_angle)
            assert scorefxn(current_pose) - energies[-1] < 1e-6
            # print(f"Rejecting {selected_tool} over {backbone_tool} for residue {residue_idx}")

    # Writes the final PDB
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, target)
    current_pose.dump_pdb(output_path)
    target_name, _ = os.path.splitext(target)
    energies_path = os.path.join(output_dir, f"energies_{target_name}.txt")
    with open(energies_path, "w") as f:
        f.write(f"Attempted = {attempted}\n")
        f.write(f"Accepted = {accepted}\n")
        f.write(f"Rejected = {rejected}\n")
        f.write(f"Percent accepted = {float(accepted) / float(attempted)}\n")
        f.write(f"Energies:\n{energies}\n")
    plot_path = os.path.join(output_dir, f"energies_plot_{target_name}.png")
    save_graph(energies, plot_path)

def pack_dir(dataset_name, backbone_tool, nonnative_dir, repacked_dirs, output_dir, length):
    # Sets up the arguments for the different processes
    args_list = []
    for target in os.listdir(nonnative_dir): # Sorting the targets doesn't matter if multiprocessing
        args_list.append((
            target,
            dataset_name,
            backbone_tool,
            nonnative_dir,
            repacked_dirs,
            output_dir,
            length,
        ))
        # break # To only test with one PDB

    # Starts all the processes
    num_processes = os.cpu_count() // 2
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        _ = pool.starmap(pack_target, args_list)
    print(f"Total time for all targets = {time.time() - start_time}")

def main():
    tools_subset = all_tools
    tools_name = "all_tools"
    iter_count = 6500
    for casp in casps:
        for af in bb_types:
            if casp == "casp14":
                continue

            nonnative_dir = os.path.join(DATA_ROOT_DIR, casp, f"{casp}_{af}_predictions")
            repacked_dirs = {}
            for tool in tools_subset:
                repacked_dirs[tool] = os.path.join(DATA_ROOT_DIR, casp, f"repacking_{af}_bb", f"{tool}_predictions")

            output_dir = os.path.join(OUTPUT_ROOT_DIR, f"{casp}_{af}")
            pack_dir(casp, af, nonnative_dir, repacked_dirs, output_dir, iter_count)

        #     break
        # break

if __name__ == "__main__":
    pyrosetta_init()
    main()
