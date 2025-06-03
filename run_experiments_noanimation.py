#!/usr/bin/python
import argparse
import glob
from pathlib import Path
from cbs import CBSSolver
from single_agent_planner import get_sum_of_cost
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import os

SOLVER = "CBS"


def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise FileNotFoundError(filename + " does not exist.")
    with open(filename, 'r') as f:
        rows, columns = map(int, f.readline().split(' '))
        my_map = []
        for _ in range(rows):
            line = f.readline()
            my_map.append([cell == '@' for cell in line])
        num_agents = int(f.readline())
        starts, goals = [], []
        for _ in range(num_agents):
            sx, sy, gx, gy = map(int, f.readline().split(' '))
            starts.append((sx, sy))
            goals.append((gx, gy))
    return my_map, starts, goals


def process_goal_permutation(goal_permutation, my_map, starts, disjoint):
    try:
        cbs = CBSSolver(my_map, starts, goal_permutation)
        paths = cbs.find_solution(disjoint)
        return get_sum_of_cost(paths) if paths else float('inf')
    except Exception as e:
        print(f"[ERROR] Permutation {goal_permutation}: {e}")
        return float('inf')


def compute_goal_matrix_list(goals, lamda):
    return [[goals[idx] for idx in np.argsort(np.abs(l))] for l in lamda]


def cem_plan(map_file, my_map, starts, goals, disjoint=False, num_samples=100, num_elite=10, maxiter=20):
    dim = len(goals)
    mean = np.zeros(dim)
    cov = 0.01 * np.identity(dim)
    cost_track = []

    for _ in range(maxiter):
        lamda = np.random.multivariate_normal(mean, cov, size=num_samples)
        goal_perm_list = compute_goal_matrix_list(goals, lamda)
        with multiprocessing.Pool() as pool:
            costs = pool.starmap(process_goal_permutation,
                                 [(perm, my_map, starts, disjoint) for perm in goal_perm_list])
        elite_idxs = np.argsort(costs)[:num_elite]
        cost_track.append(min(costs))
        mean = np.mean(lamda[elite_idxs], axis=0)
        cov = np.cov(lamda[elite_idxs], rowvar=False)

    # Final result from best permutation
    idx_sort = np.argsort(np.abs(mean))
    permuted = [goals[idx] for idx in idx_sort]
    best_cost = cost_track[-1]

    # Save results
    os.makedirs("results_1", exist_ok=True)
    filename_only = os.path.basename(map_file)
    result_file = os.path.join("results_1", "results_1.csv")
    goal_perm_str = ";".join([f"{g[0]}-{g[1]}" for g in permuted])
    with open(result_file, "a") as f:
        f.write(f"{filename_only},{best_cost},{goal_perm_str}\n")

    # Save plot
    os.makedirs("plots_1", exist_ok=True)
    plot_path = os.path.join("plots_1", filename_only.replace(".txt", "_cem_plot.png"))
    plt.figure(figsize=(10, 6))
    plt.plot(cost_track)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title(f"CEM Optimization Progress ({filename_only})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.clf()

    # XML saving can be implemented here if needed
    # Example:
    # save_to_xml(permuted, paths, f"results_1/{filename_only.replace('.txt', '.xml')}")

    return permuted, best_cost, cost_track


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CEM-based goal assignment on MAPF instance(s)")
    parser.add_argument('--instance', type=str, required=True,
                        help='Path to MAPF instance file(s), e.g., "instances/*.txt"')
    parser.add_argument('--disjoint', action='store_true', help='Use disjoint splitting strategy')

    # Optional for compatibility
    parser.add_argument('--solver', type=str, default='CBS', help='Solver to use (currently only CBS supported)')
    parser.add_argument('--batch', action='store_true',
                        help='(Unused) Suppress display (animation is removed)')

    args = parser.parse_args()

    for file in sorted(glob.glob(args.instance)):
        print(f"\n=== Processing {file} ===")
        try:
            my_map, starts, goals = import_mapf_instance(file)
            cem_plan(file, my_map, starts, goals, disjoint=args.disjoint)
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")
