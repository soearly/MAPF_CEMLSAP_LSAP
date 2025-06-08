#!/usr/bin/python
import argparse
import glob
from pathlib import Path
from cbs import CBSSolver
from visualize import Animation
from single_agent_planner import get_sum_of_cost
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import csv
from scipy.optimize import linear_sum_assignment
from collections import deque

SOLVER = "CBS"

def extract_number(f):
    return int(re.search(r'(\d+)', os.path.basename(f)).group())

def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise FileNotFoundError(filename + " does not exist.")
    with open(filename, 'r') as f:
        rows, columns = map(int, f.readline().split(' '))
        my_map = []
        for _ in range(rows):
            line = f.readline()
            my_map.append([cell == '@' for cell in line.strip()])
        num_agents = int(f.readline())
        starts, goals = [], []
        for _ in range(num_agents):
            sx, sy, gx, gy = map(int, f.readline().split(' '))
            starts.append((sx, sy))
            goals.append((gx, gy))
    return my_map, starts, goals

def is_reachable(my_map, start, goal):
    rows, cols = len(my_map), len(my_map[0])
    visited = set()
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not my_map[nr][nc] and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False

def lsap_assignment(my_map, starts, goals):
    n = len(starts)
    INF_COST = 9999
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if is_reachable(my_map, starts[i], goals[j]):
                cost_matrix[i, j] = abs(starts[i][0] - goals[j][0]) + abs(starts[i][1] - goals[j][1])
            else:
                cost_matrix[i, j] = INF_COST

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    best_cost = cost_matrix[row_ind, col_ind].sum()

    if best_cost >= INF_COST:
        return [], INF_COST
    else:
        best_perm = [goals[j] for j in col_ind]
        return best_perm, best_cost

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

def get_arrival_time(path):
    goal = path[-1]
    for t in range(len(path) - 2, -1, -1):
        if path[t] != goal:
            return t + 1
    return 0

def cem_plan(map_file, my_map, starts, goals, disjoint=False, num_samples=100, num_elite=10, maxiter=20):
    dim = len(goals)
    mean = np.zeros(dim)
    cov = 0.01 * np.identity(dim)
    cost_track = []

    # Compute lamda_lsap from LSAP assignment
    best_perm_lsap, _ = lsap_assignment(my_map, starts, goals)
    if not best_perm_lsap:
        print("⚠️ LSAP failed to find feasible assignment; lamda_lsap set to zeros.")
        lamda_lsap = np.zeros(dim)
    else:
        lamda_lsap = np.zeros(dim)
        goal_indices = {goal: idx for idx, goal in enumerate(goals)}
        for pos, g in enumerate(best_perm_lsap):
            lamda_lsap[goal_indices[g]] = pos

        # Normalize to [0, 1]
        if np.max(lamda_lsap) != np.min(lamda_lsap):
            lamda_lsap = (lamda_lsap - np.min(lamda_lsap)) / (np.max(lamda_lsap) - np.min(lamda_lsap))
            # Optional: center around 0 for symmetric influence
            # lamda_lsap = lamda_lsap - 0.5
        else:
            lamda_lsap = np.zeros_like(lamda_lsap)

    for _ in range(maxiter):
        lamda = np.random.multivariate_normal(mean, cov, size=num_samples)
        lamda = lamda + lamda_lsap[np.newaxis, :]  # Add normalized LSAP bias

        goal_perm_list = compute_goal_matrix_list(goals, lamda)
        with multiprocessing.Pool() as pool:
            costs = pool.starmap(process_goal_permutation,
                                 [(perm, my_map, starts, disjoint) for perm in goal_perm_list])
        elite_idxs = np.argsort(costs)[:num_elite]
        cost_track.append(min(costs))
        mean = np.mean(lamda[elite_idxs], axis=0)
        cov = np.cov(lamda[elite_idxs], rowvar=False)

    idx_sort = np.argsort(np.abs(mean))
    permuted = [goals[idx] for idx in idx_sort]
    best_cost = cost_track[-1]

    filename_only = os.path.basename(map_file)

    os.makedirs("results_cemlsap_v4", exist_ok=True)
    result_file = os.path.join("results_cemlsap_v4", "results_cemlsap_vs.csv")
    goal_perm_str = ";".join([f"{g[0]}-{g[1]}" for g in permuted])
    with open(result_file, "a") as f:
        f.write(f"{filename_only},{best_cost},{goal_perm_str}\n")

    os.makedirs("plots_cemlsap_v4", exist_ok=True)
    plot_path = os.path.join("plots_cemlsap_v4", filename_only.replace(".txt", "_cem_plot.png"))
    plt.figure(figsize=(10, 6))
    plt.plot(cost_track)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title(f"CEM Optimization Progress ({filename_only})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.clf()

    print("***Simulating Best Plan Found by CEM***")
    try:
        cbs = CBSSolver(my_map, starts, permuted)
        paths = cbs.find_solution(disjoint=disjoint)
        if paths:
            animation = Animation(my_map, starts, permuted, paths)
            os.makedirs("videos_cemlsap_v4", exist_ok=True)
            gif_path = os.path.join("videos_cemlsap_v4", filename_only.replace(".txt", "_cem_animation.gif"))
            animation.save(gif_path, time_step=1.0, writer="pillow")
            print(f"✅ Animation saved to: {gif_path}")

            sum_of_costs = sum(len(path) - 1 for path in paths)
            arrival_times = [get_arrival_time(path) for path in paths]
            makespan = max(arrival_times)
            flowtime = sum(arrival_times)

            print(f"📊 Metrics for {filename_only}")
            print(f"   Sum of Costs: {sum_of_costs}")
            print(f"   Makespan:     {makespan}")
            print(f"   Flowtime:     {flowtime}")

            with open("results_cemlsap_v4/soc_results.csv", "a", newline='') as f1, \
                 open("results_cemlsap_v4/makespan_results.csv", "a", newline='') as f2, \
                 open("results_cemlsap_v4/flowtime_results.csv", "a", newline='') as f3:
                csv.writer(f1).writerow([filename_only, sum_of_costs])
                csv.writer(f2).writerow([filename_only, makespan])
                csv.writer(f3).writerow([filename_only, flowtime])

            move_log_path = os.path.join("results_cemlsap_v4", filename_only.replace(".txt", "_movements.txt"))
            with open(move_log_path, "w") as f:
                max_len = max(len(p) for p in paths)
                for t in range(max_len):
                    f.write(f"Timestep {t}:\n")
                    for i, path in enumerate(paths):
                        pos = path[min(t, len(path)-1)]
                        f.write(f"  Agent {i}: {pos}\n")
                    f.write("\n")

            per_agent_log_path = os.path.join("results_cemlsap_v4", filename_only.replace(".txt", "_agentpaths.txt"))
            with open(per_agent_log_path, "w") as f:
                for i, path in enumerate(paths):
                    f.write(f"Agent {i}:\n")
                    for t, pos in enumerate(path):
                        f.write(f"  Time {t}: {pos}\n")
                    f.write("\n")

        else:
            print(f"⚠️  No valid paths found for {filename_only}, skipping animation and metrics.")
    except Exception as e:
        print(f"⚠️  Animation error for {filename_only}: {e}")

    return permuted, best_cost, cost_track

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CEM-based goal assignment on MAPF instance(s)")
    parser.add_argument('--instance', type=str, required=True,
                        help='Path to MAPF instance file(s), e.g., "instances/*.txt"')
    parser.add_argument('--disjoint', action='store_true', help='Use disjoint splitting strategy')
    parser.add_argument('--solver', type=str, default='CBS', help='Solver to use (currently only CBS supported)')
    parser.add_argument('--batch', action='store_true', help='(Unused) Suppress display (animation is saved regardless)')
    args = parser.parse_args()

    os.makedirs("results_cemlsap_v4", exist_ok=True)
    for metric_file, header in [
        ("soc_results.csv", ["instance", "sum_of_costs"]),
        ("makespan_results.csv", ["instance", "makespan"]),
        ("flowtime_results.csv", ["instance", "flowtime"])
    ]:
        path = os.path.join("results_cemlsap_vs", metric_file)
        if not os.path.exists(path):
            with open(path, "w", newline='') as f:
                csv.writer(f).writerow(header)

    for file in sorted(glob.glob(args.instance), key=extract_number):
        print(f"\n=== Processing {file} ===")
        try:
            my_map, starts, goals = import_mapf_instance(file)
            cem_plan(file, my_map, starts, goals, disjoint=args.disjoint)
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")
