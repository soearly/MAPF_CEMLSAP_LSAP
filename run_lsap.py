#!/usr/bin/python
import argparse
import glob
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
from pathlib import Path
from visualize import Animation
from cbs import CBSSolver
from single_agent_planner import get_sum_of_cost
import re
import csv

def extract_number(f):
    return int(re.search(r'(\d+)', os.path.basename(f)).group())

def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise FileNotFoundError(f"{filename} does not exist.")
    with open(filename, 'r') as f:
        rows, columns = map(int, f.readline().split())
        my_map = []
        for _ in range(rows):
            line = f.readline()
            my_map.append([c == '@' for c in line.strip()])
        num_agents = int(f.readline())
        starts, goals = [], []
        for _ in range(num_agents):
            sx, sy, gx, gy = map(int, f.readline().split())
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

def get_arrival_time(path):
    goal = path[-1]
    for t in range(len(path) - 2, -1, -1):
        if path[t] != goal:
            return t + 1
    return 0

def write_timestep_view(log_path, instance_file, paths):
    with open(log_path, "a") as f:
        f.write(f"== Timestep-wise View for {instance_file} ==\n")
        max_time = max(len(path) for path in paths)
        for t in range(max_time):
            f.write(f"Timestep {t}:\n")
            for i, path in enumerate(paths):
                pos = path[t] if t < len(path) else path[-1]
                f.write(f"  Agent {i}: {pos}\n")
            f.write("\n")

def write_per_agent_view(log_path, instance_file, paths):
    with open(log_path, "a") as f:
        f.write(f"== Per-Agent View for {instance_file} ==\n")
        for i, path in enumerate(paths):
            f.write(f"Agent {i}:\n")
            for t, pos in enumerate(path):
                f.write(f"  Time {t}: {pos}\n")
            f.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LSAP-based MAPF goal assignment with CBS path planning")
    parser.add_argument('--instance', type=str, required=True,
                        help='Path to MAPF instance file(s), e.g., "instances/*.txt"')
    parser.add_argument('--disjoint', action='store_true', help='Use disjoint splitting strategy in CBS')
    parser.add_argument('--solver', type=str, default='CBS', help='Solver to use (only CBS supported)')
    parser.add_argument('--batch', action='store_true', help='Suppress display (unused here)')

    args = parser.parse_args()

    os.makedirs("results_lsap_flow_makespan_2move", exist_ok=True)
    os.makedirs("videos_lsap_flow_makespan_2move", exist_ok=True)

    result_file_path = os.path.join("results_lsap_flow_makespan_2move", "results_lsap.csv")
    timestep_log_path = os.path.join("results_lsap_flow_makespan_2move", "movements_timestep.txt")
    agent_log_path = os.path.join("results_lsap_flow_makespan_2move", "movements_per_agent.txt")
    flow_file_path = os.path.join("results_lsap_flow_makespan_2move", "flowtime_results.csv")
    makespan_file_path = os.path.join("results_lsap_flow_makespan_2move", "makespan_results.csv")

    if not os.path.exists(flow_file_path):
        with open(flow_file_path, "w", newline='') as f:
            csv.writer(f).writerow(["instance", "flowtime"])

    if not os.path.exists(makespan_file_path):
        with open(makespan_file_path, "w", newline='') as f:
            csv.writer(f).writerow(["instance", "makespan"])

    with open(result_file_path, "w", buffering=1) as result_file, \
         open(flow_file_path, "a", newline='') as flow_f, \
         open(makespan_file_path, "a", newline='') as makespan_f:

        flow_writer = csv.writer(flow_f)
        makespan_writer = csv.writer(makespan_f)

        for file in sorted(glob.glob(args.instance), key=extract_number):
            print(f"\n📂 Running on: {file}")
            my_map, starts, goals = import_mapf_instance(file)

            print("🚀 Optimizing with LSAP...")
            best_perm, best_cost = lsap_assignment(my_map, starts, goals)

            filename_only = os.path.basename(file)
            if best_cost == 9999:
                print("⚠️ No feasible assignment found due to blocked start-goal pairs.")
                result_file.write(f"{filename_only},9999,No feasible assignment\n")
                continue

            try:
                cbs = CBSSolver(my_map, starts, best_perm)
                paths = cbs.find_solution(args.disjoint)
                if not paths:
                    print("⚠️ No valid paths found for final permutation.")
                    result_file.write(f"{filename_only},9999,No valid paths\n")
                else:
                    actual_cost = get_sum_of_cost(paths)
                    arrival_times = [get_arrival_time(p) for p in paths]
                    flowtime = sum(arrival_times)
                    makespan = max(arrival_times)

                    goal_str = ";".join([f"{g[0]}-{g[1]}" for g in best_perm])
                    result_file.write(f"{filename_only},{actual_cost},{goal_str}\n")
                    print(f"✅ Result saved for {filename_only}: Actual Cost={actual_cost}, Flowtime={flowtime}, Makespan={makespan}")

                    # Write flow and makespan
                    flow_writer.writerow([filename_only, flowtime])
                    makespan_writer.writerow([filename_only, makespan])

                    # Write movement logs
                    write_timestep_view(log_path=timestep_log_path,
                                        instance_file=filename_only,
                                        paths=paths)

                    write_per_agent_view(log_path=agent_log_path,
                                         instance_file=filename_only,
                                         paths=paths)

                    # Create animation
                    animation = Animation(my_map, starts, best_perm, paths)
                    gif_filename = os.path.join("videos_lsap_flow_makespan_2move", filename_only.replace(".txt", "_lsap_animation.gif"))
                    animation.save(gif_filename, time_step=1.0, writer='pillow')
                    print(f"🎞️ Animation saved to: {gif_filename}")
            except Exception as e:
                print(f"❌ Error during animation or path finding: {e}")
                result_file.write(f"{filename_only},9999,Error during planning\n")
