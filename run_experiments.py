#!/usr/bin/python
import argparse
import glob
from pathlib import Path
from cbs import CBSSolver
from independent import IndependentSolver
from prioritized import PrioritizedPlanningSolver
from visualize import Animation
from single_agent_planner import get_sum_of_cost
from itertools import permutations
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import os

SOLVER = "CBS"

# Function to print MAPF instance with start and goal locations
def print_mapf_instance(my_map, starts, goals):
    print('Start locations')
    print_locations(my_map, starts)
    print('Goal locations')
    print_locations(my_map, goals)


# Helper function to print locations on the map
def print_locations(my_map, locations):
    starts_map = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
    for i in range(len(locations)):
        starts_map[locations[i][0]][locations[i][1]] = i
    to_print = ''
    for x in range(len(my_map)):
        for y in range(len(my_map[0])):
            if starts_map[x][y] >= 0:
                to_print += str(starts_map[x][y]) + ' '
            elif my_map[x][y]:
                to_print += '@ '
            else:
                to_print += '. '
        to_print += '\n'
    print(to_print)


# Function to import MAPF instance from a file
def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    # Read map dimensions
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    # Initialize map
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)  # Obstacle
            elif cell == '.':
                my_map[-1].append(False)  # Free space
    # Read number of agents
    line = f.readline()
    num_agents = int(line)
    # Initialize starts and goals
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx, gy))
    f.close()
    return my_map, starts, goals


# Function to process each goal permutation and calculate cost
def process_goal_permutation(goal_permutation, my_map, starts, disjoint):
    cbs = CBSSolver(my_map, starts, goal_permutation)
    paths = cbs.find_solution(disjoint)
    cost = get_sum_of_cost(paths)
    return cost


# Function to compute the list of goal permutations based on lambda (parameter distribution)
def compute_goal_matrix_list(goals, lamda):
    goal_perm_list = []
    for i in range(len(lamda)):
        idx_sort = np.argsort(np.abs(lamda[i]))  # Sort lambda samples based on magnitude
        permuted = [goals[idx] for idx in idx_sort]  # Permute the goal list based on sorted lambda
        goal_perm_list.append(permuted)
    return goal_perm_list


# Cross-Entropy Method to optimize the cost function
def cem_plan(my_map, starts, goals, disjoint=False, num_samples=100, num_elite=10, maxiter=20):
    dim = len(goals)  # Number of goals (agents)
    mean = np.zeros(dim)  # Initial mean of the distribution
    cov = 0.01*np.identity(dim)  # Initial covariance (identity matrix)
    cost_track = []
    # Main loop for Cross-Entropy Method
    for _ in range(maxiter):
        # Sample lambda values from multivariate normal distribution
        lamda = np.random.multivariate_normal(mean, cov, size=num_samples)
        goal_perm_list = compute_goal_matrix_list(goals, lamda)

        # Compute costs for each goal permutation
        with multiprocessing.Pool() as pool:
            costs = pool.starmap(process_goal_permutation,
                                 [(perm, my_map, starts, disjoint) for perm in goal_perm_list])

        # Select elite samples based on lowest cost
        idx_sorted = np.argsort(costs)  # Sort based on cost
        cost_min = np.min(costs)
        cost_track.append(cost_min)
        lamda_elite = lamda[idx_sorted[:num_elite]]  # Select the elite samples

        # Update distribution: mean and covariance
        mean = np.mean(lamda_elite, axis=0)
        cov = np.cov(lamda_elite, rowvar=False)

    # Get the best permutation (minimum cost)
    best_idx = idx_sorted[0]
    best_perm = goal_perm_list[best_idx]
    best_cost = costs[best_idx]
    #plt.plot(cost_track)
    #plt.show()

    # Plot cost convergence
    plt.plot(cost_track)
    plt.title("CEM Cost Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")


    # Create 'plots' folder if it doesn't exist
    os.makedirs("plots_5", exist_ok=True)

    # Create filename based on instance file name
    plot_filename = os.path.join("plots_5", file.replace(".txt", "_cem_plot.png"))
    print(f"Instance filename: {file}")  # Debug line
    filename_only = os.path.basename(file)  # Remove path
    plot_filename = os.path.join("plots_5", filename_only.replace(".txt", "_cem_plot.png"))

    # Save the plot and display it
    plt.savefig(plot_filename)
    #plt.show()
    plt.clf()  # Clear figure for next instance (if running batch)



    return best_perm, best_cost

# Main entry point for running experiments
if __name__ == '__main__':
    # Argument parsing for running experiments
    parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')
    parser.add_argument('--instance', type=str, default=None, help='The name of the instance file(s)')
    parser.add_argument('--batch', action='store_true', default=False, help='Use batch output instead of animation')
    parser.add_argument('--disjoint', action='store_true', default=False, help='Use the disjoint splitting')
    parser.add_argument('--solver', type=str, default=SOLVER,
                        help='The solver to use (one of: {CBS, Independent, Prioritized}), defaults to ' + str(SOLVER))

    args = parser.parse_args()

    result_file = open("results_5.csv", "w", buffering=1)

    # Loop over instance files to run experiments
    for file in sorted(glob.glob(args.instance)):
        print("***Import an instance***")
        my_map, starts, goals_permutation = import_mapf_instance(file)
        if args.solver == "CBS":
            print("***Run CBS with CEM Optimization***")

            # Run Cross Entropy Method to find the best goal assignment
            best_perm, best_cost = cem_plan(my_map, starts, goals_permutation, args.disjoint)
            print(f"Best goal assignment: {best_perm}")
            print(f"Best cost: {best_cost}")
            #result_file.write("{},{}\n".format(file, best_cost))
            result_file.write("{},{},{}\n".format(file, best_cost, ";".join([f"{x[0]}-{x[1]}" for x in best_perm])))

        else:
            raise RuntimeError("Unknown solver!")

        # If not in batch mode, run simulation for testing the paths
        #if not args.batch:
        #    print("***Test paths on a simulation***")
        if not args.batch:
            print("***Simulating Best Plan Found by CEM***")

            # Run CBS again to get the actual path for animation
            cbs = CBSSolver(my_map, starts, best_perm)
            paths = cbs.find_solution(args.disjoint)

            # Show the animation
            #animation.show()

            animation = Animation(my_map, starts, best_perm, paths)

            # Save as GIF
            filename_only = os.path.basename(file)
            gif_filename = os.path.join("videos", filename_only.replace(".txt", "_animation.gif"))

            os.makedirs("videos", exist_ok=True)
            animation.save(gif_filename, time_step=1.0, writer='pillow')


    result_file.close()
