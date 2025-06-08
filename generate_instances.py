import os
import random

def generate_random_instance(filename, rows, cols, num_agents, obstacle_prob=0.2):
    with open(filename, 'w') as f:
        f.write(f"{rows} {cols}\n")

        # Generate map
        my_map = []
        for _ in range(rows):
            row = ''
            for _ in range(cols):
                row += '@' if random.random() < obstacle_prob else '.'
            my_map.append(row)
            f.write(row + '\n')

        # Generate starts and goals
        f.write(f"{num_agents}\n")
        free_cells = [(r, c) for r in range(rows) for c in range(cols) if my_map[r][c] == '.']
        starts = random.sample(free_cells, num_agents)
        goals = random.sample([c for c in free_cells if c not in starts], num_agents)

        for (sx, sy), (gx, gy) in zip(starts, goals):
            f.write(f"{sx} {sy} {gx} {gy}\n")

# Generate multiple instances
os.makedirs("instances_random", exist_ok=True)

for i in range(500):
    filename = f"instances_random/{i}.txt"
    generate_random_instance(filename, rows=20, cols=20, num_agents=5)
