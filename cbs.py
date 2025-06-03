import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost

DEBUG = False


class NoSolutionException(Exception):
    """Raised when no solution can be found for a path."""
    pass


def find_shorter_paths(path_1, path_2):
    path1 = path_1.copy()
    path2 = path_2.copy()
    if len(path1) < len(path2):
        shorter = path1
        path = len(path2) - len(path1)
    else:
        shorter = path2
        path = len(path1) - len(path2)

    for _ in range(path):
        shorter.append(shorter[-1])
    return path1, path2


def detect_collision(path_1, path_2):
    path1, path2 = find_shorter_paths(path_1, path_2)

    for t in range(len(path1)):
        position1 = get_location(path1, t)
        position2 = get_location(path2, t)
        if position1 == position2:
            return [position1], t, 'vertex'
        if t < len(path1) - 1:
            next_position1 = get_location(path1, t + 1)
            next_position2 = get_location(path2, t + 1)
            if position1 == next_position2 and position2 == next_position1:
                return [position1, next_position1], t + 1, 'edge'
    return None


def detect_collisions(paths):
    collisions = []
    for a1 in range(len(paths)):
        for a2 in range(a1 + 1, len(paths)):
            collisions_info = detect_collision(paths[a1], paths[a2])
            if collisions_info:
                collisions.append({
                    'a1': a1,
                    'a2': a2,
                    'loc': collisions_info[0],
                    'timestep': collisions_info[1],
                    'type': collisions_info[2]
                })
    return collisions


def standard_splitting(collision):
    constraints = []
    if collision['type'] == 'vertex':
        constraints.append({
            'agent': collision['a1'],
            'loc': collision['loc'],
            'timestep': collision['timestep'],
            'final': False
        })
        constraints.append({
            'agent': collision['a2'],
            'loc': collision['loc'],
            'timestep': collision['timestep'],
            'final': False
        })
    elif collision['type'] == 'edge':
        constraints.append({
            'agent': collision['a1'],
            'loc': collision['loc'],
            'timestep': collision['timestep'],
            'final': False
        })
        constraints.append({
            'agent': collision['a2'],
            'loc': list(reversed(collision['loc'])),
            'timestep': collision['timestep'],
            'final': False
        })
    return constraints


def disjoint_splitting(collision):
    choice = random.randint(0, 1)
    agents = [collision['a1'], collision['a2']]
    agent = agents[choice]

    if choice == 0:
        loc = collision['loc']
    else:
        loc = list(reversed(collision['loc']))

    return [
        {
            'agent': agent,
            'loc': loc,
            'timestep': collision['timestep'],
            'positive': True,
            'final': False
        },
        {
            'agent': agent,
            'loc': loc,
            'timestep': collision['timestep'],
            'positive': False,
            'final': False
        }
    ]


def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst


class CBSSolver(object):
    def __init__(self, my_map, starts, goals, time_max=None):
        self.start_time = 0
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.time_max = time_max if time_max else float('inf')

        self.open_list = []
        self.cont = 0

        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        if DEBUG:
            print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        if DEBUG:
            print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        self.start_time = timer.time()
        root = {
            'cost': 0,
            'constraints': [],
            'paths': [],
            'collisions': []
        }
        for i in range(self.num_of_agents):
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise NoSolutionException(f"No solution for agent {i} with initial constraints.")
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        while self.open_list and timer.time() - self.start_time < self.time_max:
            next_node = self.pop_node()
            if not next_node['collisions']:
                self.print_results(next_node)
                return next_node['paths']

            collision = next_node['collisions'][0]
            constraints = standard_splitting(collision) if not disjoint else disjoint_splitting(collision)

            for c in constraints:
                skip_node = False
                new_node = {
                    'cost': 0,
                    'constraints': [*next_node['constraints'], c],
                    'paths': next_node['paths'].copy(),
                    'collisions': []
                }
                agent = c['agent']
                path = a_star(self.my_map, self.starts[agent], self.goals[agent],
                              self.heuristics[agent], agent, new_node['constraints'])
                if path is None:
                    continue  # Try the next constraint instead of raising

                new_node['paths'][agent] = path

                if c.get('positive', False):
                    renew_agents = paths_violate_constraint(c, new_node['paths'])
                    for r_agent in renew_agents:
                        new_c = c.copy()
                        new_c['agent'] = r_agent
                        new_c['positive'] = False
                        new_node['constraints'].append(new_c)
                        r_path = a_star(self.my_map, self.starts[r_agent], self.goals[r_agent],
                                        self.heuristics[r_agent], r_agent, new_node['constraints'])
                        if r_path is None:
                            skip_node = True
                            break
                        new_node['paths'][r_agent] = r_path

                if not skip_node:
                    new_node['collisions'] = detect_collisions(new_node['paths'])
                    new_node['cost'] = get_sum_of_cost(new_node['paths'])
                    self.push_node(new_node)

        raise NoSolutionException("CBS failed to find a solution within time or constraints.")

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
