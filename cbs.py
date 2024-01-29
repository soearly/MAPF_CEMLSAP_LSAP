import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost

DEBUG = False


def find_shorter_paths(path_1, path_2):
    # Between path1 and path2, find shorter path
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
    ##############################
    # 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    # Detects if an agent collides with another even after one of the two reached the goal
    path1, path2 = find_shorter_paths(path_1, path_2)

    for t in range( len(path1) ):
        # Vertex collison check
        position1 = get_location(path1, t)
        position2 = get_location(path2, t)
        if position1 == position2:
            return [position1], t, 'vertex'
        # Edge collison check
        if t < len(path1) - 1:
            next_position1 = get_location(path1, t + 1)
            next_position2 = get_location(path2, t + 1)
            if position1 == next_position2 and position2 == next_position1:
                return [position1, next_position1], t + 1, 'edge'
    return None


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           Use detect_collision function to find a collision between two robots.

    collisions = []
    # a1: agent 1 and a2: agent 2
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
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    constraints = []

    # # Task 3.2 Testing
    # constraints.append({
    #    'agent': 0,
    #    'loc': [(1, 4)],
    #    'timestep': 3,
    #    'final' : False,
    # })
    # constraints.append({
    #     'agent': 1,
    #     'loc': [(1, 4)],
    #     'timestep': 3,
    #     'final' : False,
    # })


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
    ##############################
    # 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly
    choice = random.randint(0, 1) # choose the agent randomly
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
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals, time_max=None):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.start_time = 0
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.time_max =  time_max if time_max else float('inf')

        self.open_list = []
        self.cont = 0

        # compute heuristics for the low-level search
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
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """
        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {
            'cost': 0,
            'constraints': [],
            'paths': [],
            'collisions': []
        }
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)
        # Task 3.1: Testing
        if DEBUG:
            print(root['collisions'])

        # Task 3.2: Testing
        if DEBUG:
            for collision in root['collisions']:
                print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node())
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #             create a copy of any objects that your child nodes might inherit

        while self.open_list and timer.time() - self.start_time < self.time_max:
            # 1. Get next node
            next_node = self.pop_node()
            # 2. If this node has no collision, return solution
            if not next_node['collisions']:
                self.print_results(next_node)
                return next_node['paths']
            # 3. Otherwise, choose the first collision
            # collision = random.choice(next_node['collisions'])
            collision = next_node['collisions'][0]
            # 4.3 Adjusting the High-Level Search
            # constraints = disjoint_splitting(collision) if disjoint else standard_splitting(collision)
            constraints = standard_splitting(collision)
            for c in constraints:
                skip_node = False
                nodes_constraint = {'cost': 0,
                     'constraints': [*next_node['constraints'], c],  # all constraints for next_node
                     'paths': next_node['paths'].copy(),
                     'collisions': []
                     }
                agent = c['agent']
                path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent],
                              agent, nodes_constraint['constraints'])
                if path:
                    nodes_constraint['paths'][agent] = path
                    if c['positive']:
                        renew_agents = paths_violate_constraint(c, nodes_constraint['paths'])
                        for r_agent in renew_agents:
                            new_c = c.copy()
                            new_c['agent'] = r_agent
                            new_c['positive'] = False
                            nodes_constraint['constraints'].append(new_c)
                            r_path = a_star(self.my_map, self.starts[r_agent], self.goals[r_agent],
                                            self.heuristics[r_agent], r_agent,nodes_constraint['constraints'])
                            if r_path is None:
                                skip_node = True
                                break # at least one agent no solution
                            else:
                                nodes_constraint['paths'][r_agent] = r_path
                    if(not skip_node):
                        nodes_constraint['collisions'] = detect_collisions(nodes_constraint['paths'])
                        nodes_constraint['cost'] = get_sum_of_cost(nodes_constraint['paths'])
                        self.push_node(nodes_constraint)
                else:
                    raise BaseException('No solutions')
                    

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
