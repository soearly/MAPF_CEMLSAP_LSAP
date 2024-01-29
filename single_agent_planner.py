import heapq

def move(loc, direction):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[direction][0], loc[1] + directions[direction][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for direction in range(4):
            child_loc = move(loc, direction)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Return a table that constains the list of constraints of
    # the given agent for each time step. The table can be used
    # for a more efficient constraint violation check in the 
    # is_constrained function.

    constraints_table = dict()
    for constraint in constraints:
        # Supporting positive constraints
        if (not 'positive' in constraint.keys()):
            constraint['positive'] = False
        if constraint['agent'] == agent:
            timestep = constraint['timestep']
            if timestep not in constraints_table:
                constraints_table[timestep] = [constraint]
            else:
                constraints_table[timestep].append(constraint)
    return constraints_table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def flat_constraints_list(list_of_constraints_arr):
    constraints = []
    for constr_arr in list_of_constraints_arr:
        for constraint in constr_arr:
            constraints.append(constraint)
    return constraints


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.
    if next_time in constraint_table:
        constraints = constraint_table[next_time]
        for constraint in constraints:
            if [next_loc] == constraint['loc'] or [curr_loc, next_loc] == constraint['loc']:
                return True
    else:
        constraints = [constraint for t, constraint in constraint_table.items() if t < next_time]
        constraints = flat_constraints_list(constraints)
        for constraint in constraints:
            if [next_loc] == constraint['loc'] and constraint['final']:
                return True
    return False


def is_future_constrained(goal_loc, timestep, constraint_table):
    # Check if there is a constraint in the future
    constraints = [constraint for t, constraint in constraint_table.items() if t > timestep]
    constraints = flat_constraints_list(constraints)
    for constraint in constraints:
        if [goal_loc] == constraint['loc']:
            return True
    return False


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    constraints_table = build_constraint_table(constraints, agent)
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'time': earliest_goal_timestep}
    push_node(open_list, root)
    closed_list[(start_loc, 0)] = root
    max_map_width = max([len(e) for e in my_map])

    while len(open_list) > 0:
        curr = pop_node(open_list)
        # 1.1 -> 1.3: Default goal
        # if curr['loc'] == goal_loc:
        #############################
        # 1.4: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc and not is_future_constrained(goal_loc, curr['time'], constraints_table):
            return get_path(curr)
        for dir in range(5):
            # directions 0-3 means the agent make a move, direction 4 means the agent stay in the same location
            if dir < 4:
                child_loc = move(curr['loc'], dir)
                # check if the child location is outsite the map or the agent go against an obstacle
                if child_loc[0] < 0 or child_loc[1] < 0 or child_loc[0] >= len(my_map) or child_loc[1] >= max_map_width or my_map[child_loc[0]][child_loc[1]]:
                    continue
                child = {'loc': child_loc,
                         'g_val': curr['g_val'] + 1,
                         'h_val': h_values[child_loc],
                         'parent': curr,
                         'time': curr['time'] + 1}
            else:
                # the agent remains still
                child = {'loc': curr['loc'],
                         'g_val': curr['g_val'] + 1,  # remaining in the same cell has a cost
                         'h_val': curr['h_val'],
                         'parent': curr,
                         'time': curr['time'] + 1}
            # check if the child violates the constraints
            if is_constrained(curr['loc'], child['loc'], child['time'], constraints_table):
                continue
            if (child['loc'], child['time']) in closed_list:
                existing_node = closed_list[(child['loc'], child['time'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['time'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['time'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions
