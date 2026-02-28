"""
Assignment 1 - Question 1: Search Strategies
Manuscript Sorting Problem (8-Puzzle Variant)

Author: AI Assignment Solution
Course: AI CSL7610

Problem Formulation (Section 1.1):
- State: Tuple of 9 elements representing the 3x3 grid (row-major order).
  'B' denotes the blank slot. e.g., (1,2,3,'B',4,6,7,5,8)
- Actions: Move blank Up, Down, Left, Right (swap blank with adjacent tile)
- Goal State: (1,2,3,4,5,6,7,8,'B')
- Path Cost: Each move costs 1 unit of System Energy (uniform cost)

Input: read from input.txt
Output: Success/Failure, Heuristic/Parameters, Path, States Explored, Time Taken
"""

import time
import heapq
import math
import random
import sys
from collections import deque

# 1.1 PROBLEM FORMULATION (used by ALL algorithms)

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 'B')
ROWS, COLS = 3, 3

def parse_state(text):
    """Parse a state string like '123;B46;758' into a tuple."""
    text = text.strip().replace(';', '').replace(' ', '')
    state = []
    for ch in text:
        if ch == 'B':
            state.append('B')
        else:
            state.append(int(ch))
    return tuple(state)

def read_input(filename="input.txt"):
    """Read start and goal states from input.txt."""
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    start = None
    goal = None
    for line in lines:
        if line.lower().startswith("start"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                start = parse_state(parts[1])
        elif line.lower().startswith("goal"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                goal = parse_state(parts[1])
    if start is None:
        start = parse_state(lines[0])
    if goal is None:
        goal = GOAL_STATE
    return start, goal

def find_blank(state):
    """Return index of the blank ('B') in the state tuple."""
    return state.index('B')

def get_neighbors(state):
    """
    Generate all valid successor states by moving the blank
    Up, Down, Left, Right. Returns list of (action, new_state).
    Each move costs 1 unit of System Energy.
    """
    idx = find_blank(state)
    row, col = idx // COLS, idx % COLS
    neighbors = []
    moves = [('Up', -1, 0), ('Down', 1, 0), ('Left', 0, -1), ('Right', 0, 1)]
    for action, dr, dc in moves:
        nr, nc = row + dr, col + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            new_idx = nr * COLS + nc
            lst = list(state)
            lst[idx], lst[new_idx] = lst[new_idx], lst[idx]
            neighbors.append((action, tuple(lst)))
    return neighbors

def is_goal(state, goal=GOAL_STATE):
    """Check if current state matches the goal state."""
    return state == goal

def reconstruct_path(came_from, current):
    """Reconstruct the path from start to current state."""
    path = []
    while current in came_from:
        prev, action = came_from[current]
        path.append((action, current))
        current = prev
    path.reverse()
    return path

# HEURISTICS (used by Informed, IDA*, and Adversarial Search)

def h1_misplaced(state, goal=GOAL_STATE):
    """h1: Number of misplaced manuscripts (not counting blank)."""
    count = 0
    for i in range(len(state)):
        if state[i] != 'B' and state[i] != goal[i]:
            count += 1
    return count

def h2_manhattan(state, goal=GOAL_STATE):
    """h2: Total Manhattan Distance of all manuscripts from goal positions."""
    distance = 0
    for i in range(len(state)):
        if state[i] != 'B':
            val = state[i]
            goal_idx = goal.index(val)
            r1, c1 = i // COLS, i % COLS
            r2, c2 = goal_idx // COLS, goal_idx % COLS
            distance += abs(r1 - r2) + abs(c1 - c2)
    return distance

# A. UNINFORMED SEARCH

def bfs(start, goal=GOAL_STATE):
    """
    Breadth-First Search: Explores all states at depth d before depth d+1.
    Guarantees minimum number of moves (optimal for uniform cost).
    Uses a FIFO queue and visited set to prevent revisiting states.
    """
    t0 = time.time()
    if is_goal(start, goal):
        return {"success": True, "path": [], "states_explored": 1,
                "time": time.time() - t0, "algorithm": "BFS"}

    frontier = deque()
    frontier.append(start)
    visited = {start}
    came_from = {}
    states_explored = 0

    while frontier:
        current = frontier.popleft()
        states_explored += 1

        for action, neighbor in get_neighbors(current):
            if neighbor not in visited:
                came_from[neighbor] = (current, action)
                if is_goal(neighbor, goal):
                    path = reconstruct_path(came_from, neighbor)
                    return {"success": True, "path": path,
                            "states_explored": states_explored,
                            "time": time.time() - t0, "algorithm": "BFS"}
                visited.add(neighbor)
                frontier.append(neighbor)

    return {"success": False, "path": [], "states_explored": states_explored,
            "time": time.time() - t0, "algorithm": "BFS"}


def dfs(start, goal=GOAL_STATE, max_depth=50):
    """
    Depth-First Search: Explores as deep as possible before backtracking.
    Uses a depth limit (max_depth) to prevent infinite paths.
    Maintains a visited set to avoid revisiting explored states including
    root, parent, and all previously explored states.
    """
    t0 = time.time()
    if is_goal(start, goal):
        return {"success": True, "path": [], "states_explored": 1,
                "time": time.time() - t0, "algorithm": "DFS"}

    stack = [(start, 0)]  # (state, depth)
    visited = {start}
    came_from = {}
    states_explored = 0

    while stack:
        current, depth = stack.pop()
        states_explored += 1

        if is_goal(current, goal):
            path = reconstruct_path(came_from, current)
            return {"success": True, "path": path,
                    "states_explored": states_explored,
                    "time": time.time() - t0, "algorithm": "DFS"}

        if depth >= max_depth:
            continue

        for action, neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = (current, action)
                stack.append((neighbor, depth + 1))

    return {"success": False, "path": [], "states_explored": states_explored,
            "time": time.time() - t0, "algorithm": "DFS"}

# B. INFORMED SEARCH

def greedy_best_first(start, goal=GOAL_STATE, heuristic=None):
    """
    Greedy Best-First Search: Expands the node with the lowest h(n).
    Uses only heuristic value (ignores path cost g(n)).
    Prioritizes nodes that appear closest to the goal.
    Not guaranteed to find optimal solution.
    """
    if heuristic is None:
        heuristic = h2_manhattan
    t0 = time.time()
    h_name = heuristic.__name__

    frontier = []
    counter = 0
    h_val = heuristic(start, goal)
    heapq.heappush(frontier, (h_val, counter, start))
    visited = {start}
    came_from = {}
    states_explored = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        states_explored += 1

        if is_goal(current, goal):
            path = reconstruct_path(came_from, current)
            return {"success": True, "path": path,
                    "states_explored": states_explored,
                    "time": time.time() - t0,
                    "algorithm": f"Greedy Best-First ({h_name})"}

        for action, neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = (current, action)
                counter += 1
                h_val = heuristic(neighbor, goal)
                heapq.heappush(frontier, (h_val, counter, neighbor))

    return {"success": False, "path": [], "states_explored": states_explored,
            "time": time.time() - t0,
            "algorithm": f"Greedy Best-First ({h_name})"}


def a_star(start, goal=GOAL_STATE, heuristic=None):
    """
    A* Search: Expands node with lowest f(n) = g(n) + h(n).
    g(n) = path cost from start to n (each move costs 1).
    h(n) = heuristic estimate to goal.
    Optimal and complete when heuristic is admissible.
    """
    if heuristic is None:
        heuristic = h2_manhattan
    t0 = time.time()
    h_name = heuristic.__name__

    frontier = []
    counter = 0
    g_score = {start: 0}
    h_val = heuristic(start, goal)
    f_val = h_val
    heapq.heappush(frontier, (f_val, counter, start))
    visited = set()
    came_from = {}
    states_explored = 0

    while frontier:
        f, _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)
        states_explored += 1

        if is_goal(current, goal):
            path = reconstruct_path(came_from, current)
            return {"success": True, "path": path,
                    "states_explored": states_explored,
                    "time": time.time() - t0,
                    "algorithm": f"A* ({h_name})"}

        for action, neighbor in get_neighbors(current):
            if neighbor not in visited:
                new_g = g_score[current] + 1  # path cost = 1 per move
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    h_val = heuristic(neighbor, goal)
                    f_val = new_g + h_val
                    came_from[neighbor] = (current, action)
                    counter += 1
                    heapq.heappush(frontier, (f_val, counter, neighbor))

    return {"success": False, "path": [], "states_explored": states_explored,
            "time": time.time() - t0,
            "algorithm": f"A* ({h_name})"}

# C. MEMORY-BOUNDED & LOCAL SEARCH

def ida_star(start, goal=GOAL_STATE, heuristic=None):
    """
    Iterative Deepening A* (IDA*): Memory-efficient version of A*.
    Uses depth-first search with an f-cost threshold that increases
    each iteration. Uses the same heuristics as A*.
    Space complexity: O(d) where d is solution depth.
    """
    if heuristic is None:
        heuristic = h2_manhattan
    t0 = time.time()
    h_name = heuristic.__name__

    threshold = heuristic(start, goal)
    path_stack = [start]
    total_explored = 0
    iterations = 0

    def search(g, threshold):
        nonlocal total_explored
        node = path_stack[-1]
        total_explored += 1
        f = g + heuristic(node, goal)

        if f > threshold:
            return f
        if is_goal(node, goal):
            return "FOUND"

        min_t = float('inf')
        for action, neighbor in get_neighbors(node):
            if neighbor not in set(path_stack):  # avoid cycles on current path
                path_stack.append(neighbor)
                t = search(g + 1, threshold)
                if t == "FOUND":
                    return "FOUND"
                if t < min_t:
                    min_t = t
                path_stack.pop()
        return min_t

    while True:
        iterations += 1
        result = search(0, threshold)
        if result == "FOUND":
            # Build path with actions
            final_path = []
            for i in range(1, len(path_stack)):
                prev = path_stack[i - 1]
                curr = path_stack[i]
                # Determine the action
                prev_blank = find_blank(prev)
                curr_blank = find_blank(curr)
                pr, pc = prev_blank // COLS, prev_blank % COLS
                cr, cc = curr_blank // COLS, curr_blank % COLS
                dr, dc = cr - pr, cc - pc
                action_map = {(-1, 0): 'Up', (1, 0): 'Down', (0, -1): 'Left', (0, 1): 'Right'}
                action = action_map.get((dr, dc), 'Move')
                final_path.append((action, curr))
            return {"success": True, "path": final_path,
                    "states_explored": total_explored,
                    "iterations": iterations,
                    "time": time.time() - t0,
                    "algorithm": f"IDA* ({h_name})"}
        if result == float('inf'):
            return {"success": False, "path": [],
                    "states_explored": total_explored,
                    "iterations": iterations,
                    "time": time.time() - t0,
                    "algorithm": f"IDA* ({h_name})"}
        threshold = result


def simulated_annealing(start, goal=GOAL_STATE, T_init=1000.0, cooling_rate=0.995,
                         T_min=0.001, max_iterations=100000):
    """
    Simulated Annealing: Local search that can escape local optima.
    Energy function: Manhattan Distance to goal.
    Cooling Schedule: T = T_init * (cooling_rate ^ iteration)
    Acceptance probability: P = e^(-deltaE / T) when deltaE > 0.
    Occasionally accepts worse moves to escape local maxima.
    """
    t0 = time.time()
    current = start
    current_energy = h2_manhattan(current, goal)
    best = current
    best_energy = current_energy
    T = T_init
    states_explored = 0
    path = []

    for i in range(max_iterations):
        if current_energy == 0:
            break

        neighbors = get_neighbors(current)
        if not neighbors:
            break

        action, neighbor = random.choice(neighbors)
        neighbor_energy = h2_manhattan(neighbor, goal)
        delta_e = neighbor_energy - current_energy
        states_explored += 1

        # Accept better moves always; worse moves with probability P = e^(-deltaE/T)
        if delta_e <= 0 or random.random() < math.exp(-delta_e / T):
            current = neighbor
            current_energy = neighbor_energy
            path.append((action, neighbor))

            if current_energy < best_energy:
                best = current
                best_energy = current_energy

        # Cooling schedule
        T = T * cooling_rate
        if T < T_min:
            T = T_min

    success = is_goal(current, goal)
    return {
        "success": success,
        "path": path if success else path[:50],  # truncate path for display
        "states_explored": states_explored,
        "time": time.time() - t0,
        "final_energy": current_energy,
        "algorithm": f"Simulated Annealing (T0={T_init}, cool={cooling_rate})",
        "parameters": f"T_init={T_init}, cooling_rate={cooling_rate}, T_min={T_min}"
    }

# D. ADVERSARIAL SEARCH EXTENSION

def utility(state, goal=GOAL_STATE):
    """
    Utility function for adversarial search.
    Returns negative Manhattan Distance (higher is better for MAX).
    Terminal: utility = 0 when goal is reached (MAX wins),
              very negative when disorder is maximum (MIN wins).
    """
    return -h2_manhattan(state, goal)

def minimax(state, depth, is_max_player, goal=GOAL_STATE, stats=None):
    """
    Minimax Search for adversarial manuscript sorting.
    MAX player (robotic sorter): tries to reach goal (maximize utility).
    MIN player (system glitch): tries to increase disorder (minimize utility).
    Terminal: goal reached or depth limit.
    """
    if stats is None:
        stats = {"nodes_evaluated": 0}
    stats["nodes_evaluated"] += 1

    if is_goal(state, goal) or depth == 0:
        return utility(state, goal), None

    neighbors = get_neighbors(state)
    if not neighbors:
        return utility(state, goal), None

    if is_max_player:
        best_val = float('-inf')
        best_action = None
        for action, neighbor in neighbors:
            val, _ = minimax(neighbor, depth - 1, False, goal, stats)
            if val > best_val:
                best_val = val
                best_action = (action, neighbor)
        return best_val, best_action
    else:
        best_val = float('inf')
        best_action = None
        for action, neighbor in neighbors:
            val, _ = minimax(neighbor, depth - 1, True, goal, stats)
            if val < best_val:
                best_val = val
                best_action = (action, neighbor)
        return best_val, best_action


def alpha_beta(state, depth, alpha, beta, is_max_player, goal=GOAL_STATE, stats=None):
    """
    Alpha-Beta Pruning: Optimized Minimax that prunes branches
    that cannot affect the final decision.
    Alpha: best value MAX can guarantee.
    Beta: best value MIN can guarantee.
    Prune when alpha >= beta.
    """
    if stats is None:
        stats = {"nodes_evaluated": 0, "pruned": 0}
    stats["nodes_evaluated"] += 1

    if is_goal(state, goal) or depth == 0:
        return utility(state, goal), None

    neighbors = get_neighbors(state)
    if not neighbors:
        return utility(state, goal), None

    if is_max_player:
        best_val = float('-inf')
        best_action = None
        for action, neighbor in neighbors:
            val, _ = alpha_beta(neighbor, depth - 1, alpha, beta, False, goal, stats)
            if val > best_val:
                best_val = val
                best_action = (action, neighbor)
            alpha = max(alpha, best_val)
            if alpha >= beta:
                stats["pruned"] += 1
                break  # Beta cutoff
        return best_val, best_action
    else:
        best_val = float('inf')
        best_action = None
        for action, neighbor in neighbors:
            val, _ = alpha_beta(neighbor, depth - 1, alpha, beta, True, goal, stats)
            if val < best_val:
                best_val = val
                best_action = (action, neighbor)
            beta = min(beta, best_val)
            if alpha >= beta:
                stats["pruned"] += 1
                break  # Alpha cutoff
        return best_val, best_action


def run_adversarial(start, goal=GOAL_STATE, depth=4):
    """Run both Minimax and Alpha-Beta and compare."""
    t0 = time.time()
    mm_stats = {"nodes_evaluated": 0}
    mm_val, mm_action = minimax(start, depth, True, goal, mm_stats)
    mm_time = time.time() - t0

    t1 = time.time()
    ab_stats = {"nodes_evaluated": 0, "pruned": 0}
    ab_val, ab_action = alpha_beta(start, depth, float('-inf'), float('inf'), True, goal, ab_stats)
    ab_time = time.time() - t1

    return {
        "minimax": {
            "value": mm_val, "action": mm_action,
            "nodes_evaluated": mm_stats["nodes_evaluated"],
            "time": mm_time
        },
        "alpha_beta": {
            "value": ab_val, "action": ab_action,
            "nodes_evaluated": ab_stats["nodes_evaluated"],
            "pruned": ab_stats["pruned"],
            "time": ab_time
        }
    }

# DISPLAY HELPERS

def state_to_grid(state):
    """Convert state tuple to 3x3 grid string."""
    lines = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            row.append(str(state[r * COLS + c]))
        lines.append(' '.join(row))
    return '\n'.join(lines)

def print_result(result):
    """Print algorithm results in the required output format."""
    print(f"\n{'='*50}")
    print(f"Algorithm: {result['algorithm']}")
    print(f"{'='*50}")
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILURE'}")
    if 'parameters' in result:
        print(f"Parameters: {result['parameters']}")
    print(f"Total States Explored: {result['states_explored']}")
    print(f"Total Time Taken: {result['time']:.6f} seconds")
    if result['success'] and result['path']:
        print(f"Path Length (Cost): {len(result['path'])} moves")
        print(f"Solution Path:")
        for i, (action, state) in enumerate(result['path']):
            print(f"  Step {i+1}: {action}")
    elif result['success']:
        print("Start state is already the goal state!")
    if 'iterations' in result:
        print(f"IDA* Iterations: {result['iterations']}")
    if 'final_energy' in result:
        print(f"Final Energy (Manhattan Distance): {result['final_energy']}")
    print()


# MAIN EXECUTION

if __name__ == "__main__":
    # Read input
    try:
        start, goal = read_input("input.txt")
    except FileNotFoundError:
        print("input.txt not found. Using default start state.")
        start = parse_state("123;B46;758")
        goal = GOAL_STATE

    print(f"Start State:\n{state_to_grid(start)}\n")
    print(f"Goal State:\n{state_to_grid(goal)}\n")

    # A. Uninformed Search
    print("\n" + "="*60)
    print("SECTION A: UNINFORMED SEARCH")
    print("="*60)

    result_bfs = bfs(start, goal)
    print_result(result_bfs)

    result_dfs = dfs(start, goal)
    print_result(result_dfs)

    # B. Informed Search
    print("\n" + "="*60)
    print("SECTION B: INFORMED SEARCH")
    print("="*60)

    result_greedy_h1 = greedy_best_first(start, goal, h1_misplaced)
    print_result(result_greedy_h1)

    result_greedy_h2 = greedy_best_first(start, goal, h2_manhattan)
    print_result(result_greedy_h2)

    result_astar_h1 = a_star(start, goal, h1_misplaced)
    print_result(result_astar_h1)

    result_astar_h2 = a_star(start, goal, h2_manhattan)
    print_result(result_astar_h2)

    # C. Memory-Bounded & Local Search
    print("\n" + "="*60)
    print("SECTION C: MEMORY-BOUNDED & LOCAL SEARCH")
    print("="*60)

    result_ida_h1 = ida_star(start, goal, h1_misplaced)
    print_result(result_ida_h1)

    result_ida_h2 = ida_star(start, goal, h2_manhattan)
    print_result(result_ida_h2)

    random.seed(42)
    result_sa = simulated_annealing(start, goal)
    print_result(result_sa)

    
    # D. Adversarial Search
    
    print("\n" + "="*60)
    print("SECTION D: ADVERSARIAL SEARCH")
    print("="*60)

    adv = run_adversarial(start, goal, depth=4)
    print(f"\nMinimax (depth=4):")
    print(f"  Utility Value: {adv['minimax']['value']}")
    print(f"  Best Action: {adv['minimax']['action']}")
    print(f"  Nodes Evaluated: {adv['minimax']['nodes_evaluated']}")
    print(f"  Time: {adv['minimax']['time']:.6f}s")
    print(f"\nAlpha-Beta Pruning (depth=4):")
    print(f"  Utility Value: {adv['alpha_beta']['value']}")
    print(f"  Best Action: {adv['alpha_beta']['action']}")
    print(f"  Nodes Evaluated: {adv['alpha_beta']['nodes_evaluated']}")
    print(f"  Branches Pruned: {adv['alpha_beta']['pruned']}")
    print(f"  Time: {adv['alpha_beta']['time']:.6f}s")
    print(f"\nEfficiency Improvement:")
    if adv['minimax']['nodes_evaluated'] > 0:
        reduction = (1 - adv['alpha_beta']['nodes_evaluated'] / adv['minimax']['nodes_evaluated']) * 100
        print(f"  Node Reduction: {reduction:.1f}%")
        if adv['minimax']['time'] > 0:
            speedup = adv['minimax']['time'] / adv['alpha_beta']['time']
            print(f"  Speedup: {speedup:.2f}x")
