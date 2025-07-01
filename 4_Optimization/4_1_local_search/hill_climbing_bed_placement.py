"""
We have four patient homes on a 1D corridor (positions 0, 2, 7, 10), 
and need to place two mobile‑care vans (beds) along that corridor to minimize 
the total walking distance from every home to its nearest van. This uses Steepest‑Ascent Hill Climbing:

1. State: positions of the two vans (e.g. [3, 8]).

2. Neighbors: move either van ±1 unit (staying within corridor 0–10).

3. Cost function: sum over homes of distance to closest van.

4. Repeatedly pick the neighbor with the lowest cost; stop when no neighbor improves.
"""


import random

# Patient homes at positions along a corridor
HOMES = [0, 2, 7, 10]
# 0 1 2 3  4 5 6 7 8 9 10
MIN_POS, MAX_POS = 0, 10

def cost(vans):
    """
    Sum of distances from each home to its nearest van.
    vans: list [pos1, pos2]
    """
    total = 0
    for h in HOMES:
        total += min(abs(h - vans[0]), abs(h - vans[1]))
    return total

def get_neighbors(vans):
    """
    Generate all neighbors by moving one van ±1 (within bounds).
    """
    neighbors = []
    for i in [0, 1]:
        for delta in (-1, 1):
            new = vans.copy()
            pos = new[i] + delta
            if MIN_POS <= pos <= MAX_POS:
                new[i] = pos
                neighbors.append(new)
    return neighbors

def hill_climb(initial):
    current = initial
    current_cost = cost(current)
    while True:
        # Find best neighbor
        neighs = get_neighbors(current)
        costs = [(cost(n), n) for n in neighs]
        best_cost, best_state = min(costs, key=lambda x: x[0])
        if best_cost >= current_cost:
            # No improvement
            return current, current_cost
        current, current_cost = best_state, best_cost

if __name__ == "__main__":
    # Start with random placement
    start = sorted(random.sample(range(MIN_POS, MAX_POS+1), 2))
    final_state, final_cost = hill_climb(start)
    print(f"Start vans at {start}, cost = {cost(start)}")
    print(f"Final vans at {final_state}, cost = {final_cost}")
