"""
We need to assign 5 nurses to 5 daily shift slots (morning, midday, evening, night, late‑night) 
so as to minimize total dissatisfaction. 
Each nurse has a preferred shift; being assigned to another shift costs 1 per difference in index. 
We use Simulated Annealing to escape local minima:

    1. State: a permutation assigning nurse→shift.

    2. Neighbor: swap two nurses’ shifts.

    3. Cost: sum of absolute differences between assigned shift index and preferred shift index.

    4. Temperature: starts high, slowly cools; worse swaps sometimes accepted with probability exp(-Δcost/T).
"""


import random
import math

# Nurses and their preferred shift indices (0–4)
PREFERENCES = {
    "Alice": 0,   # morning
    "Bob":   1,   # midday
    "Cathy": 2,   # evening
    "Dan":   3,   # night
    "Eve":   4    # late-night
}

SHIFTS = ["morning","midday","evening","night","late-night"]

def cost(state):
    """
    Total dissatisfaction: sum |assigned_index - preferred_index|.
    state: list of nurse names, index = shift assigned
    """
    total = 0
    for shift_idx, nurse in enumerate(state):
        total += abs(shift_idx - PREFERENCES[nurse])
    return total

def random_neighbor(state):
    """
    Swap two random positions (shifts).
    """
    a, b = random.sample(range(len(state)), 2)
    new = state.copy()
    new[a], new[b] = new[b], new[a]
    return new

def temperature(t, max_iter):
    """
    Linear cooling from 1.0 down to 0.01
    """
    return max(0.01, 1.0 - t / max_iter)

def simulated_annealing(max_iter=10_000):
    # Initial random schedule
    current = list(PREFERENCES.keys())
    random.shuffle(current)
    current_cost = cost(current)

    for t in range(1, max_iter+1):
        T = temperature(t, max_iter)
        candidate = random_neighbor(current)
        cand_cost = cost(candidate)
        delta = cand_cost - current_cost

        # Always accept better; sometimes accept worse
        if delta < 0 or random.random() < math.exp(-delta / T):
            current, current_cost = candidate, cand_cost

    return current, current_cost

if __name__ == "__main__":
    final_schedule, final_cost = simulated_annealing()
    print("Final assignment (shift → nurse):")
    for idx, nurse in enumerate(final_schedule):
        print(f"  {SHIFTS[idx]:12s}: {nurse}")
    print("Total dissatisfaction cost:", final_cost)
    print("Nurse preferences:")
    for nurse, pref in PREFERENCES.items():
        print(f"  {nurse:6s} prefers {SHIFTS[pref]:12s} shift")
