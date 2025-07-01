import random
from collections import Counter

# Transition probabilities: P(next_state | current_state)
transitions = {
    "Healthy": {"Healthy": 0.85, "Sick": 0.15},
    "Sick":    {"Healthy": 0.30, "Sick": 0.70}
}

def sample_next(state):
    r = random.random()
    cum = 0.0
    for next_state, p in transitions[state].items():
        cum += p
        if r < cum:
            return next_state
    return state  # fallback

def simulate_chain(days=7, start="Healthy"):
    state = start
    for _ in range(days):
        state = sample_next(state)
    return state

# Monte Carlo estimate
N = 100_000
results = Counter(simulate_chain() for _ in range(N))
prob_healthy = results["Healthy"] / N
print(f"Estimated P(patient healthy after 7 days) ≈ {prob_healthy:.4f}")
