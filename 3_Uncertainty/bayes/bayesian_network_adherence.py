
# Uses Pyro to model and infer P(Adherence | SideEffects)

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive

# -----------------------------------------------------------------------------
# 1. Define the generative model
# -----------------------------------------------------------------------------
def model():
    # SideEffects ∼ Categorical([0.3, 0.7])  
    #   categories: 0="yes", 1="no"
    se = pyro.sample("SideEffects",
                     dist.Categorical(torch.tensor([0.3, 0.7])))
    
    # BeliefInMedication | SideEffects
    # tensor[se] gives the 2-vector of P(high),P(low)
    belief_probs = torch.tensor([
        [0.2, 0.8],  # if se=0 (yes)
        [0.8, 0.2],  # if se=1 (no)
    ])
    belief = pyro.sample("Belief",
                         dist.Categorical(belief_probs[se]))
    
    # Adherence | SideEffects, BeliefInMedication
    # tensor[se, belief] gives the 2-vector of P(adhere),P(miss)
    adherence_probs = torch.tensor([
        [  # se = yes
            [0.5, 0.5],  # belief=high
            [0.2, 0.8],  # belief=low
        ],
        [  # se = no
            [0.9, 0.1],  # belief=high
            [0.6, 0.4],  # belief=low
        ],
    ])
    adherence = pyro.sample("Adherence",
                            dist.Categorical(adherence_probs[se, belief]))
    
    # Return all three so we can inspect if needed
    return {"SideEffects": se, "Belief": belief, "Adherence": adherence}

# -----------------------------------------------------------------------------
# 2. Exact joint probability P(no SE, high belief, adhere)
# -----------------------------------------------------------------------------
#   P(SideEffects=no)=0.7
#   P(Belief=high | no)=0.8
#   P(Adhere | no,high)=0.9
joint = 0.7 * 0.8 * 0.9
print(f"P(no SE, high belief, adhere) = {joint:.5f}")

# -----------------------------------------------------------------------------
# 3. Monte Carlo inference for P(Adherence | SideEffects = yes)
# -----------------------------------------------------------------------------
# Condition on SideEffects=0 ("yes")
conditioned_model = pyro.condition(model, data={"SideEffects": torch.tensor(0)})

# Draw many samples of the conditioned model
predictive = Predictive(conditioned_model, num_samples=20_000,
                        return_sites=["Adherence"])
samples = predictive()["Adherence"]  # Tensor of shape [20000]

# Empirical estimate of P(adhere) and P(miss)
counts = torch.bincount(samples, minlength=2).float()
probs = counts / counts.sum()

print("\nP(Adherence | SideEffects = yes):")
for label, p in zip(["adhere", "miss"], probs.tolist()):
    print(f"  {label}: {p:.4f}")




