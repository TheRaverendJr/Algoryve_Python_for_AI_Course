import numpy as np

states = ["Infected","NotInfected"]
obs_map = {"fever":0,"no_fever":1}
obs_seq = ["fever","fever","no_fever","fever","no_fever"]
obs = [obs_map[x] for x in obs_seq]

start = np.array([0.5,0.5])
Tmat  = np.array([[0.7,0.3],[0.1,0.9]])
Emat  = np.array([[0.9,0.1],[0.2,0.8]])

T = len(obs); N = len(states)
dp = np.zeros((N,T)); bp = np.zeros((N,T),dtype=int)

# init
dp[:,0] = start * Emat[:,obs[0]]

# recurse
for t in range(1,T):
    for j in range(N):
        prob = dp[:,t-1] * Tmat[:,j]
        bp[j,t] = np.argmax(prob)
        dp[j,t] = prob[bp[j,t]] * Emat[j,obs[t]]

# backtrack
path = np.zeros(T,dtype=int)
path[-1] = np.argmax(dp[:,T-1])
for t in range(T-2,-1,-1):
    path[t] = bp[path[t+1],t+1]

decoded = [states[i] for i in path]
print("Obs:", obs_seq)
print("States:", decoded)













































# #!/usr/bin/env python3
# # hidden_markov_model_pyro_final.py
# # Pyro‐based HMM with exact MAP (Viterbi) inference—tested 2025

# import torch
# import pyro
# import pyro.distributions as dist
# from pyro.infer import infer_discrete, config_enumerate

# # 1) Prepare observations
# obs_str = ["fever", "fever", "no_fever", "fever", "no_fever"]
# obs_map = {"fever": 0, "no_fever": 1}
# obs = torch.tensor([obs_map[x] for x in obs_str])

# # 2) Model definition with enumeration
# @config_enumerate(default="parallel")
# def model(observations):
#     T = observations.size(0)

#     # Initial state: 0=Infected, 1=NotInfected
#     init = torch.tensor([0.5, 0.5])
#     z = pyro.sample("z_0", dist.Categorical(init))
    
#     # Emission table: P(obs|state)
#     E = torch.tensor([[0.9, 0.1],    # if Infected
#                       [0.2, 0.8]])   # if NotInfected
#     pyro.sample("x_0", dist.Categorical(E[z]), obs=observations[0])

#     # Transition: P(z_t | z_{t-1})
#     Tmat = torch.tensor([[0.7, 0.3],
#                          [0.1, 0.9]])
#     for t in range(1, T):
#         z = pyro.sample(f"z_{t}", dist.Categorical(Tmat[z]))
#         pyro.sample(f"x_{t}", dist.Categorical(E[z]), obs=observations[t])

# # 3) Set up MAP inference
# map_infer = infer_discrete(
#     model,
#     first_available_dim=-1,  # enumeration dimension
#     temperature=0.0         # pure MAP
# )

# # 4) Run inference
# latent = map_infer(observations=obs)

# # 5) Decode the Viterbi path
# state_names = ["Infected", "NotInfected"]
# decoded = [state_names[int(latent[f"z_{t}"].item())] for t in range(len(obs))]

# # 6) Print
# print("Observations:", obs_str)
# print("Most likely states:", decoded)























































# import numpy as np
# from pomegranate import HiddenMarkovModel, DiscreteDistribution

# # Emission: sensor accuracy
# emissions = [
#     DiscreteDistribution({"fever": 0.9, "no_fever": 0.1}),   # Infected
#     DiscreteDistribution({"fever": 0.2, "no_fever": 0.8})    # Not Infected
# ]

# # Transition probabilities
# transition_matrix = np.array([
#     [0.7, 0.3],  # If infected today, 70% remain, 30% recover
#     [0.1, 0.9]   # If not infected, 10% get infected, 90% remain healthy
# ])

# # Initial state distribution
# start_probs = np.array([0.5, 0.5])

# # Build and bake HMM
# model = HiddenMarkovModel.from_matrix(
#     transition_matrix,
#     emissions,
#     start_probs,
#     state_names=["Infected", "NotInfected"]
# )
# model.bake()

# # Example observation sequence from sensor
# observations = ["fever", "fever", "no_fever", "fever", "no_fever"]

# # Run Viterbi to get most likely infection status sequence
# states = model.predict(observations)
# decoded = [model.states[s].name for s in states]

# print("Observations: ", observations)
# print("Inferred States:", decoded)
