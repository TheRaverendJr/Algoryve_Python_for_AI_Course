import itertools

# Priors: prevalence in the population
priors = {
    "Flu": 0.05,
    "Cold": 0.20
}

# Likelihoods: P(symptom | disease)
likelihoods = {
    "Flu":    {"fever": 0.90, "cough": 0.80},
    "Cold":   {"fever": 0.30, "cough": 0.70}
}


# Observed patient symptoms
observed = {"fever": True, "cough": True}

# Compute unnormalized posteriors P(disease) * Π P(symptom | disease)
unnormalized = {}
for disease in priors:
    prob = priors[disease]
    for symptom, present in observed.items():
        p_sym = likelihoods[disease][symptom]
        prob *= p_sym if present else (1 - p_sym)
    unnormalized[disease] = prob

# Normalize to sum to 1
total = sum(unnormalized.values())
posteriors = {d: p / total for d, p in unnormalized.items()}

print("Posterior probabilities given symptoms:")
for disease, p in posteriors.items():
    print(f"  {disease}: {p:.4f}")

