"""
A small Linear Programming model to allocate two treatment machines (X₁, X₂) 
to maximize total patients treated under cost and labor constraints:

    1. Decision variables: hours x₁, x₂ to run machines 1 and 2.

    2. Objective: maximize 10 x₁ + 12 x₂ patients treated (we convert to minimization by negating).

    3. Costs: running costs or labor use become constraints:

        3.1 Machine 1 uses 5 labor units/hour, machine 2 uses 2 units/hour; ≤ 20 units total.

        3.2 Budget constraint: 50 $ per hour for machine 1, 80 $ per hour for machine 2; ≤ 2000 $.

We use scipy.optimize.linprog.
"""


from scipy.optimize import linprog

# Maximize: 10 x1 + 12 x2  →  minimize: -10 x1 -12 x2
c = [-10, -12]

# Inequality constraints A_ub @ [x1, x2] ≤ b_ub
# 5 x1 + 2 x2 ≤ 20      (labor units)
# 50 x1 + 80 x2 ≤ 2000  (budget in $)
A_ub = [
    [5, 2],
    [50, 80]
]
b_ub = [20, 2000]

# Bounds: hours ≥ 0
bounds = [(0, None), (0, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if res.success:
    x1, x2 = res.x
    print(f"Run Machine1 for {x1:.2f} hrs, Machine2 for {x2:.2f} hrs")
    print(f"Total patients treated ≈ {10*x1 + 12*x2:.1f}")
else:
    print("No feasible solution found")
