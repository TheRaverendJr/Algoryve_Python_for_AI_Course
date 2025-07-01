"""
We have 3 patients (A, B, C) needing to see 2 doctors (Dr. X, Dr. Y) across 3 slots (Mon, Tue, Wed). 
Each patient must be assigned one slot with one doctor, and no doctor may see two patients on the same day. 
This is a Constraint Satisfaction Problem solved via backtracking search 
with a simple MRV (Minimum Remaining Values) heuristic:

    1. Variables: patient–doctor pairs (e.g. “A‑X” means patient A with Dr X), domain = {Mon, Tue, Wed}.

    2. Constraints: for any given doctor, all their assigned days must be distinct.

    3. We pick the unassigned variable with fewest available days (MRV), try each day that doesn’t violate the doctor’s existing assignments, and recurse.
"""


# Patients and doctors
PATIENTS = ["A", "B", "C"]
DOCTORS  = ["X", "Y"]
DAYS     = ["Mon", "Tue", "Wed"]

# Domain: every possible (doctor, day) pair
DOMAIN = [(d, day) for d in DOCTORS for day in DAYS]

def select_unassigned(assignment, domains):
    """
    MRV: pick the patient with the fewest remaining options.
    """
    unassigned = [p for p in PATIENTS if p not in assignment]
    return min(unassigned, key=lambda p: len(domains[p]))

def consistent(patient, choice, assignment):
    """
    choice = (doctor, day).
    Ensure no other patient has chosen the same (doctor, day).
    """
    for other_patient, other_choice in assignment.items():
        if other_choice == choice:
            return False
    return True

def backtrack(assignment, domains):
    # If every patient has an appointment, we're done
    if len(assignment) == len(PATIENTS):
        return assignment

    # Pick the next patient to assign
    patient = select_unassigned(assignment, domains)

    for choice in domains[patient]:
        if consistent(patient, choice, assignment):
            assignment[patient] = choice
            result = backtrack(assignment, domains)
            if result:
                return result
            # Undo
            del assignment[patient]

    return None  # no valid choice for this branch

if __name__ == "__main__":
    # Each patient can choose any (doctor, day) initially
    domains = {p: DOMAIN.copy() for p in PATIENTS}

    solution = backtrack({}, domains)
    if solution:
        print("Schedule found:")
        for patient in PATIENTS:
            doctor, day = solution[patient]
            print(f"  Patient {patient}: Doctor {doctor} on {day}")
    else:
        print("No valid schedule.")














































# # Patients and doctors
# PATIENTS = ["A", "B", "C"]
# DOCTORS  = ["X", "Y"]
# DAYS     = ["Mon", "Tue", "Wed"]

# # Variables: e.g. "A-X" means patient A with doctor X
# VARIABLES = [f"{p}-{d}" for p in PATIENTS for d in DOCTORS]

# def select_unassigned(assignment, domains):
#     """
#     MRV heuristic: pick var with smallest domain.
#     """
#     unassigned = [v for v in VARIABLES if v not in assignment]
#     # sort by domain size
#     return min(unassigned, key=lambda v: len(domains[v]))

# def consistent(var, day, assignment):
#     """
#     Check: no other patient with same doctor on same day.
#     """
#     patient, doctor = var.split("-")
#     for other_var, other_day in assignment.items():
#         _, other_doc = other_var.split("-")
#         if other_doc == doctor and other_day == day:
#             return False
#     return True

# def backtrack(assignment, domains):
#     if len(assignment) == len(VARIABLES):
#         return assignment  # complete

#     var = select_unassigned(assignment, domains)
#     for day in domains[var]:
#         if consistent(var, day, assignment):
#             # Tentatively assign
#             assignment[var] = day
#             result = backtrack(assignment, domains)
#             if result:
#                 return result
#             # Backtrack
#             del assignment[var]

#     return None  # failure

# if __name__ == "__main__":
#     # All domains start as all days
#     domains = {v: DAYS.copy() for v in VARIABLES}
#     solution = backtrack({}, domains)
#     if solution:
#         print("Schedule found:")
#         for var in sorted(solution):
#             print(f"  {var}: {solution[var]}")
#     else:
#         print("No valid schedule.")
