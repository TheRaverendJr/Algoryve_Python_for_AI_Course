
# Basic CNF conversion and resolution-by-refutation to decide entailment.
# resolution.py

from logic import Expr, Or, Not, And, Implication
from itertools import combinations

def to_cnf(expr: Expr) -> Expr:
    """
    Stub: full CNF conversion would
    1. eliminate biconditionals
    2. eliminate implications (P → Q → ¬P ∨ Q)
    3. push negations inward (De Morgan)
    4. distribute ∨ over ∧
    Here we assume input is already in CNF form for simplicity.
    """
    return expr

def extract_clauses(cnf: Expr) -> list:
    """
    Given CNF = (C1) ∧ (C2) ∧ ..., return [C1, C2, ...],
    each Ci either an Or or a single literal.
    """
    if isinstance(cnf, And):
        return list(cnf.operands)
    else:
        return [cnf]

def resolve(c1: Expr, c2: Expr):
    """
    Given two clauses (each a disjunction of literals), return
    resolvents when a complementary pair exists, else [].
    """
    lits1 = c1.operands if isinstance(c1, Or) else [c1]
    lits2 = c2.operands if isinstance(c2, Or) else [c2]
    resolvents = []
    for l1, l2 in combinations(lits1 + lits2, 2):
        if isinstance(l1, Not) and l1.operand == l2 or isinstance(l2, Not) and l2.operand == l1:
            # drop the complementary pair
            new_lits = [l for l in lits1 + lits2 if l not in (l1, l2)]
            if not new_lits:
                resolvents.append(None)  # empty clause
            elif len(new_lits) == 1:
                resolvents.append(new_lits[0])
            else:
                resolvents.append(Or(*new_lits))
    return resolvents

def resolution_entails(kb: Expr, alpha: Expr) -> bool:
    """
    Check KB ⊨ alpha by refutation:
    add ¬alpha to KB, convert all to CNF, repeatedly resolve until
    empty clause or no new clauses.
    """
    clauses = extract_clauses(to_cnf(And(kb, Not(alpha))))
    new = set()
    while True:
        n = len(clauses)
        pairs = combinations(clauses, 2)
        for (ci, cj) in pairs:
            for resolvent in resolve(ci, cj):
                if resolvent is None:
                    return True
                new.add(str(resolvent))
        # no new clauses
        if len(new) == 0 or len(clauses) == n:
            return False
        # incorporate new
        for lit_str in new:
            # naive reparse omitted; in practice store Clause objects
            pass
