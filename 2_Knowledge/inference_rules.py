
# Illustrates Modus Ponens and And-Elimination for quick by-hand inference.
# inference_rules.py

from logic import Expr, Implication, And

def modus_ponens(rule: Implication, fact: Expr) -> Expr:
    """
    If rule is (P → Q) and fact matches P, returns Q.
    Otherwise returns None.
    """
    if rule.antecedent == fact:
        return rule.consequent
    return None

def and_elimination(conj: And):
    """
    Given (P ∧ Q ∧ ...), yields each P, Q, ... individually.
    """
    return list(conj.operands)