
from abc import ABC, abstractmethod

class Expr(ABC):
    """Abstract base for any logical expression."""
    @abstractmethod
    def evaluate(self, model: dict) -> bool:
        """Return truth value under the given symbol→bool assignment."""
        pass

class Symbol(Expr):
    """An atomic proposition, e.g. 'fever' or 'thirst'."""
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, model: dict) -> bool:
        return model.get(self.name, False)

    def __repr__(self):
        return self.name

class Not(Expr):
    """Negation: ¬P."""
    def __init__(self, operand: Expr):
        self.operand = operand

    def evaluate(self, model: dict) -> bool:
        return not self.operand.evaluate(model)

    def __repr__(self):
        return f"¬{self.operand}"

class And(Expr):
    """Conjunction: P ∧ Q ∧ ..."""
    def __init__(self, *operands: Expr):
        self.operands = operands

    def evaluate(self, model: dict) -> bool:
        return all(op.evaluate(model) for op in self.operands)

    def __repr__(self):
        return "(" + " ∧ ".join(map(str, self.operands)) + ")"

class Or(Expr):
    """Inclusive disjunction: P ∨ Q ∨ ..."""
    def __init__(self, *operands: Expr):
        self.operands = operands

    def evaluate(self, model: dict) -> bool:
        return any(op.evaluate(model) for op in self.operands)

    def __repr__(self):
        return "(" + " ∨ ".join(map(str, self.operands)) + ")"

class Implication(Expr):
    """Implication: P → Q (false only if P true and Q false)."""
    def __init__(self, antecedent: Expr, consequent: Expr):
        self.antecedent = antecedent
        self.consequent = consequent

    def evaluate(self, model: dict) -> bool:
        if not self.antecedent.evaluate(model):
            return True
        return self.consequent.evaluate(model)

    def __repr__(self):
        return f"({self.antecedent} → {self.consequent})"

class Biconditional(Expr):
    """Biconditional: P ↔ Q (true when P and Q match)."""
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def evaluate(self, model: dict) -> bool:
        return self.left.evaluate(model) == self.right.evaluate(model)

    def __repr__(self):
        return f"({self.left} ↔ {self.right})"