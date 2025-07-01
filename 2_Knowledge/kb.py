# Implements the Knowledge Base and a simple Model Checking entailment procedure.
# kb.py

from logic import Expr, Symbol
from typing import List

def check_all(kb: Expr, query: Expr, symbols: List[Symbol], model: dict) -> bool:
    """
    Recursively enumerate all models. If KB true in a model, query must be true.
    """
    if not symbols:
        # If KB holds, query must hold
        return (not kb.evaluate(model)) or query.evaluate(model)
    else:
        p = symbols[0]
        rest = symbols[1:]
        # assign True
        model[p.name] = True
        ok_true = check_all(kb, query, rest, model)
        # assign False
        model[p.name] = False
        ok_false = check_all(kb, query, rest, model)
        return ok_true and ok_false

class KnowledgeBase:
    def __init__(self, sentences: Expr, symbols: List[Symbol]):
        self.sentences = sentences
        self.symbols = symbols

    def entails(self, query: Expr) -> bool:
        """Return True when KB ⊨ query via model checking."""
        return check_all(self.sentences, query, self.symbols, {})