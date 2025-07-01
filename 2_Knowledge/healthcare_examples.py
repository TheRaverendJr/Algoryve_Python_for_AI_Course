
# Puts it all together with a patient-diagnosis KB.
# healthcare_example.py

from logic import Symbol, And, Implication, Or, Not
from kb import KnowledgeBase
from inference_rules import modus_ponens, and_elimination

# Define symptoms and diagnosis
fever         = Symbol("fever")       # Patient has fever
cough         = Symbol("cough")       # Patient has cough
shortness     = Symbol("shortness")   # Patient has shortness of breath
pneumonia     = Symbol("pneumonia")   # Patient has pneumonia
flu           = Symbol("flu")         # Patient has influenza

# Build KB:
#   1. fever ∧ cough → pneumonia
#   2. fever ∧ shortness → pneumonia
#   3. fever ∧ cough → flu
kb_sentences = And(
    Implication(And(fever, cough), pneumonia),
    Implication(And(fever, shortness), pneumonia),
    Implication(And(fever, cough), flu)
)
symbols = [fever, cough, shortness, pneumonia, flu]
kb = KnowledgeBase(kb_sentences, symbols)

# Patient case: fever and cough, but no shortness
patient_model = {
    "fever": True, 
    "cough": True, 
    "shortness": False
}

# 1) By model checking: does KB entail pneumonia?
print("Entails pneumonia:", kb.entails(pneumonia))  # True

# 2) By Modus Ponens: from (fever ∧ cough) → pneumonia and fact fever∧cough
rule = Implication(And(fever, cough), pneumonia)
fact = And(fever, cough)
print("Modus Ponens gives:", modus_ponens(rule, fact))  # pneumonia

# 3) By And-Elimination: get individual symptoms from conjunction
print("Symptoms present:", and_elimination(fact))      # [fever, cough]

