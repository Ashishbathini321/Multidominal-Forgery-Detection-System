import numpy as np

def normalize_prob(p):
    try:
        return float(np.clip(p, 0.0, 1.0))
    except:
        return 0.0

def final_decision(prob, low=0.30, high=0.65):
    """
    Returns:
    - "AI Generated"
    - "Real"
    - "Uncertain"
    """
    prob = normalize_prob(prob)

    if prob >= high:
        return "AI Generated"
    elif prob <= low:
        return "Real"
    else:
        return "Uncertain"

def confidence_score(prob):
    prob = normalize_prob(prob)
    return round(prob * 100, 2)