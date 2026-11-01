"""Small explicit sampling reference, including stable tie breaking."""
import math
import random


def probabilities(logits, temperature=1.0, top_k=0, top_p=1.0):
    if not logits or any(math.isnan(x) or x == math.inf for x in logits):
        raise ValueError("logits must contain at least one finite value, no NaN/+inf")
    if not any(math.isfinite(x) for x in logits):
        raise ValueError("all tokens are masked")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("invalid temperature")
    if type(top_k) is not int or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid top_k/top_p")
    ordered = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    if temperature == 0:
        return [float(i == ordered[0]) for i in range(len(logits))]
    keep = ordered[:top_k] if top_k else ordered
    maximum = max(logits[i] for i in keep)
    weights = {i: math.exp((logits[i] - maximum) / temperature) for i in keep}
    total = sum(weights.values())
    if top_p < 1:
        kept = []
        cumulative = 0.0
        for i in keep:
            kept.append(i)
            cumulative += weights[i] / total
            if cumulative >= top_p:
                break
        weights = {i: weights[i] for i in kept}
        total = sum(weights.values())
    return [weights.get(i, 0.0) / total for i in range(len(logits))]


def sample(logits, rng: random.Random, temperature=0.0, top_k=0, top_p=1.0):
    p = probabilities(logits, temperature, top_k, top_p)
    if temperature == 0:
        return p.index(1.0)
    threshold = rng.random()
    cumulative = 0.0
    last_nonzero = 0
    for i, probability in enumerate(p):
        if probability:
            last_nonzero = i
        cumulative += probability
        if threshold < cumulative:
            return i
    return last_nonzero
