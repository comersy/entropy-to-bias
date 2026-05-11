import math

def entropy(distribution: dict) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    """
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0)


def kl_divergence(p: dict, q: dict, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence from p to q.
    Not symmetric. epsilon avoids division by zero.
    """
    all_keys = set(p.keys()) | set(q.keys())
    return sum(
        p.get(k, epsilon) * math.log2(p.get(k, epsilon) / max(q.get(k, 0), epsilon))
        for k in all_keys
        if p.get(k, 0) > 0
    )



def hellinger(p: dict, q: dict) -> float:
    """
    Compute Hellinger distance between p and q.
    Symmetric, bounded in [0, 1].
    """
    all_keys = set(p.keys()) | set(q.keys())
    return math.sqrt(
        sum((math.sqrt(p.get(k, 0)) - math.sqrt(q.get(k, 0)))**2 for k in all_keys)
    ) / math.sqrt(2)


def bhattacharyya(p: dict, q: dict, epsilon: float = 1e-10) -> float:
    """
    Compute Bhattacharyya distance between p and q.
    Symmetric.
    """
    all_keys = set(p.keys()) | set(q.keys())
    coefficient = sum(
        math.sqrt(p.get(k, epsilon) * q.get(k, epsilon))
        for k in all_keys
    )
    return -math.log(coefficient)