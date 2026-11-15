
import numpy as np


def precision_at_k(recommended, relevant, k=10):
    recommended = recommended[:k]

    if not recommended:
        return 0.0

    hits = len(set(recommended) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    if not relevant:
        return 0.0

    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    relevant = set(relevant)

    dcg = sum(
        1 / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / idcg
