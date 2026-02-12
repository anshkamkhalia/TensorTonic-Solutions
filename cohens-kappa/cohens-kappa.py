import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    agreements = sum(r1 == r2 for r1, r2 in zip(rater1, rater2))

    p_e = 0
    p_o = agreements/len(rater1)
    uniques = set(rater1 + rater2) 
    for k in uniques:
        n_k1 = sum(r == k for r in rater1)
        n_k2 = sum(r == k for r in rater2)
        p_e += (n_k1 / len(rater1)) * (n_k2 / len(rater2))

    try:
        kappa = (p_o - p_e) / (1 - p_e)
    except:
        return 1.0
    return kappa