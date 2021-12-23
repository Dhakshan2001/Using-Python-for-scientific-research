import numpy as np
def distance(p1,p2):
    """Find the distance between points p1 & p2."""
    return np.sqrt(np.sum(np.power(p2-p1,2)))
