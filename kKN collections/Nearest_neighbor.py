import numpy as np
def find_nearest_neighbors(p,points,k=5):
    """Find the k neareest neighbors of point p and return their indices"""
    distances=np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i]=distance(p,points[i])
    ind=np.argsort(distances)
    return ind[:k]