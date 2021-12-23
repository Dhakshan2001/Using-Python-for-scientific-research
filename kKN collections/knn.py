def knn_predict(p,points,outcomes,k=5):
    ind = find_nearest_neighbors(p,points,k)
    return majority_vote(outcomes[ind])