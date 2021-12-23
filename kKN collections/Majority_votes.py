import numpy as np
import random
def majority_vote(votes):
    """Return the most common element in votes"""           
    vote_counts={}
    for vote in votes:
        if vote in vote_counts:
           vote_counts[vote] += 1
        else:
           vote_counts[vote] = 1    
    winners = []
    max_count = max(vote_counts.values())
    for vote,count in vote_counts.items():
        if count ==max_count:
            winners.append(vote)
    return random.choice(winners)

import scipy.stats as ss
def majority_vote_short(votes):
    """Return the most common element in votes"""           
    mode, count = ss.mstats.mode(votes)
    return mode
