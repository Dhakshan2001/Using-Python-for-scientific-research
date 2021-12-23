import networkx as nx 
import matplotlib.pyplot as plt

#Basics

G=nx.Graph()
G.add_node(1)
G.add_nodes_from([2,3])
G.add_nodes_from(['u','v'])
G.nodes()
G.add_edge(1,2)
G.add_edge('u','v') 
G.add_edges_from([(1,3),(1,4),(1,5),(1,6)])
G.remove_node(2)
G.remove_nodes_from(['u','v'])
G.remove_edge(1,3)
G.remove_edges_from([(1,2),('u','v')])
G.number_of_nodes()
G.number_of_edges()

#Graph visualization

G=nx.karate_club_graph()
nx.draw(G, with_labels=True, node_color="lightblue",edge_color="black")
plt.savefig("karate_graph.pdf")
G.degree()
G.degree(33)

#Random graphs

from scipy.stats import bernoulli
N=20
p=0.2

def er_graph(N,p):
    """Generate an ER graph"""
    G=nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p=p):
                G.add_edge(node1,node2)
    return G

nx.draw(er_graph(50,0.08), node_size=40,node_color="gray")
plt.savefig("er1.pdf")

def plot_degree_distribution(G):
    degree_sequence = [d for n,d in G.degree()]
    plt.hist(degree_sequence,histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree distribution")
    
G1=er_graph(500,0.08)
plot_degree_distribution(G1)
G2=er_graph(500,0.08)
plot_degree_distribution(G2)
G3=er_graph(500,0.08)
plot_degree_distribution(G3)
plt.savefig("hist_3.pdf")

#Discriptive statistics of Empirical Social Networks:

import numpy as np
A1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv",delimiter=",")
A2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv",delimiter=",")
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basic_net_stats(G):
    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of nodes: %d" % G.number_of_edges())
    degree_sequence = [d for n, d in G.degree()]
    print("Average degree: %.2f" % np.mean(degree_sequence))
    
basic_net_stats(G1)
basic_net_stats(G2)

plot_degree_distribution(G1)
plot_degree_distribution(G2)
plt.savefig("village_hist.pdf")


#Finding the largest connected components

gen = nx.connected_components(G1)
g = gen.__next__()

def connected_component_subgraphs(G):
    return [G.subgraph(c) for c in nx.connected_components(G)]

G1_LCC = max(connected_component_subgraphs(G1),key=len)
G2_LCC = max(connected_component_subgraphs(G2),key=len)

G1_LCC.number_of_nodes()/G1.number_of_nodes()
G2_LCC.number_of_nodes()/G2.number_of_nodes()

plt.figure()
nx.draw(G1_LCC, node_color="blue", edge_color="black", node_size=20)
plt.savefig("village1.pdf")
plt.figure()
nx.draw(G2_LCC, node_color="green", edge_color="black", node_size=20)
plt.savefig("village2.pdf")




