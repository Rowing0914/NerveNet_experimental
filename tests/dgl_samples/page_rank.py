"""
Description:
    In this script, we will look at the different levels of message passing API
    with running example of PageRank on a small graph. The message passing and
    feature transformations are all User-Defined Functions(UDFs).

Goal:
    to implement PageRank using DGL message passing interface.

The PageRank algorithm:
    In each iteration of PageRank, every node(web-page) first scatters its PageRank Value(PV) uniformly
    to its downstream nodes. The new PV of each node is computed by aggregating the received PV from
    its neighbours, which is then adjusted by the damping factor:

    - Formula
        PV(u) = (1-d)/N + d * sum_{v in N(u)} PV(v)/D(v)
        where:
            N: the number of nodes in the graph
            D(v): out-degree of a node(e.g., number of out-going links)
            N(u): neighbouring nodes
"""

import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl

N = 100
DAMP = 0.85
K = 10
g = nx.nx.erdos_renyi_graph(N, 0.1)
g = dgl.DGLGraph(g)
# nx.draw(g.to_networkx(), node_size=50, node_color=[[.5, .5, .5,]])
# plt.show()

"""
According to the algorithm, PageRank consists of two phases in a typical scatter-gather pattern. We first initialise
the PV of each node to 1/N and store each node's out-degree as a node feature.
"""

g.ndata["pv"] = torch.ones(N)/N
g.ndata["deg"] = g.out_degrees(g.nodes()).float()

def pagerank_message_func(edges):
    """ divides every node's PV by its out-degree and passes the result as message to its neighbours """
    return {"pv": edges.src["pv"]/edges.src["deg"]}

def pagerank_reduce_func(nodes):
    """ removes and aggregates the messages from its `mailbox` and computes new PV """
    msgs = torch.sum(nodes.mailbox["pv"], dim=1)
    pv = (1 - DAMP) / N + DAMP * msgs
    return {"pv" : pv}

g.register_message_func(pagerank_message_func)
g.register_reduce_func(pagerank_reduce_func)

def pagerank_naive(g):
    # Phase #1: send out messages along all edges
    for u, v in zip(*g.edges()):
        g.send((u, v))

    # Phase #2: receive messages to compute new PageRank values
    for v in g.nodes():
        g.recv(v)

def pagerank_batch(g):
    g.send(g.edges())
    g.recv(g.nodes())

def pagerank_level2(g):
    g.update_all()

def pagerank_buildin(g):
    g.ndata["pv"] = g.ndata["pv"]/g.ndata["deg"]
    g.update_all(message_func=dgl.function.copy_src(src="pv", out="m"),
                 reduce_func=dgl.function.sum(msg="m", out="m_sum"))
    g.ndata["pv"] = (1 - DAMP) / N + DAMP * g.ndata["m_sum"]

for k in range(K):
    pagerank_buildin(g)
print(g.ndata["pv"])