"""Dataset figures for the presentation (same generator + seed as the experiment)."""
import random, numpy as np, torch, networkx as nx, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
import torch_geometric.transforms as T
random.seed(0); np.random.seed(0); torch.manual_seed(0)
ds = ExplainerDataset(BAGraph(300,5), HouseMotif(), 80, transform=T.Constant())
d = ds[0]; y = d.y.numpy()
G = nx.Graph(); G.add_nodes_from(range(d.num_nodes))
gt = set()
for (a,b),m in zip(d.edge_index.t().tolist(), d.edge_mask.tolist()):
    G.add_edge(a,b)
    if m: gt.add((min(a,b),max(a,b)))
col = {0:"#b8c0cc",1:"#f4a261",2:"#e63946",3:"#2a9d8f"}
names = {0:"0: BA base node",1:"1: house – middle",2:"2: house – bottom",3:"3: house – roof"}
plt.rcParams["font.family"]="DejaVu Sans"
# full graph
pos = nx.spring_layout(G, seed=3, k=0.12, iterations=200)
fig, ax = plt.subplots(figsize=(8,6.2))
base = [e for e in G.edges if (min(e),max(e)) not in gt]
nx.draw_networkx_edges(G,pos,edgelist=base,edge_color="#d5d9df",width=.4,ax=ax)
nx.draw_networkx_edges(G,pos,edgelist=list(gt),edge_color="#e63946",width=1.6,ax=ax)
nx.draw_networkx_nodes(G,pos,node_color=[col[c] for c in y],node_size=[14 if c==0 else 30 for c in y],linewidths=0,ax=ax)
for c in range(4): ax.scatter([],[],c=col[c],label=names[c],s=40)
ax.legend(frameon=False,loc="lower left",fontsize=9); ax.axis("off")
fig.tight_layout(); fig.savefig("figures/dataset_full.png",dpi=200,transparent=False,facecolor="white"); plt.close(fig)
# single house + attachment + immediate BA neighbours
house = [n for n in range(d.num_nodes) if y[n]>0]
n0 = house[0]; grp = [n for n in house if abs(n-n0)<5 and y[n]>0][:5]
grp = sorted(set(m for m in house if nx.has_path(G.subgraph(set(house)&set(G)) ,n0,m)) ) if False else None
# house nodes of the motif containing n0 = connected component in GT edge graph
H = nx.Graph(list(gt)); comp = nx.node_connected_component(H,n0)
ring = set(comp)
for n in comp: ring |= set(G.neighbors(n))
S = G.subgraph(ring)
p2 = nx.spring_layout(S, seed=5, k=.9)
fig, ax = plt.subplots(figsize=(6,5))
nx.draw_networkx_edges(S,p2,edge_color="#c9ced6",width=1,ax=ax)
nx.draw_networkx_edges(S,p2,edgelist=[e for e in S.edges if (min(e),max(e)) in gt],edge_color="#e63946",width=3,ax=ax)
nx.draw_networkx_nodes(S,p2,node_color=[col[y[n]] for n in S],node_size=[260 if y[n] else 90 for n in S],ax=ax)
ax.axis("off"); fig.tight_layout(); fig.savefig("figures/dataset_motif.png",dpi=200,facecolor="white"); plt.close(fig)
# class counts
print({c:int((y==c).sum()) for c in range(4)}, d.num_nodes, d.edge_index.size(1), len(gt))
