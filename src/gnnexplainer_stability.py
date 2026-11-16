"""
Investigating the Stability of GNNExplainer Under Graph Perturbations
Explainable AI (DS5622) - Track 1 experimental investigation.

Pipeline
  1. Build the synthetic BA-House benchmark (PyG ExplainerDataset: BAGraph + HouseMotif,
     ground-truth motif masks available).
  2. Train + validate a 3-layer GCN (kept FIXED for all later experiments).
  3. For a sample of correctly-classified motif nodes run
       (a) GNNExplainer over several random seeds              -> seed stability
       (b) GNNExplainer on prediction-preserving perturbed graphs -> perturbation stability
       (c) random-edge baseline with the same budget k           -> baseline
  4. Metrics: Jaccard stability, fidelity under edge removal, sufficiency,
     sparsity (fixed budget k) and (bonus) precision against the ground-truth house motif.
  5. Stats (paired Wilcoxon + bootstrap CIs), CSVs, plots, results.md.

Explanations are handled as UNDIRECTED edges: the GCN uses both directions, so the two
directed mask values are summed and the top-k undirected edges inside the target node's
3-hop computation subgraph are selected.

Usage:  python src/gnnexplainer_stability.py            (full run, from the repo root)
        python src/gnnexplainer_stability.py --quick    (small smoke test)
"""
import argparse
import itertools
import json
import os
import random
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
import torch_geometric.transforms as T

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- config
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="results")
args = ap.parse_args()

CFG = dict(
    data_seed=0, model_seed=0,
    num_base_nodes=300, ba_edges=5, num_motifs=80,
    hidden=32, layers=3, train_epochs=400, lr=0.01,
    n_targets=30,                 # target nodes explained
    n_seeds=10,                   # explainer seeds per node
    explainer_epochs=200,
    k=6,                          # primary fixed edge budget (a house has 6 undirected edges)
    k_sweep=[3, 6, 9, 12],
    perturb_fracs=[0.05, 0.10],   # fraction of undirected subgraph edges modified
    perturb_types=["add", "drop"],
    n_perturb_draws=5,            # prediction-preserving perturbed graphs per (node, type, frac)
    max_perturb_tries=30,
    n_random_runs=10,             # random-baseline repeats
)
if args.quick:
    CFG.update(n_targets=6, n_seeds=4, explainer_epochs=60, n_perturb_draws=2,
               train_epochs=200, num_motifs=40, num_base_nodes=150, n_random_runs=4)

os.makedirs(args.out, exist_ok=True)
dev = torch.device("cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# --------------------------------------------------------------------------- data
set_seed(CFG["data_seed"])
dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=CFG["num_base_nodes"], num_edges=CFG["ba_edges"]),
    motif_generator=HouseMotif(),
    num_motifs=CFG["num_motifs"],
    transform=T.Constant(),       # constant node features -> structure is all that matters
)
data = dataset[0].to(dev)
N = data.num_nodes
n_classes = int(data.y.max()) + 1
print(f"Graph: {N} nodes, {data.edge_index.size(1)} directed edges, {n_classes} classes, "
      f"{int((data.y > 0).sum())} motif nodes")

perm = torch.randperm(N, generator=torch.Generator().manual_seed(CFG["data_seed"]))
n_tr, n_va = int(.6 * N), int(.2 * N)
train_idx, val_idx, test_idx = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


# --------------------------------------------------------------------------- model
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid, out_dim, layers):
        super().__init__()
        dims = [in_dim] + [hid] * (layers - 1)
        self.convs = torch.nn.ModuleList(GCNConv(a, b) for a, b in zip(dims, dims[1:]))
        self.convs.append(GCNConv(hid, hid))
        self.lin = torch.nn.Linear(hid, out_dim)

    def forward(self, x, edge_index):
        for c in self.convs:
            x = F.relu(c(x, edge_index))
        return self.lin(x)


set_seed(CFG["model_seed"])
model = GCN(data.num_features, CFG["hidden"], n_classes, CFG["layers"]).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=5e-4)
best_val, best_state = -1, None
for ep in range(CFG["train_epochs"]):
    model.train(); opt.zero_grad()
    out = model(data.x, data.edge_index)
    F.cross_entropy(out[train_idx], data.y[train_idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(1)
    va = (pred[val_idx] == data.y[val_idx]).float().mean().item()
    if va >= best_val:
        best_val, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
with torch.no_grad():
    pred = model(data.x, data.edge_index).argmax(1)
acc = {n: (pred[i] == data.y[i]).float().mean().item()
       for n, i in [("train", train_idx), ("val", val_idx), ("test", test_idx)]}
print("GCN accuracy:", acc)

# --------------------------------------------------------------------------- helpers
LAYERS = CFG["layers"]


def und(a, b):
    return (a, b) if a < b else (b, a)


def undirected_edges(edge_index):
    ei = edge_index.t().tolist()
    return sorted({und(a, b) for a, b in ei if a != b})


def to_directed_index(und_edges, device=dev):
    if len(und_edges) == 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    t = torch.tensor(und_edges, dtype=torch.long, device=device).t()
    return torch.cat([t, t.flip(0)], dim=1)


def subgraph_und_edges(node, edge_index):
    """Undirected edges inside the 3-hop computation subgraph of `node`."""
    subset, _, _, _ = k_hop_subgraph(node, LAYERS, edge_index, num_nodes=N)
    s = set(subset.tolist())
    return [e for e in undirected_edges(edge_index) if e[0] in s and e[1] in s], s


@torch.no_grad()
def prob(node, edge_index, cls):
    return F.softmax(model(data.x, edge_index)[node], -1)[cls].item()


@torch.no_grad()
def predict(node, edge_index):
    return model(data.x, edge_index)[node].argmax().item()


explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=CFG["explainer_epochs"], lr=0.01),
    explanation_type="model",
    node_mask_type=None,
    edge_mask_type="object",
    model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
)


def gnnexplainer_scores(node, edge_index, seed):
    """Return {undirected edge: importance} over the node's computation subgraph."""
    set_seed(seed)
    expl = explainer(data.x, edge_index, index=node)
    m = expl.edge_mask.detach().cpu()
    cand, _ = subgraph_und_edges(node, edge_index)
    sc = {e: 0.0 for e in cand}
    for (a, b), v in zip(edge_index.t().tolist(), m.tolist()):
        e = und(a, b)
        if e in sc:
            sc[e] += v
    return sc


def top_k(scores, k, rng=None):
    # stable deterministic tie-break by edge id
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return frozenset(e for e, _ in items[:k])


def random_k(cand, k, rng):
    idx = rng.choice(len(cand), size=min(k, len(cand)), replace=False)
    return frozenset(cand[i] for i in idx)


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def remove_edges(edge_index, edges):
    """Remove undirected `edges` (both directions) from edge_index."""
    rem = set(edges) | {(b, a) for a, b in edges}
    keep = [i for i, (a, b) in enumerate(edge_index.t().tolist()) if (a, b) not in rem]
    return edge_index[:, keep]


def metrics(node, edge_index, cls, p0, expl):
    """fidelity (prob drop on removal), sufficiency (prediction kept), sufficiency prob."""
    fid = p0 - prob(node, remove_edges(edge_index, expl), cls)
    keep = to_directed_index(sorted(expl))
    suff_pred = predict(node, keep) == cls
    suff_prob = prob(node, keep, cls)
    return fid, float(suff_pred), suff_prob


def gt_precision(expl, gt_edges):
    return len(expl & gt_edges) / max(len(expl), 1)


# ground truth house edges (undirected) from dataset
gt_mask = data.edge_mask.bool()
GT_UND = {und(a, b) for (a, b), m in zip(data.edge_index.t().tolist(), gt_mask.tolist()) if m}


def perturb(node, edge_index, kind, frac, expl0, rng):
    """Random prediction-agnostic perturbation of the computation subgraph.
    add : insert round(frac*|E_sub|) new undirected edges between subgraph nodes.
    drop: delete round(frac*|E_sub|) edges, never the ones in the original explanation
          (conservative: favours stability, so any instability found is a lower bound)."""
    cand, nodes = subgraph_und_edges(node, edge_index)
    m = max(1, int(round(frac * len(cand))))
    if kind == "add":
        existing = set(undirected_edges(edge_index))
        nl = sorted(nodes)
        new = set()
        while len(new) < m:
            a, b = rng.choice(nl, 2, replace=False)
            e = und(int(a), int(b))
            if e not in existing and e not in new:
                new.add(e)
        ei = torch.cat([edge_index, to_directed_index(sorted(new))], dim=1)
    else:
        pool = [e for e in cand if e not in expl0]
        sel = [pool[i] for i in rng.choice(len(pool), size=min(m, len(pool)), replace=False)]
        ei = remove_edges(edge_index, sel)
    return ei, m


# --------------------------------------------------------------------------- targets
rng = np.random.RandomState(CFG["data_seed"])
motif_ok = [i for i in range(N) if data.y[i] > 0 and pred[i] == data.y[i]]
print(f"{len(motif_ok)} correctly classified motif nodes; sampling {CFG['n_targets']}")
targets = sorted(rng.choice(motif_ok, size=min(CFG["n_targets"], len(motif_ok)), replace=False).tolist())

K = CFG["k"]
seed_rows, pert_rows, node_rows = [], [], []
t0 = time.time()
for ti, node in enumerate(targets):
    cls = int(pred[node])
    p0 = prob(node, data.edge_index, cls)
    cand, _ = subgraph_und_edges(node, data.edge_index)
    gt_sub = GT_UND & set(cand)

    # ---- (a) seeds
    scores = {s: gnnexplainer_scores(node, data.edge_index, s) for s in range(CFG["n_seeds"])}
    expl = {s: top_k(sc, K) for s, sc in scores.items()}
    rand_expl = [random_k(cand, K, rng) for _ in range(CFG["n_random_runs"])]

    for s in range(CFG["n_seeds"]):
        fid, sp, spp = metrics(node, data.edge_index, cls, p0, expl[s])
        node_rows.append(dict(node=node, method="GNNExplainer", run=s, fidelity=fid, sufficiency=sp,
                              suff_prob=spp, sparsity=len(expl[s]), gt_precision=gt_precision(expl[s], GT_UND)))
    for r, e in enumerate(rand_expl):
        fid, sp, spp = metrics(node, data.edge_index, cls, p0, e)
        node_rows.append(dict(node=node, method="Random", run=r, fidelity=fid, sufficiency=sp,
                              suff_prob=spp, sparsity=len(e), gt_precision=gt_precision(e, GT_UND)))

    gj = [jaccard(expl[a], expl[b]) for a, b in itertools.combinations(range(CFG["n_seeds"]), 2)]
    rj = [jaccard(a, b) for a, b in itertools.combinations(rand_expl, 2)]
    # k-sweep seed stability
    ksweep = {}
    for kk in CFG["k_sweep"]:
        g = [top_k(scores[s], kk) for s in scores]
        ksweep[kk] = (np.mean([jaccard(a, b) for a, b in itertools.combinations(g, 2)]),
                      np.mean([jaccard(random_k(cand, kk, rng), random_k(cand, kk, rng)) for _ in range(30)]))
    seed_rows.append(dict(node=node, n_cand=len(cand), p_orig=p0,
                          jacc_gnnexp=np.mean(gj), jacc_random=np.mean(rj),
                          **{f"jacc_gnnexp_k{k}": v[0] for k, v in ksweep.items()},
                          **{f"jacc_random_k{k}": v[1] for k, v in ksweep.items()}))

    # ---- (b) perturbations (explainer seed fixed to 0 -> isolates the graph change)
    ref = expl[0]
    for kind, frac in itertools.product(CFG["perturb_types"], CFG["perturb_fracs"]):
        got = tries_total = 0
        for d in range(CFG["n_perturb_draws"]):
            ok = False
            for _ in range(CFG["max_perturb_tries"]):
                tries_total += 1
                ei_p, m = perturb(node, data.edge_index, kind, frac, ref, rng)
                if predict(node, ei_p) == cls:
                    ok = True; break
            if not ok:
                continue
            got += 1
            cand_p, _ = subgraph_und_edges(node, ei_p)
            sc_p = gnnexplainer_scores(node, ei_p, 0)
            e_p = top_k(sc_p, K)
            r_p = random_k(cand_p, K, rng)
            p0p = prob(node, ei_p, cls)
            fid, sp, spp = metrics(node, ei_p, cls, p0p, e_p)
            pert_rows.append(dict(node=node, kind=kind, frac=frac, n_changed=m, draw=d,
                                  jacc_gnnexp=jaccard(ref, e_p),
                                  jacc_random=jaccard(rand_expl[0], r_p),
                                  fidelity=fid, sufficiency=sp, p_perturbed=p0p))
        node_rows.append(dict(node=node, method=f"_accept_{kind}_{frac}", run=0,
                              fidelity=np.nan, sufficiency=got / CFG["n_perturb_draws"],
                              suff_prob=tries_total, sparsity=np.nan, gt_precision=np.nan))
    if (ti + 1) % 5 == 0 or ti == 0:
        print(f"  [{ti+1}/{len(targets)}] node {node}: seed-J={np.mean(gj):.3f} "
              f"rand-J={np.mean(rj):.3f}  ({time.time()-t0:.0f}s)", flush=True)

seed_df, pert_df, run_df = pd.DataFrame(seed_rows), pd.DataFrame(pert_rows), pd.DataFrame(node_rows)
acc_df = run_df[run_df.method.str.startswith("_accept")]
run_df = run_df[~run_df.method.str.startswith("_accept")]
seed_df.to_csv(f"{args.out}/seed_stability_per_node.csv", index=False)
pert_df.to_csv(f"{args.out}/perturbation_stability.csv", index=False)
run_df.to_csv(f"{args.out}/fidelity_sufficiency_runs.csv", index=False)


# --------------------------------------------------------------------------- statistics
def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float); r = np.random.RandomState(seed)
    b = [r.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def wilcoxon(a, b):
    try:
        return stats.wilcoxon(a, b).pvalue
    except ValueError:
        return float("nan")


summary = {}
lines = []
lines.append("# Results: Stability of GNNExplainer under graph perturbations\n")
lines.append(f"*Dataset*: BA-House synthetic graph ({N} nodes, {n_classes} classes), "
             f"GCN accuracy train/val/test = {acc['train']:.3f}/{acc['val']:.3f}/{acc['test']:.3f}. "
             f"{len(targets)} target nodes, budget k = {K} undirected edges, "
             f"{CFG['n_seeds']} explainer seeds, {CFG['explainer_epochs']} epochs.\n")

# seed stability
g, r = seed_df.jacc_gnnexp, seed_df.jacc_random
lines.append("## 1. Seed stability (graph fixed, different explainer initialisations)\n")
lines.append("| Method | mean pairwise Jaccard | 95% bootstrap CI |\n|---|---|---|")
for n_, v in [("GNNExplainer", g), ("Random-edge baseline", r)]:
    lo, hi = boot_ci(v)
    lines.append(f"| {n_} | {v.mean():.3f} | [{lo:.3f}, {hi:.3f}] |")
p = wilcoxon(g, r)
lines.append(f"\nPaired Wilcoxon (GNNExplainer vs Random, per node): p = {p:.2e}.\n")
summary["seed_jaccard_gnnexp"] = g.mean(); summary["seed_jaccard_random"] = r.mean()
frac_lt = float((g < 1).mean())
lines.append(f"Fraction of target nodes whose explanation is *not* identical across all seeds "
             f"(mean J < 1): {frac_lt:.2f}.\n")

# H0 test: is the mean Jaccard above a 'consistently high' threshold (0.8)?
THR = 0.8
t_stat = stats.wilcoxon(g - THR, alternative="less").pvalue if (g != THR).any() else float("nan")
lines.append(f"**H0 check** (overlap consistently high, threshold J >= {THR}): one-sided Wilcoxon "
             f"for median J < {THR}, p = {t_stat:.3g}.\n")

# k sweep
lines.append("### Effect of budget k (seed stability)\n")
lines.append("| k | GNNExplainer J | Random J |\n|---|---|---|")
for kk in CFG["k_sweep"]:
    lines.append(f"| {kk} | {seed_df[f'jacc_gnnexp_k{kk}'].mean():.3f} | {seed_df[f'jacc_random_k{kk}'].mean():.3f} |")

# perturbation
lines.append("\n## 2. Perturbation stability (prediction-preserving perturbations, explainer seed fixed)\n")
lines.append("| Perturbation | magnitude | GNNExplainer J | Random J | p (Wilcoxon, per node) | acceptance rate |\n|---|---|---|---|---|---|")
pert_summary = []
for (kind, frac), d in pert_df.groupby(["kind", "frac"]):
    pn = d.groupby("node")[["jacc_gnnexp", "jacc_random"]].mean()
    pv = wilcoxon(pn.jacc_gnnexp, pn.jacc_random)
    a = acc_df[acc_df.method == f"_accept_{kind}_{frac}"].sufficiency.mean()
    lines.append(f"| {kind} | {int(frac*100)}% of subgraph edges | {pn.jacc_gnnexp.mean():.3f} | "
                 f"{pn.jacc_random.mean():.3f} | {pv:.2e} | {a:.2f} |")
    pert_summary.append(dict(kind=kind, frac=frac, gnnexp=pn.jacc_gnnexp.mean(), random=pn.jacc_random.mean()))

# fidelity / sufficiency
lines.append("\n## 3. Fidelity, sufficiency, sparsity and ground-truth agreement (clean graph)\n")
lines.append("| Method | Fidelity (prob drop on removal) | Sufficiency (pred. kept) | P(class) when retaining only explanation | Sparsity (#edges) | Precision vs. house motif |\n|---|---|---|---|---|---|")
fs = run_df.groupby(["method", "node"]).mean(numeric_only=True).groupby("method").mean()
for m_ in ["GNNExplainer", "Random"]:
    row = fs.loc[m_]
    lines.append(f"| {m_} | {row.fidelity:.3f} | {row.sufficiency:.3f} | {row.suff_prob:.3f} | {row.sparsity:.1f} | {row.gt_precision:.3f} |")
pn = run_df.groupby(["method", "node"]).mean(numeric_only=True).reset_index()
for col in ["fidelity", "sufficiency", "gt_precision"]:
    a = pn[pn.method == "GNNExplainer"].set_index("node")[col]
    b = pn[pn.method == "Random"].set_index("node")[col]
    lines.append(f"\n- Wilcoxon {col}: GNNExplainer vs Random p = {wilcoxon(a, b.loc[a.index]):.2e}")
# fidelity on perturbed graphs
pf = pert_df.groupby(["kind", "frac"])[["fidelity", "sufficiency"]].mean()
lines.append("\n### Fidelity / sufficiency of GNNExplainer on perturbed graphs\n")
lines.append(pf.round(3).to_markdown())

# stability vs. fidelity relation
node_fid = run_df[run_df.method == "GNNExplainer"].groupby("node").fidelity.mean()
sj = seed_df.set_index("node").jacc_gnnexp
rho, prho = stats.spearmanr(node_fid.loc[sj.index], sj)
lines.append(f"\nSpearman correlation between per-node fidelity and seed stability: rho = {rho:.3f} (p = {prho:.3g}).\n")

# failure cases
worst = seed_df.sort_values("jacc_gnnexp").head(5)[["node", "n_cand", "jacc_gnnexp", "jacc_random"]]
lines.append("## 4. Least stable target nodes (failure cases)\n")
lines.append(worst.round(3).to_markdown(index=False))
lines.append("\n")
open(f"{args.out}/results.md", "w").write("\n".join(lines))

# --------------------------------------------------------------------------- plots
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
C = {"GNNExplainer": "#2a6fdb", "Random": "#9aa0a6"}

# Fig 1: stability across conditions
conds, gv, rv = ["Seeds"], [seed_df.jacc_gnnexp.values], [seed_df.jacc_random.values]
for (kind, frac), d in pert_df.groupby(["kind", "frac"]):
    pn_ = d.groupby("node")[["jacc_gnnexp", "jacc_random"]].mean()
    conds.append(f"{kind} {int(frac*100)}%"); gv.append(pn_.jacc_gnnexp.values); rv.append(pn_.jacc_random.values)
fig, ax = plt.subplots(figsize=(9, 4.5))
pos = np.arange(len(conds))
b1 = ax.boxplot(gv, positions=pos - .18, widths=.3, patch_artist=True, showfliers=False)
b2 = ax.boxplot(rv, positions=pos + .18, widths=.3, patch_artist=True, showfliers=False)
for b, c in [(b1, C["GNNExplainer"]), (b2, C["Random"])]:
    for patch in b["boxes"]:
        patch.set_facecolor(c); patch.set_alpha(.8)
    for med in b["medians"]:
        med.set_color("black")
ax.axhline(THR, ls="--", c="red", lw=1); ax.text(len(conds) - .5, THR + .01, "H0 threshold (0.8)", color="red", ha="right", fontsize=8)
ax.set_xticks(pos); ax.set_xticklabels(conds); ax.set_ylabel(f"Jaccard similarity (top-{K} edges)")
ax.set_ylim(0, 1.05); ax.set_title("Explanation stability across conditions")
ax.legend([b1["boxes"][0], b2["boxes"][0]], ["GNNExplainer", "Random baseline"], loc="upper right", frameon=False)
fig.tight_layout(); fig.savefig(f"{args.out}/fig1_stability_conditions.png"); plt.close(fig)

# Fig 2: fidelity & sufficiency & GT precision
fig, axs = plt.subplots(1, 3, figsize=(11, 4))
for ax, col, ttl in zip(axs, ["fidelity", "sufficiency", "gt_precision"],
                        ["Fidelity (prob. drop on removal)", "Sufficiency (prediction kept)", "Precision vs. house motif"]):
    vals = [pn[pn.method == m_][col].values for m_ in ["GNNExplainer", "Random"]]
    means = [v.mean() for v in vals]; cis = [boot_ci(v) for v in vals]
    ax.bar([0, 1], means, color=[C["GNNExplainer"], C["Random"]],
           yerr=[[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]], capsize=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["GNNExplainer", "Random"]); ax.set_title(ttl, fontsize=10)
fig.suptitle(f"Explanation quality at k = {K} (mean over nodes, 95% CI)"); fig.tight_layout()
fig.savefig(f"{args.out}/fig2_fidelity_sufficiency.png"); plt.close(fig)

# Fig 3: k sweep
fig, ax = plt.subplots(figsize=(5.5, 4))
ks = CFG["k_sweep"]
ax.plot(ks, [seed_df[f"jacc_gnnexp_k{k}"].mean() for k in ks], "o-", c=C["GNNExplainer"], label="GNNExplainer")
ax.plot(ks, [seed_df[f"jacc_random_k{k}"].mean() for k in ks], "s-", c=C["Random"], label="Random")
ax.set_xlabel("edge budget k"); ax.set_ylabel("mean pairwise Jaccard (seeds)"); ax.set_ylim(0, 1.05)
ax.set_title("Seed stability vs. budget"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{args.out}/fig3_k_sweep.png"); plt.close(fig)

# Fig 4: stability vs fidelity per node
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(node_fid.loc[sj.index], sj, c=C["GNNExplainer"])
ax.set_xlabel("mean fidelity of GNNExplainer"); ax.set_ylabel("seed stability (Jaccard)")
ax.set_title(f"Per-node stability vs. fidelity (rho={rho:.2f})")
fig.tight_layout(); fig.savefig(f"{args.out}/fig4_stability_vs_fidelity.png"); plt.close(fig)

# Fig 5: edge selection frequency on the least stable node
import networkx as nx
node = int(worst.iloc[0].node)
cand, _ = subgraph_und_edges(node, data.edge_index)
cnt = {e: 0 for e in cand}
for s in range(CFG["n_seeds"]):
    for e in top_k(gnnexplainer_scores(node, data.edge_index, s), K):
        cnt[e] += 1
G = nx.Graph(); G.add_edges_from(cand)
pos_ = nx.spring_layout(G, seed=1)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
nx.draw_networkx_edges(G, pos_, edgelist=cand, edge_color="#cccccc", width=.8, ax=ax)
sel = [e for e in cand if cnt[e] > 0]
nx.draw_networkx_edges(G, pos_, edgelist=sel, edge_color=[cnt[e] for e in sel], edge_cmap=plt.cm.viridis,
                       edge_vmin=0, edge_vmax=CFG["n_seeds"], width=3, ax=ax)
nx.draw_networkx_edges(G, pos_, edgelist=[e for e in cand if e in GT_UND], edge_color="red", style="dotted", width=1.2, ax=ax)
nx.draw_networkx_nodes(G, pos_, node_size=25, node_color="#444", ax=ax)
nx.draw_networkx_nodes(G, pos_, nodelist=[node], node_size=120, node_color="orange", ax=ax)
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, CFG["n_seeds"]))
fig.colorbar(sm, ax=ax, label=f"# seeds selecting edge (of {CFG['n_seeds']})")
ax.set_title(f"Node {node}: edge-selection frequency (red dotted = ground-truth house)"); ax.axis("off")
fig.tight_layout(); fig.savefig(f"{args.out}/fig5_example_node.png"); plt.close(fig)

json.dump(dict(config=CFG, accuracy=acc, summary={k: float(v) for k, v in summary.items()}),
          open(f"{args.out}/run_summary.json", "w"), indent=2)
print("\n" + open(f"{args.out}/results.md").read())
print(f"Done in {time.time()-t0:.0f}s. Outputs in ./{args.out}/")
