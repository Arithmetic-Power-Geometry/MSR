from __future__ import annotations
import argparse, json, time, random
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.datasets import load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon, spearmanr
import matplotlib.pyplot as plt

METHODS = ["FT", "L2-SP", "EWC", "Euclidean-MSR", "MSR"]
DATASETS = ["digits", "wine", "breast_cancer"]
DEFAULT_SEEDS = (7, 19, 31)
DEFAULT_EDIT_SIZES = (1, 5, 10)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, d: int, c: int, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, c))

    def forward(self, x):
        return self.net(x)


def load_data(name: str, seed: int):
    if name == "digits":
        ds, h = load_digits(), 64
    elif name == "wine":
        ds, h = load_wine(), 32
    elif name == "breast_cancer":
        ds, h = load_breast_cancer(), 32
    else:
        raise ValueError(f"Unknown dataset: {name}")
    X = ds.data.astype("float32")
    y = ds.target.astype("int64")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype("float32")
    Xte = scaler.transform(Xte).astype("float32")
    return (
        torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte), torch.tensor(yte),
        len(np.unique(y)), h, scaler,
    )


def train_base(model, X, y, epochs=180, lr=0.012):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
    return model


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def flat_params_live(model):
    return torch.cat([p.reshape(-1) for p in model.parameters()])


def set_from_flat(model, vec):
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[i:i+n].reshape_as(p))
            i += n


def grad_flat(model, scalar):
    gs = torch.autograd.grad(
        scalar, tuple(model.parameters()), retain_graph=True, allow_unused=False
    )
    return torch.cat([g.reshape(-1) for g in gs])


def preservation_sensitivity(model, X, batch=48, floor=1e-4):
    """Diagonal sensitivity: mean squared gradient of base decision margins."""
    n = min(batch, len(X))
    idx = torch.linspace(0, len(X)-1, n).long()
    acc = None
    for j in idx:
        logits = model(X[j:j+1])[0]
        order = torch.argsort(logits, descending=True)
        margin = logits[order[0]] - logits[order[1]]
        g = grad_flat(model, margin).detach().square()
        acc = g if acc is None else acc + g
    s = acc / max(n, 1) + floor
    return s / (s.mean() + 1e-12)


def projection_repair(model, Xe, yt, metric_diag, margin=0.75, rounds=10, ridge=1e-6):
    """Iterated local metric projection for target-vs-strongest-competitor constraints."""
    total_delta = torch.zeros_like(flat_params(model))
    rounds_used = 0
    for r in range(rounds):
        rounds_used = r + 1
        logits = model(Xe)
        rows, rhs = [], []
        for i in range(len(Xe)):
            li = logits[i]
            target = int(yt[i])
            mask = torch.ones_like(li, dtype=torch.bool)
            mask[target] = False
            other_idx = torch.arange(len(li))[mask]
            comp = int(other_idx[torch.argmax(li[mask])])
            m = li[target] - li[comp]
            need = float(margin - m.detach())
            if need <= 0:
                continue
            rows.append(grad_flat(model, m))
            rhs.append(need)
        if not rows:
            break
        A = torch.stack(rows)
        b = torch.tensor(rhs, dtype=A.dtype)
        Dinv = 1.0 / metric_diag
        AD = A * Dinv.unsqueeze(0)
        K = AD @ A.T + ridge * torch.eye(A.shape[0], dtype=A.dtype)
        alpha = torch.linalg.solve(K, b)
        delta = Dinv * (A.T @ alpha)
        set_from_flat(model, flat_params(model) + delta)
        total_delta += delta.detach()
    return model, rounds_used, total_delta


def finetune(model, Xe, yt, base, fisher=None, l2=0.0, ewc=0.0, steps=180, lr=0.012):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    base = base.detach()
    used = 0
    for step in range(steps):
        used = step + 1
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(Xe), yt)
        v = flat_params_live(model)
        d = v - base
        if l2:
            loss = loss + l2 * d.square().mean()
        if ewc and fisher is not None:
            loss = loss + ewc * (fisher * d.square()).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if (model(Xe).argmax(1) == yt).all() and step > 20:
                break
    return model, used


def clone_model(model, d, c, h):
    m = MLP(d, c, h)
    m.load_state_dict(deepcopy(model.state_dict()))
    return m


def target_margin(logits, targets):
    out = []
    for i in range(len(logits)):
        t = int(targets[i])
        mask = torch.ones_like(logits[i], dtype=torch.bool)
        mask[t] = False
        out.append(logits[i, t] - torch.max(logits[i][mask]))
    return torch.stack(out)


def evaluate(model, base_model, Xe, yt, Xp, yp, base_vec, common_metric):
    with torch.no_grad():
        le = model(Xe)
        lp = model(Xp)
        bp = base_model(Xp)
        edit = float((le.argmax(1) == yt).float().mean())
        min_margin = float(target_margin(le, yt).min())
        preserve = float((lp.argmax(1) == bp.argmax(1)).float().mean())
        acc = float((lp.argmax(1) == yp).float().mean())
        drift = float((lp - bp).square().mean())
        kl = float(nn.functional.kl_div(
            nn.functional.log_softmax(lp, dim=1),
            nn.functional.softmax(bp, dim=1), reduction="batchmean"
        ))
    d = flat_params(model) - base_vec
    rel = float(torch.linalg.norm(d) / (torch.linalg.norm(base_vec) + 1e-12))
    sc = float(torch.sqrt((common_metric * d.square()).sum() + 1e-18))
    changed = int(round((1-preserve) * len(Xp)))
    return dict(
        edit_success=edit, min_target_margin=min_margin, preservation=preserve,
        preserve_accuracy=acc, logit_drift=drift, preservation_kl=kl,
        relative_param_change=rel, structural_cost=sc, collateral_changes=changed,
    )


def _build_case(dataset: str, seed: int):
    seed_all(seed)
    Xtr, ytr, Xte, yte, c, h, scaler = load_data(dataset, seed)
    d = Xtr.shape[1]
    base = MLP(d, c, h)
    train_base(base, Xtr, ytr, epochs=180 if dataset == "digits" else 240)
    base.eval()
    base_vec = flat_params(base).clone()
    with torch.no_grad():
        pred = base(Xte).argmax(1)
        correct = torch.where(pred == yte)[0]
    g = torch.Generator().manual_seed(seed + 123)
    correct = correct[torch.randperm(len(correct), generator=g)]
    return Xtr, ytr, Xte, yte, c, h, d, base, base_vec, correct


def run_single_case(dataset="digits", seed=7, n_edits=5, beta=8.0, margin=0.75, methods: Iterable[str]=METHODS):
    Xtr, ytr, Xte, yte, c, h, d, base, base_vec, correct = _build_case(dataset, seed)
    if len(correct) < n_edits + 10:
        raise RuntimeError("Not enough correctly classified examples for requested edit size")
    eidx = correct[:n_edits]
    eset = set(eidx.tolist())
    pidx = torch.tensor([i for i in range(len(Xte)) if i not in eset])
    Xe, Xp, yp = Xte[eidx], Xte[pidx], yte[pidx]
    with torch.no_grad():
        logits = base(Xe)
        order = torch.argsort(logits, dim=1, descending=True)
        yt = order[:, 1]
    sens = preservation_sensitivity(base, Xp)
    common_metric = 1.0 + 50.0 * sens
    rows = []
    for method in methods:
        m = clone_model(base, d, c, h)
        t = time.perf_counter()
        iterations = 0
        if method == "FT":
            m, iterations = finetune(m, Xe, yt, base_vec, steps=160)
        elif method == "L2-SP":
            m, iterations = finetune(m, Xe, yt, base_vec, l2=50.0, steps=180)
        elif method == "EWC":
            m, iterations = finetune(m, Xe, yt, base_vec, fisher=sens, ewc=50.0, steps=180)
        elif method == "Euclidean-MSR":
            m, iterations, _ = projection_repair(m, Xe, yt, torch.ones_like(sens), margin=margin)
        elif method == "MSR":
            m, iterations, _ = projection_repair(m, Xe, yt, 1.0 + beta * sens, margin=margin)
        else:
            raise ValueError(method)
        elapsed = time.perf_counter() - t
        ev = evaluate(m, base, Xe, yt, Xp, yp, base_vec, common_metric)
        ev.update(dataset=dataset, seed=seed, n_edits=n_edits, method=method,
                  beta=beta if method == "MSR" else np.nan, margin=margin,
                  iterations=iterations, runtime_s=elapsed)
        rows.append(ev)
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot=2000, seed=2026):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def paired_statistics(df: pd.DataFrame):
    idx = ["dataset", "seed", "n_edits"]
    records = []
    for baseline in [m for m in METHODS if m != "MSR"]:
        a = df[df.method == "MSR"].set_index(idx)
        b = df[df.method == baseline].set_index(idx)
        common = a.index.intersection(b.index)
        for metric, direction in [("preservation", "greater"), ("structural_cost", "less"),
                                  ("logit_drift", "less"), ("preservation_kl", "less")]:
            diff = (a.loc[common, metric] - b.loc[common, metric]).to_numpy()
            try:
                p = float(wilcoxon(diff, alternative=direction, zero_method="wilcox").pvalue)
            except ValueError:
                p = 1.0
            lo, hi = bootstrap_ci(diff)
            records.append(dict(
                baseline=baseline, metric=metric, n=len(diff), mean_difference=float(diff.mean()),
                median_difference=float(np.median(diff)), ci95_low=lo, ci95_high=hi,
                wins=int((diff > 0).sum()) if direction == "greater" else int((diff < 0).sum()),
                ties=int((diff == 0).sum()), one_sided_p=p,
            ))
    return pd.DataFrame(records)


def run_beta_sweep(outdir: Path, seeds=(7,19,31), betas=(0,1,2,4,8,16,32,64), n_edits=5):
    rows = []
    for dataset in DATASETS:
        for seed in seeds:
            Xtr, ytr, Xte, yte, c, h, d, base, base_vec, correct = _build_case(dataset, seed)
            eidx = correct[:n_edits]
            eset = set(eidx.tolist())
            pidx = torch.tensor([i for i in range(len(Xte)) if i not in eset])
            Xe, Xp, yp = Xte[eidx], Xte[pidx], yte[pidx]
            with torch.no_grad():
                order = torch.argsort(base(Xe), dim=1, descending=True)
                yt = order[:, 1]
            sens = preservation_sensitivity(base, Xp)
            common_metric = 1 + 50 * sens
            for beta in betas:
                m = clone_model(base, d, c, h)
                m, rounds_used, _ = projection_repair(m, Xe, yt, 1 + beta*sens, rounds=10)
                ev = evaluate(m, base, Xe, yt, Xp, yp, base_vec, common_metric)
                ev.update(dataset=dataset, seed=seed, beta=beta, n_edits=n_edits, rounds=rounds_used)
                rows.append(ev)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "beta_sweep.csv", index=False)
    return df


def run_sequential(outdir: Path, seeds=(7,19,31), steps=10, beta=8.0):
    """Sequential single-example repairs, measuring cumulative cost and forgetting/locality."""
    rows = []
    for dataset in DATASETS:
        for seed in seeds:
            Xtr, ytr, Xte, yte, c, h, d, base, base_vec, correct = _build_case(dataset, seed)
            seq_idx = correct[:steps]
            pidx = torch.tensor([i for i in range(len(Xte)) if i not in set(seq_idx.tolist())])
            Xp, yp = Xte[pidx], yte[pidx]
            with torch.no_grad():
                order = torch.argsort(base(Xte[seq_idx]), dim=1, descending=True)
                targets = order[:, 1]
            sens = preservation_sensitivity(base, Xp)
            common_metric = 1 + 50*sens
            for method in ["FT", "Euclidean-MSR", "MSR"]:
                m = clone_model(base, d, c, h)
                cum_step_cost = 0.0
                edited_x, edited_y = [], []
                prev_vec = flat_params(m).clone()
                for t in range(steps):
                    xe = Xte[seq_idx[t:t+1]]
                    yt = targets[t:t+1]
                    edited_x.append(xe)
                    edited_y.append(yt)
                    if method == "FT":
                        m, _ = finetune(m, xe, yt, flat_params(m).clone(), steps=100, lr=.01)
                    elif method == "Euclidean-MSR":
                        m, _, _ = projection_repair(m, xe, yt, torch.ones_like(sens), rounds=8)
                    else:
                        m, _, _ = projection_repair(m, xe, yt, 1+beta*sens, rounds=8)
                    cur_vec = flat_params(m).clone()
                    step_delta = cur_vec - prev_vec
                    step_cost = float(torch.sqrt((common_metric*step_delta.square()).sum()+1e-18))
                    cum_step_cost += step_cost
                    prev_vec = cur_vec
                    Xed = torch.cat(edited_x, dim=0)
                    Yed = torch.cat(edited_y, dim=0)
                    with torch.no_grad():
                        retained_edits = float((m(Xed).argmax(1) == Yed).float().mean())
                        preservation = float((m(Xp).argmax(1) == base(Xp).argmax(1)).float().mean())
                        drift = float((m(Xp)-base(Xp)).square().mean())
                    total_delta = cur_vec-base_vec
                    total_struct = float(torch.sqrt((common_metric*total_delta.square()).sum()+1e-18))
                    rows.append(dict(dataset=dataset, seed=seed, method=method, step=t+1,
                                     retained_edits=retained_edits, preservation=preservation,
                                     logit_drift=drift, cumulative_step_cost=cum_step_cost,
                                     total_structural_cost=total_struct))
    df = pd.DataFrame(rows)
    df.to_csv(outdir/"sequential_results.csv", index=False)
    # Does cumulative repair cost track collateral change better than edit count?
    summary=[]
    for method in ["FT", "Euclidean-MSR", "MSR"]:
        q=df[df.method==method].copy()
        q["collateral"] = 1-q.preservation
        rho_cost,p_cost=spearmanr(q.cumulative_step_cost,q.collateral)
        rho_count,p_count=spearmanr(q.step,q.collateral)
        summary.append(dict(method=method,rho_cost=float(rho_cost),p_cost=float(p_cost),
                            rho_edit_count=float(rho_count),p_edit_count=float(p_count)))
    pd.DataFrame(summary).to_csv(outdir/"sequential_correlations.csv",index=False)
    return df


def save_plots(df: pd.DataFrame, beta_df: pd.DataFrame, seq_df: pd.DataFrame, outdir: Path):
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)
    # Main edit-count plots
    for metric, ylabel, fn in [
        ("preservation", "Prediction preservation", "preservation_vs_edits.pdf"),
        ("structural_cost", "Common structural repair cost", "structural_cost_vs_edits.pdf"),
        ("relative_param_change", "Relative parameter change", "parameter_change_vs_edits.pdf"),
        ("logit_drift", "Preservation logit MSE", "logit_drift_vs_edits.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(6.3,4.1))
        for method in METHODS:
            q = df[df.method == method].groupby("n_edits")[metric].agg(["mean", "std"])
            ax.errorbar(q.index, q["mean"], yerr=q["std"], marker="o", capsize=3, label=method)
        ax.set_xlabel("Number of simultaneous edits")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir/fn)
        plt.close(fig)
    # Tradeoff
    fig, ax = plt.subplots(figsize=(6.3,4.1))
    for method in METHODS:
        q = df[df.method == method]
        ax.scatter(q.relative_param_change, q.preservation, s=24, alpha=.65, label=method)
    ax.set_xlabel("Relative parameter change")
    ax.set_ylabel("Prediction preservation")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir/"repair_tradeoff.pdf")
    plt.close(fig)
    # Beta frontier
    fig, ax = plt.subplots(figsize=(6.3,4.1))
    q = beta_df.groupby("beta").agg(preservation=("preservation","mean"), structural_cost=("structural_cost","mean"))
    ax.plot(q.index, q.preservation, marker="o")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("MSR sensitivity weight beta")
    ax.set_ylabel("Mean prediction preservation")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figdir/"beta_sensitivity.pdf")
    plt.close(fig)
    # Sequential preservation
    fig, ax = plt.subplots(figsize=(6.3,4.1))
    for method in ["FT","Euclidean-MSR","MSR"]:
        q=seq_df[seq_df.method==method].groupby("step").preservation.agg(["mean","std"])
        ax.errorbar(q.index,q["mean"],yerr=q["std"],marker="o",capsize=3,label=method)
    ax.set_xlabel("Sequential edit step")
    ax.set_ylabel("Prediction preservation")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figdir/"sequential_preservation.pdf"); plt.close(fig)


def run(outdir: Path, seeds=DEFAULT_SEEDS, ks=DEFAULT_EDIT_SIZES):
    outdir.mkdir(parents=True, exist_ok=True)
    rows=[]
    # Train each base model once, then evaluate all edit sizes on the same base.
    for dataset in DATASETS:
        for seed in seeds:
            Xtr, ytr, Xte, yte, c, h, d, base, base_vec, correct = _build_case(dataset, seed)
            for k in ks:
                if len(correct) < k + 10:
                    continue
                eidx = correct[:k]
                eset = set(eidx.tolist())
                pidx = torch.tensor([i for i in range(len(Xte)) if i not in eset])
                Xe, Xp, yp = Xte[eidx], Xte[pidx], yte[pidx]
                with torch.no_grad():
                    order = torch.argsort(base(Xe), dim=1, descending=True)
                    yt = order[:,1]
                sens = preservation_sensitivity(base, Xp)
                common_metric = 1 + 50*sens
                for method in METHODS:
                    m=clone_model(base,d,c,h); t=time.perf_counter(); iterations=0
                    if method=='FT': m,iterations=finetune(m,Xe,yt,base_vec,steps=160)
                    elif method=='L2-SP': m,iterations=finetune(m,Xe,yt,base_vec,l2=50.0,steps=180)
                    elif method=='EWC': m,iterations=finetune(m,Xe,yt,base_vec,fisher=sens,ewc=50.0,steps=180)
                    elif method=='Euclidean-MSR': m,iterations,_=projection_repair(m,Xe,yt,torch.ones_like(sens),rounds=10)
                    elif method=='MSR': m,iterations,_=projection_repair(m,Xe,yt,1+8*sens,rounds=10)
                    elapsed=time.perf_counter()-t
                    ev=evaluate(m,base,Xe,yt,Xp,yp,base_vec,common_metric)
                    ev.update(dataset=dataset,seed=seed,n_edits=k,method=method,
                              beta=8.0 if method=='MSR' else np.nan,margin=.75,
                              iterations=iterations,runtime_s=elapsed)
                    rows.append(ev)
                print(f"completed {dataset} seed={seed} edits={k}")
    df=pd.DataFrame(rows)
    df.to_csv(outdir/"raw_results.csv",index=False)

    agg=df.groupby(["dataset","n_edits","method"]).agg(
        edit_success_mean=("edit_success","mean"), edit_success_std=("edit_success","std"),
        preservation_mean=("preservation","mean"), preservation_std=("preservation","std"),
        preserve_accuracy_mean=("preserve_accuracy","mean"), preserve_accuracy_std=("preserve_accuracy","std"),
        logit_drift_mean=("logit_drift","mean"), logit_drift_std=("logit_drift","std"),
        preservation_kl_mean=("preservation_kl","mean"), preservation_kl_std=("preservation_kl","std"),
        relative_param_change_mean=("relative_param_change","mean"), relative_param_change_std=("relative_param_change","std"),
        structural_cost_mean=("structural_cost","mean"), structural_cost_std=("structural_cost","std"),
        runtime_s_mean=("runtime_s","mean"), runtime_s_std=("runtime_s","std"),
    ).reset_index()
    agg.to_csv(outdir/"aggregate_results.csv",index=False)

    overall=df.groupby("method").agg(
        edit_success=("edit_success","mean"), preservation=("preservation","mean"),
        preserve_accuracy=("preserve_accuracy","mean"), logit_drift=("logit_drift","mean"),
        preservation_kl=("preservation_kl","mean"), relative_param_change=("relative_param_change","mean"),
        structural_cost=("structural_cost","mean"), runtime_s=("runtime_s","mean"),
    ).reset_index()
    overall.to_csv(outdir/"overall_summary.csv",index=False)

    stats=paired_statistics(df)
    stats.to_csv(outdir/"paired_tests.csv",index=False)

    idx=["dataset","seed","n_edits"]
    a=df[df.method=="MSR"].set_index(idx); b=df[df.method=="Euclidean-MSR"].set_index(idx)
    matched=pd.DataFrame(index=a.index)
    for metric in ["preservation","structural_cost","relative_param_change","logit_drift","preservation_kl"]:
        matched[f"msr_{metric}"]=a[metric]
        matched[f"euclidean_{metric}"]=b[metric]
        matched[f"delta_{metric}"]=a[metric]-b[metric]
    matched.reset_index().to_csv(outdir/"msr_vs_euclidean_matched.csv",index=False)

    beta_df=run_beta_sweep(outdir,seeds=(7,),betas=(0,1,2,4,8,16,32))
    seq_df=run_sequential(outdir,seeds=(7,),steps=6)
    save_plots(df,beta_df,seq_df,outdir)

    meta={
        "seeds":list(seeds),"edit_sizes":list(ks),"datasets":DATASETS,
        "torch":torch.__version__,"numpy":np.__version__,"pandas":pd.__version__,
        "beta_default":8.0,"common_metric_beta":50.0,"target_margin":0.75,
        "beta_sweep_seeds":[7],"sequential_seeds":[7],"sequential_steps":6,
        "notes":"All datasets are bundled with scikit-learn; no external data download or API key is required."
    }
    (outdir/"run_metadata.json").write_text(json.dumps(meta,indent=2))
    return df


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",default="results")
    ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()
    seeds=(7,19,31) if args.quick else DEFAULT_SEEDS
    run(Path(args.outdir),seeds=seeds)
