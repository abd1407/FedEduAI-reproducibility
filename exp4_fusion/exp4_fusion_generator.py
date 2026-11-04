"""FedEduAI, Experiment 4: multimodal fusion on a generated cohort (controlled mechanism demonstration).

Regenerates the cohort described in Section IV-E of the paper with declared seeds, trains the single-modality
and fusion predictors, and writes the results (JSON + Markdown) and Fig. 6.

Generative process (three stages, as stated in the paper)
  1. Latent structure: one shared ability a ~ N(0, 1) per learner and four unique factors u_j ~ N(0, 1), j in
     {academic, aptitude, engagement, knowledge progress}, all mutually independent.
  2. Observed scores: z_j = lambda_j * a + sqrt(1 - lambda_j^2) * u_j (unit variance), then x_j = 50 + 15 z_j
     clipped to [0, 100].  lambda_j (loading on the shared ability) is the per-modality mixing coefficient.
  3. Outcome: logit(p) = beta_0 + sum_j beta_j z_j ; y ~ Bernoulli(p).  No single modality is sufficient because
     every beta_j > 0 and the unique factors carry most of the variance of each score.

Predictors (5-fold stratified cross-validation, ROC-AUC on the held-out fold)
  - single modality: logistic regression on one score;
  - early fusion: logistic regression on the four scores;
  - weighted late fusion (SIS): logistic regression on the four standardised scores, weights w_j = the
    normalised positive coefficients (sum to 1), SIS = sum_j w_j x_j, AUC of SIS on the held-out fold;
  - attention-style neural fusion: per-modality value v_j = s_j x_j + c_j and attention
    alpha = softmax(W x + b) over the four modalities, output sigmoid(sum_j alpha_j v_j + d), trained with Adam
    (numpy implementation, full-batch, 300 epochs, lr 0.05, L2 1e-3).

Usage: python exp4_fusion_generator.py --out results_dir
"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

MODALITIES = ["academic", "aptitude", "engagement", "knowledge_progress"]
LABELS = {"academic": "Academic", "aptitude": "Aptitude", "engagement": "Engagement", "knowledge_progress": "Knowledge\nProgress"}

# ---- declared configuration -------------------------------------------------------------------------------
CONFIG = {
    "n_learners": 1500,
    "cohort_seed": 2026,          # numpy default_rng seed for stages 1-3
    "split_seed": 42,             # StratifiedKFold shuffle seed
    "attention_seed": 7,          # weight initialisation of the attention network
    "n_splits": 5,
    "score_mean": 50.0, "score_sd": 15.0,
    # per-modality loading on the shared ability (mixing coefficient)
    "lambda": {"academic": 0.60, "aptitude": 0.55, "engagement": 0.45, "knowledge_progress": 0.60},
    # outcome model on the standardised latent scores z_j
    "beta0": -0.40,
    "beta": {"academic": 0.48, "aptitude": 0.40, "engagement": 0.36, "knowledge_progress": 0.57},
    "attention": {"epochs": 300, "lr": 0.05, "l2": 1e-3, "hidden": None},
}


def generate(cfg):
    rng = np.random.default_rng(cfg["cohort_seed"])
    n = cfg["n_learners"]
    a = rng.standard_normal(n)
    z, x = {}, {}
    for m in MODALITIES:
        lam = cfg["lambda"][m]
        u = rng.standard_normal(n)
        z[m] = lam * a + np.sqrt(1 - lam ** 2) * u
        x[m] = np.clip(cfg["score_mean"] + cfg["score_sd"] * z[m], 0, 100)
    logit = cfg["beta0"] + sum(cfg["beta"][m] * z[m] for m in MODALITIES)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    X = np.column_stack([x[m] for m in MODALITIES])
    return X, y, p


def attention_fusion(Xtr, ytr, Xte, seed, epochs, lr, l2):
    """Attention-style fusion: alpha = softmax(W x + b); v = s*x + c; out = sigmoid(alpha.v + d)."""
    rng = np.random.default_rng(seed)
    k = Xtr.shape[1]
    W = rng.normal(0, 0.1, (k, k)); b = np.zeros(k); s = rng.normal(0, 0.1, k); c = np.zeros(k); d = np.zeros(1)
    params = [W, b, s, c, d]
    mom = [np.zeros_like(q) for q in params]; vel = [np.zeros_like(q) for q in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(ytr)
    for t in range(1, epochs + 1):
        logits_a = Xtr @ W.T + b                     # (n,k)
        logits_a -= logits_a.max(1, keepdims=True)
        alpha = np.exp(logits_a); alpha /= alpha.sum(1, keepdims=True)
        v = Xtr * s + c                              # (n,k)
        out = (alpha * v).sum(1) + d                 # (n,)
        p = 1 / (1 + np.exp(-out))
        g_out = (p - ytr) / n                        # dL/dout
        g_d = np.array([g_out.sum()])
        g_v = g_out[:, None] * alpha                 # (n,k)
        g_s = (g_v * Xtr).sum(0) + l2 * s
        g_c = g_v.sum(0)
        g_alpha = g_out[:, None] * v                 # (n,k)
        g_logits = alpha * (g_alpha - (g_alpha * alpha).sum(1, keepdims=True))
        g_W = g_logits.T @ Xtr + l2 * W
        g_b = g_logits.sum(0)
        grads = [g_W, g_b, g_s, g_c, g_d]
        for i, (q, g) in enumerate(zip(params, grads)):
            mom[i] = b1 * mom[i] + (1 - b1) * g
            vel[i] = b2 * vel[i] + (1 - b2) * g * g
            q -= lr * (mom[i] / (1 - b1 ** t)) / (np.sqrt(vel[i] / (1 - b2 ** t)) + eps)
    la = Xte @ W.T + b; la -= la.max(1, keepdims=True); al = np.exp(la); al /= al.sum(1, keepdims=True)
    return (al * (Xte * s + c)).sum(1) + d


def run(cfg):
    X, y, p_true = generate(cfg)
    skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["split_seed"])
    res = {m: [] for m in MODALITIES}
    res.update({"late_fusion": [], "early_fusion": [], "attention_fusion": []})
    weights = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for j, m in enumerate(MODALITIES):
            clf = LogisticRegression().fit(Xtr[:, [j]], y[tr])
            res[m].append(roc_auc_score(y[te], clf.decision_function(Xte[:, [j]])))
        clf = LogisticRegression().fit(Xtr, y[tr])
        res["early_fusion"].append(roc_auc_score(y[te], clf.decision_function(Xte)))
        coef = np.clip(clf.coef_[0], 0, None); w = coef / coef.sum()
        weights.append(w)
        sis = X[te] @ w                               # weighted late fusion on the 0-100 scores
        res["late_fusion"].append(roc_auc_score(y[te], sis))
        att = attention_fusion(Xtr, y[tr], Xte, cfg["attention_seed"] + fold, **{k: v for k, v in cfg["attention"].items() if k != "hidden"})
        res["attention_fusion"].append(roc_auc_score(y[te], att))
    summary = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "folds": [float(a) for a in v]} for k, v in res.items()}
    wmean = np.mean(weights, 0)
    out = {
        "config": cfg,
        "cohort": {"n": int(len(y)), "positive_rate": float(y.mean()),
                    "score_means": {m: float(X[:, j].mean()) for j, m in enumerate(MODALITIES)},
                    "score_sds": {m: float(X[:, j].std(ddof=1)) for j, m in enumerate(MODALITIES)},
                    "score_correlations": np.corrcoef(X.T).round(3).tolist(),
                    "oracle_auc_true_probability": float(roc_auc_score(y, p_true))},
        "auc": summary,
        "sis_weights_mean": {m: float(wmean[j]) for j, m in enumerate(MODALITIES)},
        "sis_weights_per_fold": [[float(a) for a in w] for w in weights],
    }
    return out, X, y


def figure(out, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = MODALITIES + ["late_fusion", "early_fusion", "attention_fusion"]
    names = [LABELS[m] for m in MODALITIES] + ["Late\nFusion", "Early\nFusion", "Attention\nFusion"]
    means = [out["auc"][k]["mean"] for k in keys]; sds = [out["auc"][k]["sd"] for k in keys]
    colors = ["#2b5fb3"] * 4 + ["#c8732a"] * 3
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=160)
    xs = np.arange(len(keys))
    ax.bar(xs, means, color=colors, yerr=sds, capsize=5, error_kw={"elinewidth": 1.6, "ecolor": "#444"})
    for xi, m, s in zip(xs, means, sds):
        ax.text(xi, m + s + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=13)
    ax.set_ylabel("ROC-AUC", fontsize=13); ax.set_ylim(0.60, max(means) + 0.06)
    ax.set_title("Experiment 4 — Multi-Modal Fusion vs Single Modality", fontsize=15, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    w = out["sis_weights_mean"]
    txt = f"SIS weights: Academic {w['academic']:.2f}  Engagement {w['engagement']:.2f}  Knowledge progress {w['knowledge_progress']:.2f}  Aptitude {w['aptitude']:.2f}"
    ax.text(0.99, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color="#444",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f2f4f7", "edgecolor": "#cfd4dc"})
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args(); od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    out, X, y = run(CONFIG)
    (od / "exp4_results.json").write_text(json.dumps(out, indent=1))
    np.savetxt(od / "exp4_cohort.csv", np.column_stack([X, y]), delimiter=",", header=",".join(MODALITIES + ["success"]), comments="", fmt="%.4f")
    lines = ["| Predictor | AUC mean ± SD (5 folds) |", "|---|---:|"]
    for k in MODALITIES + ["late_fusion", "early_fusion", "attention_fusion"]:
        lines.append(f"| {k} | {out['auc'][k]['mean']:.3f} ± {out['auc'][k]['sd']:.3f} |")
    lines.append(""); lines.append("SIS weights (mean over folds): " + ", ".join(f"{m} {v:.3f}" for m, v in out["sis_weights_mean"].items()))
    lines.append(f"Positive rate {out['cohort']['positive_rate']:.3f}; oracle AUC {out['cohort']['oracle_auc_true_probability']:.3f}")
    (od / "exp4_results.md").write_text("\n".join(lines))
    print("\n".join(lines))
    if not a.no_fig: figure(out, od / "fig6.png")
