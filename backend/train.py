"""Reviewer-grade experiment pipeline for the biome-locked forest cover study.

Produces models/*.joblib and models/metrics.json containing:
  1. Main comparison (6 models): train/test acc, macro-F1, balanced acc, kappa, MCC, overfit gap
  2. Statistical rigor: 5-fold CV mean+-std, McNemar test of best model vs every other
  3. Reasoning cost, measured: fit time, batch inference ms/1k, single-sample latency, model size
  4. Ablations: n_estimators, tree depth, feature groups, data fraction, scaling
  5. Biome-locked evaluation: within-wilderness-area vs leave-one-area-out generalisation
  6. Per-class precision/recall/F1, confusion matrix, grouped feature importance

Usage:
  python train.py --quick                 # ~10-20 min, good for development
  python train.py                         # full 581,012 rows (hours; add --skip-gb to save most of it)
"""
import argparse, json, platform, time, warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.stats import chi2
from sklearn.base import clone
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path(__file__).parent / "models"
OUT.mkdir(exist_ok=True)

FEATURES = (["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
             "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
             "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
            + [f"Wilderness_Area{i}" for i in range(1, 5)] + [f"Soil_Type{i}" for i in range(1, 41)])
CLASSES = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow",
           "Aspen", "Douglas-fir", "Krummholz"]
AREAS = ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"]
GROUPS = {"Elevation": [0], "Slope and aspect": [1, 2], "Water distance": [3, 4],
          "Road distance": [5], "Hillshade": [6, 7, 8], "Fire-point distance": [9],
          "Wilderness area": list(range(10, 14)), "Soil type": list(range(14, 54))}
SKIP_GB = False
DEPLOY = False  # small models for GitHub (<100 MB/file) and Render free tier (512 MB RAM)


def get_models(light=False):
    n = 50 if DEPLOY else (100 if light else 200)
    leaf = 3 if DEPLOY else 1
    m = {
        "Random Forest": RandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, n_jobs=-1, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=n, min_samples_leaf=leaf, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50 if light else 100, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=SEED),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }
    if SKIP_GB:
        m.pop("Gradient Boosting")
    return m


def strat_sub(X, y, n):
    if not n or n >= len(X):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=SEED)
    return Xs, ys


def rf(n=100, **kw):
    return RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=SEED, **kw)


def acc(y, p):
    return round(accuracy_score(y, p) * 100, 2)


def mcnemar(y, p1, p2):
    c1, c2 = p1 == y, p2 == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    if b + c == 0:
        return {"b": b, "c": c, "p": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": round(float(stat), 2), "p": float(chi2.sf(stat, 1))}


def single_latency(model, X, n=50):
    t = time.perf_counter()
    for i in range(n):
        model.predict(X[i:i + 1])
    return (time.perf_counter() - t) / n * 1000


def ablations(Xtr, ytr, Xte, yte, abl_n):
    Xa, ya = strat_sub(Xtr, ytr, abl_n)
    out = {"train_size": int(len(Xa))}

    out["n_estimators"] = []
    for n in [10, 25, 50, 100, 200]:
        t = time.perf_counter(); m = rf(n).fit(Xa, ya); fs = time.perf_counter() - t
        out["n_estimators"].append({"n": n, "test_acc": acc(yte, m.predict(Xte)), "fit_s": round(fs, 2)})
        print("  trees", out["n_estimators"][-1])

    out["max_depth"] = []
    for d in [3, 5, 10, 15, 20, 30, None]:
        m = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xa, ya)
        out["max_depth"].append({"depth": str(d), "train_acc": acc(ya, m.predict(Xa)),
                                 "test_acc": acc(yte, m.predict(Xte))})

    cols = np.arange(54)
    variants = {
        "All 54 features": cols,
        "Without soil type": cols[:14],
        "Without wilderness area": np.r_[0:10, 14:54],
        "Without water distances": np.delete(cols, [3, 4]),
        "Without all distances": np.delete(cols, [3, 4, 5, 9]),
        "10 cartographic only": cols[:10],
        "Elevation only": cols[:1],
    }
    out["feature_groups"] = []
    for name, c in variants.items():
        m = rf(100).fit(Xa[:, c], ya)
        out["feature_groups"].append({"variant": name, "n_features": int(len(c)),
                                      "test_acc": acc(yte, m.predict(Xte[:, c]))})
        print("  features", out["feature_groups"][-1])

    out["data_fraction"] = []
    for f in [0.05, 0.1, 0.25, 0.5, 1.0]:
        Xs, ys = strat_sub(Xtr, ytr, int(len(Xtr) * f))
        m = rf(100).fit(Xs, ys)
        out["data_fraction"].append({"fraction": f, "n": int(len(Xs)), "test_acc": acc(yte, m.predict(Xte))})
        print("  fraction", out["data_fraction"][-1])

    out["scaling"] = []
    for name, est in [("KNN", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
                      ("Logistic Regression", LogisticRegression(max_iter=1000))]:
        raw = clone(est).fit(Xa, ya)
        sc = make_pipeline(StandardScaler(), clone(est)).fit(Xa, ya)
        out["scaling"].append({"model": name, "raw": acc(yte, raw.predict(Xte)),
                               "scaled": acc(yte, sc.predict(Xte))})
    return out


def biome_study(X, y, n):
    """Wilderness areas act as biome proxies.
    within: train+test inside one area.  loao: train on three areas, test on the held-out one."""
    area_all = X[:, 10:14].argmax(1)
    counts = [[int(((area_all == a) & (y == c)).sum()) for c in range(7)] for a in range(4)]
    Xb, yb = strat_sub(X, y, n)
    area = Xb[:, 10:14].argmax(1)
    models = get_models(light=True)
    res = {k: {"within": [], "loao": []} for k in models}
    for a in range(4):
        idx = area == a
        Xa, ya = Xb[idx], yb[idx]
        try:
            Xt, Xv, yt, yv = train_test_split(Xa, ya, test_size=0.2, stratify=ya, random_state=SEED)
        except ValueError:
            Xt, Xv, yt, yv = train_test_split(Xa, ya, test_size=0.2, random_state=SEED)
        for name, mdl in models.items():
            res[name]["within"].append(acc(yv, clone(mdl).fit(Xt, yt).predict(Xv)))
            res[name]["loao"].append(acc(ya, clone(mdl).fit(Xb[~idx], yb[~idx]).predict(Xa)))
        print("  biome", AREAS[a], {k: (v["within"][-1], v["loao"][-1]) for k, v in res.items()})
    return {"areas": AREAS, "sample": int(len(Xb)), "models": res, "area_class_counts": counts}


def main():
    global SKIP_GB, DEPLOY
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller samples everywhere")
    ap.add_argument("--sample", type=int, default=0, help="use N rows of the dataset (0 = all)")
    ap.add_argument("--cv-sample", type=int, default=20000)
    ap.add_argument("--abl-sample", type=int, default=100000)
    ap.add_argument("--biome-sample", type=int, default=60000)
    ap.add_argument("--skip-gb", action="store_true", help="skip Gradient Boosting (slow on full data)")
    ap.add_argument("--deploy", action="store_true", help="small models for GitHub/Render (implies --quick)")
    ap.add_argument("--test-size", type=float, default=0.2)
    a = ap.parse_args()
    SKIP_GB = a.skip_gb
    DEPLOY = a.deploy
    if a.quick or a.deploy:
        a.sample, a.cv_sample, a.abl_sample, a.biome_sample = 100000, 15000, 30000, 30000

    X, y = fetch_covtype(return_X_y=True)
    y = y - 1
    full_y = y.copy()
    biome_X, biome_y = X, y  # biome study samples from the full data
    if a.sample and a.sample < len(X):
        X, y = strat_sub(X, y, a.sample)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=a.test_size, stratify=y, random_state=SEED)
    print(f"train={len(Xtr)} test={len(Xte)}")

    # ---- 1-3. main comparison ------------------------------------------------
    Xcv, ycv = strat_sub(Xtr, ytr, a.cv_sample)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, preds, per_class, importances = [], {}, {}, {}
    for name, model in get_models().items():
        print("Training", name)
        t = time.perf_counter(); model.fit(Xtr, ytr); fit_s = time.perf_counter() - t
        t = time.perf_counter(); p = model.predict(Xte); inf_s = time.perf_counter() - t
        preds[name] = p
        path = OUT / f"{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(model, path, compress=3)
        cvs = cross_val_score(clone(model), Xcv, ycv, cv=cv) * 100
        tr, te = acc(ytr, model.predict(Xtr)), acc(yte, p)
        results.append({
            "model": name, "train_acc": tr, "test_acc": te, "gap": round(tr - te, 2),
            "cv_mean": round(float(cvs.mean()), 2), "cv_std": round(float(cvs.std()), 2),
            "macro_f1": round(f1_score(yte, p, average="macro") * 100, 2),
            "bal_acc": round(balanced_accuracy_score(yte, p) * 100, 2),
            "kappa": round(cohen_kappa_score(yte, p), 4), "mcc": round(matthews_corrcoef(yte, p), 4),
            "fit_seconds": round(fit_s, 2),
            "inference_ms_per_1k": round(inf_s / len(Xte) * 1e6, 2),
            "latency_ms_single": round(single_latency(model, Xte), 2),
            "size_mb": round(path.stat().st_size / 1e6, 1),
        })
        print(" ", results[-1])
        pr, rc, f1, sup = precision_recall_fscore_support(yte, p, labels=range(7), zero_division=0)
        per_class[name] = {"precision": (pr * 100).round(1).tolist(), "recall": (rc * 100).round(1).tolist(),
                           "f1": (f1 * 100).round(1).tolist(), "support": sup.tolist()}
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            importances[name] = {g: round(float(fi[i].sum()), 4) for g, i in GROUPS.items()}

    best = max(results, key=lambda r: r["test_acc"])["model"]
    for r in results:
        r["mcnemar_p_vs_best"] = None if r["model"] == best else mcnemar(yte, preds[best], preds[r["model"]])["p"]

    examples = []
    for c in range(7):
        i = int(np.where(yte == c)[0][0])
        examples.append({"label": CLASSES[c], "features": [float(v) for v in Xte[i]]})

    metrics = {
        "meta": {"seed": SEED, "test_size": a.test_size, "sklearn": sklearn.__version__,
                 "python": platform.python_version(), "cv_sample": int(len(Xcv)), "cv_folds": 5,
                 "n_total": int(len(X))},
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "classes": CLASSES, "features": FEATURES, "areas": AREAS,
        "class_counts": {CLASSES[i]: int((full_y == i).sum()) for i in range(7)},
        "results": results, "best_model": best, "per_class": per_class,
        "confusion_matrix": confusion_matrix(yte, preds[best]).tolist(),
        "group_importance": importances, "examples": examples,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- 4. ablations, 5. biome-locked ---------------------------------------
    print("Ablations ...")
    metrics["ablation"] = ablations(Xtr, ytr, Xte, yte, a.abl_sample)
    print("Biome-locked study ...")
    metrics["biome"] = biome_study(biome_X, biome_y, a.biome_sample)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Done. Best model:", best)


if __name__ == "__main__":
    main()