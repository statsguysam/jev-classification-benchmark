"""Scores include failures; probability metrics explicitly report their coverage."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss, roc_auc_score, average_precision_score, precision_recall_fscore_support
from .types import Row, Prediction


def evaluate(rows: list[Row], predictions: list[Prediction], n_classes: int) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate an empty test set")
    by_id = {p.row_id: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r.id for r in rows}:
        raise ValueError("Predictions must match test IDs exactly, with no duplicates")
    y = np.asarray([r.label for r in rows])
    ordered = [by_id[r.id] for r in rows]
    pred = np.asarray([p.label if p.error is None and p.label is not None and 0 <= p.label < n_classes else -1 for p in ordered])
    valid = pred >= 0
    recalls = [float(np.mean(pred[y == c] == c)) for c in range(n_classes) if np.any(y == c)]
    cm = confusion_matrix(y, pred, labels=[*range(n_classes), -1])[:n_classes].tolist()
    metrics = {
        "n_test": len(rows), "n_classes": n_classes,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean(recalls)),
        "failure_rate": float(1 - valid.mean()),
        "n_failures": int((~valid).sum()),
        "confusion_matrix": cm,
        "confusion_matrix_columns": [*range(n_classes), "failure"],
    }
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=list(range(n_classes)), zero_division=0)
    metrics["per_class"] = [{"label": c, "precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c]), "support": int(support[c])} for c in range(n_classes)]
    metrics["valid_response_accuracy"] = float(np.mean(y[valid] == pred[valid])) if valid.any() else None
    latencies = [p.latency_s for p in ordered if p.latency_s >= 0]
    metrics.update({"latency_p50_s": float(np.quantile(latencies, .5)), "latency_p95_s": float(np.quantile(latencies, .95)), "prediction_time_s": float(sum(latencies))})
    for key in ("input_tokens", "output_tokens"):
        known = [getattr(p, key) for p in ordered if getattr(p, key) is not None]
        metrics[key] = int(sum(known)) if len(known) == len(rows) else None
        metrics[key + "_known_total"] = int(sum(known))
        metrics[key + "_coverage"] = len(known) / len(rows)
    probability_indices, probabilities = [], []
    for i, p in enumerate(ordered):
        if p.probabilities is None or not valid[i]:
            continue
        a = np.asarray(p.probabilities, dtype=float)
        if a.shape != (n_classes,) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() or not np.isclose(a.sum(), 1, atol=1e-6, rtol=0):
            raise ValueError(f"Invalid class probability distribution for row {p.row_id}")
        probability_indices.append(i)
        probabilities.append(a)
    metrics["probability_coverage"] = len(probabilities) / len(rows)
    metrics["n_probability_rows"] = len(probabilities)
    if probabilities:
        probs = np.asarray(probabilities)
        metrics["probability_renormalized_rows"] = int(np.sum(probs.sum(axis=1) != 1.0))
        probs = probs / probs.sum(axis=1, keepdims=True)
        targets = y[probability_indices]
        chosen = pred[probability_indices]
        metrics["zero_true_class_probability_rows"] = int(np.sum(probs[np.arange(len(probs)), targets] == 0))
        clipped = np.clip(probs, 1e-15, 1)
        clipped /= clipped.sum(axis=1, keepdims=True)
        metrics["log_loss"] = float(-np.log(clipped[np.arange(len(probs)), targets]).mean())
        metrics["brier_sum"] = float(np.mean(np.sum((probs - np.eye(n_classes)[targets]) ** 2, axis=1)))
        argmax = np.argmax(probs, axis=1)
        metrics["chosen_label_argmax_disagreement_rows"] = int(np.sum(chosen != argmax))
        conf = probs.max(axis=1)
        correct = argmax == targets
        bins = np.minimum((conf * 15).astype(int), 14)
        metrics["ece_15_equal_width"] = float(sum(np.mean(bins == b) * abs(correct[bins == b].mean() - conf[bins == b].mean()) for b in range(15) if np.any(bins == b)))
        metrics["reliability_bins"] = [{"bin": b, "count": int(np.sum(bins == b)), "mean_confidence": float(conf[bins == b].mean()) if np.any(bins == b) else None, "accuracy": float(correct[bins == b].mean()) if np.any(bins == b) else None} for b in range(15)]
        if n_classes == 2 and len(set(targets)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(targets, probs[:, 1]))
            metrics["average_precision"] = float(average_precision_score(targets, probs[:, 1]))
    return metrics


def stratified_bootstrap(y: list[int], a: list[int], b: list[int] | None = None, *, n_classes: int, samples: int = 1000, seed: int = 42) -> dict:
    """Paired differences are A minus B; resample within each true class."""
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    y, a = np.asarray(y), np.asarray(a)
    b = None if b is None else np.asarray(b)
    if y.shape != a.shape or (b is not None and b.shape != y.shape):
        raise ValueError("Prediction arrays must align")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in range(n_classes) if np.any(y == c)]
    values = {"accuracy": [], "macro_f1": []}
    def scores(t, p):
        return np.mean(t == p), f1_score(t, p, labels=list(range(n_classes)), average="macro", zero_division=0)
    for _ in range(samples):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        sa = scores(y[idx], a[idx])
        sb = (0, 0) if b is None else scores(y[idx], b[idx])
        for name, va, vb in zip(values, sa, sb):
            values[name].append(float(va - vb))
    pa, pb = scores(y, a), (0, 0) if b is None else scores(y, b)
    return {"method": "paired stratified percentile bootstrap" if b is not None else "stratified percentile bootstrap", "samples": samples, "seed": seed,
            "metrics": {name: {"estimate": float(va - vb), "ci95": np.quantile(values[name], [.025, .975]).tolist()} for name, va, vb in zip(values, pa, pb)}}
