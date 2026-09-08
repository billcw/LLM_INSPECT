"""Dependency-light reference mathematics. No model weights or network needed.

These routines also drive the hand-checkable toy lessons; they are NOT a
substitute for running the model-backed acceptance tests in verify_model.py.
"""
import numpy as np


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or not z.size or not np.isfinite(z).all():
        raise ValueError("Expected a nonempty, finite logit vector")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    if temperature == 0:
        p = np.zeros_like(z)
        p[int(np.argmax(z))] = 1
        return p
    z = (z - np.max(z)) / temperature
    p = np.exp(z)
    return p / p.sum()


def probability_metrics(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    for x in (p, q):
        if x.ndim != 1 or not np.isfinite(x).all() or (x < 0).any() or not np.isclose(x.sum(), 1):
            raise ValueError("Expected normalized probability vectors")
    if p.shape != q.shape:
        raise ValueError("Probability vectors must have the same shape")
    mid = (p + q) / 2
    def kl(a, b):
        mask = a > 0
        if (b[mask] == 0).any():
            return float("inf")
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return {"tv": float(np.abs(p - q).sum() / 2), "js_nats": (kl(p, mid) + kl(q, mid)) / 2}


def attention_reference(q, k, v):
    q, k, v = (np.asarray(a, float) for a in (q, k, v))
    if q.ndim != 2 or k.shape != q.shape or v.shape[0] != len(q):
        raise ValueError("Toy self-attention needs matching position counts")
    scores = q @ k.T / np.sqrt(q.shape[-1])
    mask = np.triu(np.ones(scores.shape, dtype=bool), 1)
    weights = np.zeros_like(scores)
    for i in range(len(q)):
        weights[i, :i+1] = softmax(scores[i, :i+1])
    return {"scores": scores, "masked": mask, "weights": weights, "output": weights @ v}


def rmsnorm(x, weight, epsilon=1e-5):
    x = np.asarray(x, float)
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + epsilon) * weight


def aggregate_heads(heads, mode):
    a = np.asarray(heads, float)
    if mode == "mean":
        return a.mean(axis=0)
    if mode == "max":
        return a.max(axis=0)
    return a[int(mode)]


def affine_fit(x, y, ridge=1.0):
    """Dual ridge fit with intercept. Stores O(n*d), not O(d*d), factors.

    Fits the remaining update y-x. A held-out vector is mapped to
    h + mean_delta + (h-mean_x) @ x_centered.T @ coefficients.
    This is a final-state regression probe, NOT the KL-trained tuned lens.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or x.shape != y.shape or len(x) < 2 or ridge <= 0:
        raise ValueError("Need matching matrices, >=2 examples, and positive ridge")
    mx = x.mean(axis=0)
    delta = y-x
    md = delta.mean(axis=0)
    xc = x-mx
    # Scale the penalty to the average Gram diagonal, avoiding layer-scale bias.
    penalty = ridge * max(float(np.mean(np.sum(xc*xc, axis=1))), 1e-12)
    coeff = np.linalg.solve(xc @ xc.T + penalty*np.eye(len(x)), delta-md)
    return {"mean_x": mx, "mean_delta": md, "x_centered": xc, "coefficients": coeff,
            "ridge": ridge, "effective_penalty": penalty}


def affine_apply(h, fit):
    h = np.asarray(h, float)
    return h + fit["mean_delta"] + (h-fit["mean_x"]) @ fit["x_centered"].T @ fit["coefficients"]


def toy_attention():
    # Exact dimensions and all values are displayed. The toy has no RoPE.
    q = [[1, 0], [0, 1], [1, 1]]
    k = [[1, 0], [0, 1], [1, 1]]
    v = [[2, 0], [0, 4], [2, 2]]
    result = attention_reference(q, k, v)
    return {"label": "TOY: invented values, 3 positions, 2 dimensions, no RoPE",
            "q": q, "k": k, "v": v, **{a: b.tolist() for a, b in result.items()},
            "scale": float(np.sqrt(2)), "row_sums": result["weights"].sum(axis=1).tolist()}


def toy_training(steps=1, learning_rate=0.2):
    """Analytic softmax-classifier update, NOT training the pretrained LLM."""
    if not 1 <= steps <= 50 or not 0 < learning_rate <= 1:
        raise ValueError("Invalid toy training settings")
    x = np.array([1.0, 0.5])
    w = np.array([[0.1, -0.2], [0.0, 0.3], [-0.1, 0.2]])
    initial = w.copy()
    target, records = 1, []
    for i in range(steps):
        logits = w @ x
        p = softmax(logits)
        dz = p.copy(); dz[target] -= 1
        gradient = np.outer(dz, x)
        new_w = w-learning_rate*gradient
        next_p = softmax(new_w @ x)
        records.append({"step": i+1, "weights_before": w.tolist(), "logits": logits.tolist(),
                        "probabilities": p.tolist(), "loss": float(-np.log(p[target])),
                        "d_loss_d_logits": dz.tolist(), "gradient": gradient.tolist(),
                        "weights_after": new_w.tolist(), "next_loss": float(-np.log(next_p[target]))})
        w = new_w
    return {"label": "TOY TRAINING ONLY: the pretrained model is never updated",
            "input": x.tolist(), "labels": ["Tokyo", "Paris", "Rome"], "target_id": target,
            "learning_rate": learning_rate, "initial_weights": initial.tolist(), "steps": records}
