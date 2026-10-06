"""Make the model's confidence trustworthy, using validation data only.

A classifier is calibrated when "80 % sure" is right about 80 % of the time. Few-shot models usually are not: on bees
the final model is under-confident and leans towards "healthy". Two standard fixes, fitted on the validation set:

  temperature + class bias:  p = softmax((z + b) / tau)        (tau > 1 softens, tau < 1 sharpens; b shifts classes)
      tau and b minimise the negative log-likelihood on validation (one tau, one b per class: C + 1 numbers)
  "unsure" threshold:  if max p < t, the app says "not sure, check by hand"; t is the lowest value at which the
      validation photos above it are at least TARGET accurate

ECE (expected calibration error): sort predictions into 10 confidence bins; ECE = sum over bins of
(share of photos in the bin) x |accuracy in the bin - mean confidence in the bin|. 0 = perfectly calibrated.
"""
import torch
import torch.nn.functional as F

TARGET = 0.90


def fit(logits, y, steps=200):
    """returns tau, bias minimising cross-entropy of softmax((logits + bias) / tau) on (logits, y).

    Step 1: tau alone by grid search (robust, also when validation is perfectly separated). Step 2: tau and bias
    refined by gradient steps with tau kept in [0.05, 20]. Any non-finite result falls back to the previous step,
    and in the worst case to "no change" (tau = 1, bias = 0). With no validation errors at all there is nothing to
    learn from, so the model is left unchanged."""
    z = logits.detach().float()
    nll = lambda t, b: float(F.cross_entropy((z + b) / t, y))  # noqa: E731
    zero = torch.zeros(z.shape[1])
    if (z.argmax(1) == y).all():               # no validation errors: nothing to learn, leave the model as it is
        return 1.0, zero
    grid = torch.logspace(-1.3, 1.3, 53)
    tau0 = float(min(grid, key=lambda t: nll(float(t), zero)))
    log_tau = torch.tensor([float(torch.log(torch.tensor(tau0)))], requires_grad=True)
    bias = torch.zeros(z.shape[1], requires_grad=True)
    opt = torch.optim.Adam([log_tau, bias], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        tau = log_tau.clamp(-3.0, 3.0).exp()
        loss = F.cross_entropy((z + bias) / tau, y) + 1e-3 * bias.pow(2).sum()
        loss.backward()
        opt.step()
    tau, b = float(log_tau.detach().clamp(-3.0, 3.0).exp()), bias.detach()
    if not (torch.isfinite(b).all() and torch.isfinite(torch.tensor(tau))) or nll(tau, b) > nll(tau0, zero):
        tau, b = tau0, zero
    if not torch.isfinite(torch.tensor(nll(tau, b))):
        tau, b = 1.0, zero
    return tau, b


def apply(logits, tau, bias):
    return ((logits.float() + bias) / tau).softmax(1)


def ece(probs, y, bins=10):
    conf, pred = probs.max(1)
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * ((pred[m] == y[m]).float().mean() - conf[m].mean()).abs()
    return float(total)


def unsure_threshold(probs, y, target=TARGET):
    """lowest confidence t such that validation photos with max p >= t are at least `target` accurate (None if never)."""
    conf, pred = probs.max(1)
    for t in torch.linspace(0.5, 0.99, 50):
        m = conf >= t
        if m.sum() >= 10 and (pred[m] == y[m]).float().mean() >= target:
            return float(t)
    return None
