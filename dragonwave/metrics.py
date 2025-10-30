
import math, torch

def perplexity(logits, targets):
    B,T,V = logits.shape
    logp = logits.log_softmax(dim=-1)
    nll = torch.nn.functional.nll_loss(logp.reshape(-1,V), targets.reshape(-1), reduction='mean')
    return math.exp(nll.item())

def ece(logits, targets, n_bins=15):
    with torch.no_grad():
        probs = logits.softmax(-1)
        conf, preds = probs.max(-1)
        correct = (preds == targets).float()
        conf = conf.reshape(-1); correct = correct.reshape(-1)
        ece = 0.0
        bins = torch.linspace(0, 1, n_bins+1, device=conf.device)
        for i in range(n_bins):
            m = (conf >= bins[i]) & (conf < bins[i+1])
            if m.any():
                acc = correct[m].mean()
                c = conf[m].mean()
                ece += (m.float().mean() * (acc - c).abs()).item()
        return ece

def brier(logits, targets):
    with torch.no_grad():
        B,T,V = logits.shape
        p = logits.softmax(-1)
        y = torch.zeros_like(p).scatter_(-1, targets.unsqueeze(-1), 1.0)
        return ((p - y)**2).mean().item()

def false_confidence_rate(logits, targets, thresh=0.7):
    with torch.no_grad():
        p = logits.softmax(-1)
        conf, pred = p.max(-1)
        wrong = (pred != targets)
        return ((wrong & (conf > thresh)).float().mean().item())
