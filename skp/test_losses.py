import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import log_loss


def torch_log_loss(p, t, w=None):
	loss = -torch.xlogy(t.float(), p.float()).sum(1)
	if isinstance(w, torch.Tensor):
		loss = loss * w
		return loss.sum() / w.sum()
	else:
		return loss.mean()


def torch_log_loss_with_softmax(logits, t):
	loss = -torch.xlogy(t.float(), F.softmax(logits, dim=1)).sum(1)
	return loss.mean()


def torch_log_loss_with_logits(logits, t, w=None):
	loss = (-t.float() * F.log_softmax(logits, dim=1)).sum(1)
	if isinstance(w, torch.Tensor):
		loss = loss * w
		return loss.sum() / w.sum()
	else:
		return loss.mean()


def torch_nll_loss(p, t):
	loss = 0
	for i in range(p.shape[1]):
		loss += nll_loss(torch.log(p[:, i]), t[:, i].long())
	return loss / p.shape[1]


def inv_logit(x):
	return torch.log(x / (1 - x))


y = torch.empty((1000, 3))
y[...] = 0.5
y = torch.bernoulli(y)
logits = torch.randn((1000, 3))
p = F.softmax(logits, dim=1)

w = torch.ones((len(y), ))
w[y[:, 1] == 1] = 2
w[y[:, 2] == 1] = 4

log_loss(y.numpy(), p.numpy())
torch_log_loss_with_logits(logits, y)

log_loss(y.numpy(), p.numpy(), sample_weight=w.numpy())
torch_log_loss_with_logits(logits, y, w=w)

log_loss(y.numpy(), p.numpy())

F.binary_cross_entropy_with_logits(logits, y, weight=w.unsqueeze(1))
F.binary_cross_entropy_with_logits(logits, y)

F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(0).sum()

F.cross_entropy(logits, y)
F.cross_entropy(logits, torch.argmax(y, 1))


binary_cross_entropy(pt, yt)
binary_cross_entropy(pt, yt, weight=wt.unsqueeze(1))
binary_cross_entropy(pt, yt, reduction="none").mean(0).sum()

torch_log_loss(pt, yt, wt)
torch_log_loss(pt, yt)

torch_nll_loss(pt, yt)

cross_entropy(pt, yt)


import numpy as np
import torch

from sklearn.metrics import log_loss


def torch_log_loss(p, t):
	loss = -torch.xlogy(t.float(), p.float()).sum(1)
	return loss.mean()


def check_equal(x, y, eps=1e-6):
	return np.abs(x - y) < eps


# No normalization
y = torch.empty((1000, 3))
y[...] = 0.5
y = torch.bernoulli(y)
p = torch.rand((1000, 3))

sk_loss = log_loss(y.numpy(), p.numpy())
pt_loss = torch_log_loss(p, y).item()

print(check_equal(sk_loss, pt_loss))

# After normalization
p = p / p.sum(1).unsqueeze(1)

sk_loss = log_loss(y.numpy(), p.numpy())
pt_loss = torch_log_loss(p, y).item()

print(check_equal(sk_loss, pt_loss))
