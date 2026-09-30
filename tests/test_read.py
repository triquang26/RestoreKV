import torch

from prgf.read import read_matching_loss


def _setup(kept_all: bool):
    torch.manual_seed(0)
    B, Hq, Hkv, L, T, n, d = 2, 4, 2, 6, 40, 3, 8
    direction = torch.randn(d)
    q = direction.expand(B, Hq, L, d) + 0.01 * torch.randn(B, Hq, L, d)
    k, v = torch.randn(B, Hkv, T + L, d), torch.randn(B, Hkv, T + L, d)
    k[:, :, :T] = k[0:1, :, :T]  # the context part of the cache is shared by the batch rows
    v[:, :, :T] = v[0:1, :, :T]
    kept = torch.ones(1, Hkv, T, dtype=torch.bool) if kept_all else torch.rand(1, Hkv, T) < 0.3
    sk = (-100 * direction).expand(1, Hkv, n, d).clone().requires_grad_()  # slots nobody attends to
    sv = torch.randn(1, Hkv, n, d, requires_grad=True)
    rows, cols = torch.tensor([0, 0, 1]), torch.tensor([1, 4, 5])
    return {0: (q, k, v)}, kept, {0: (sk, sv)}, rows, cols, T


def test_read_loss_vanishes_without_eviction():
    captured, kept, slots, rows, cols, T = _setup(kept_all=True)
    assert read_matching_loss(captured, kept, slots, rows, cols, T) < 1e-8


def test_read_loss_trains_the_slots_when_context_is_evicted():
    captured, kept, slots, rows, cols, T = _setup(kept_all=False)
    sk, sv = slots[0]
    sk.data = torch.randn_like(sk)  # attended slots
    loss = read_matching_loss(captured, kept, slots, rows, cols, T)
    assert loss > 0
    loss.backward()
    assert sk.grad.abs().sum() > 0 and sv.grad.abs().sum() > 0
    opt = torch.optim.Adam([sk, sv], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        read_matching_loss(captured, kept, slots, rows, cols, T).backward()
        opt.step()
    assert read_matching_loss(captured, kept, slots, rows, cols, T) < 0.5 * loss.item()  # slots learn the missing reads
