"""Make train_vq.py's per-step bookkeeping optional: the loss logs, the progress text and the
amplitude/frequency samples (used only by the plots) run every --log_every steps instead of
every step. The training math is unchanged. Run from the walk-the-dog worktree root."""
import re

p = "train_vq.py"
s = open(p).read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


rep("""                train_batch = next(its[class_idx])""",
    """                train_batch = next(its[class_idx])
                log_now = n_iters % args.log_every == 0""")
rep("""                _a_ = Item(params[2]).reshape(-1, args.phase_channels).numpy()""",
    """                _a_ = Item(params[2]).reshape(-1, args.phase_channels).numpy() if log_now else np.zeros((0, args.phase_channels))""")
rep("""                _f_ = Item(params[1]).reshape(-1, args.phase_channels).numpy()""",
    """                _f_ = Item(params[1]).reshape(-1, args.phase_channels).numpy() if log_now else np.zeros((0, args.phase_channels))""")
rep("""                    for i, p in enumerate(perplexities):
                        loss_recorder.add_scalar(f'{class_idx}/perplexity_{i}', p)""",
    """                    for i, p in enumerate(perplexities if log_now else []):
                        loss_recorder.add_scalar(f'{class_idx}/perplexity_{i}', p)""")
rep("""                for k in losses:
                    loss_recorder.add_scalar(f'{class_idx}/loss_{k}', losses[k].item())
                loss_recorder.add_scalar(f'{class_idx}/loss_total', loss_total.item())

                loss_descript = ' '.join([f'{k}: {v.item():.4f}' for k, v in losses.items()])
                loss_descript = f'total: {loss_total.item():.4f} ' + loss_descript
                loop.set_description(loss_descript)""",
    """                if log_now:
                    for k in losses:
                        loss_recorder.add_scalar(f'{class_idx}/loss_{k}', losses[k].item())
                    loss_recorder.add_scalar(f'{class_idx}/loss_total', loss_total.item())

                    loss_descript = ' '.join([f'{k}: {v.item():.4f}' for k, v in losses.items()])
                    loss_descript = f'total: {loss_total.item():.4f} ' + loss_descript
                    loop.set_description(loss_descript)""")
rep("""                loss_history.Add(
                    (Item(losses['rec']).item(), "Reconstruction Loss")""",
    """                if log_now: loss_history.Add(
                    (Item(losses['rec']).item(), "Reconstruction Loss")""")
if "import numpy as np" not in s:
    s = "import numpy as np\n" + s
open(p, "w").write(s)

o = "option.py"
t = open(o).read()
a = "        self.parser.add_argument('--epochs', type=int, default=10)"
assert t.count(a) == 1
t = t.replace(a, a + "\n        self.parser.add_argument('--log_every', type=int, default=1)")
open(o, "w").write(t)
print("PATCHED")
