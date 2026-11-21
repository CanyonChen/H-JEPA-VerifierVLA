"""Train a 100-update Long pilot or a shared 40-task verifier from the pinned initialization."""
import argparse
import bisect
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.b5_model import JOB, atomic_write, predictor, read_json, save_json, sha256


class Windows(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes
        self.ends = []
        self.by_task = defaultdict(list)
        size = 0
        for e in episodes:
            self.by_task[e["task"]].extend(range(size, size + e["count"]))
            size += e["count"]
            self.ends.append(size)

    def __len__(self):
        return self.ends[-1]

    @staticmethod
    @lru_cache(maxsize=12)
    def load(path):
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)

    def __getitem__(self, i):
        j = bisect.bisect_right(self.ends, i)
        idx = i - (self.ends[j - 1] if j else 0)
        ep = self.episodes[j]
        d = self.load(ep["file"])
        off = int(d["offset"][idx])
        n = int(d["mask"][idx].sum())
        frames = d["target"][off:off + n]
        pad = frames[-1:].expand(5 - n, -1, -1)
        z = torch.cat((d["context"][idx:idx + 1], frames, pad))
        return z, d["actions"][idx], d["state"][idx], d["delta"][idx], d["mask"][idx], ep["task"]


def epoch_batches(data, epoch, size=32):
    rng = random.Random(7 + epoch)
    groups = []
    for task in sorted(data.by_task):
        rows = data.by_task[task].copy()
        rng.shuffle(rows)
        groups.extend(rows[i:i + 4] for i in range(0, len(rows), 4))
    rng.shuffle(groups)
    order = [i for group in groups for i in group]
    return [order[i:i + size] for i in range(0, len(order), size)]


def rank_totals(pred, truth, valid, tasks):
    # Each ordered pair has one larger target, at the SAME sub-chunk of the same task.
    target_gap = truth[:, None] - truth[None, :]
    pairs = (tasks[:, None] == tasks[None, :])[:, :, None] & valid[:, None] & valid[None, :] & (target_gap > 0)
    hinge = F.relu(target_gap - (pred[:, None] - pred[None, :]))
    return (hinge * pairs).sum((0, 1)), pairs.sum((0, 1))


def global_rank(pred, truth, valid, tasks):
    losses, pairs = rank_totals(pred, truth, valid, tasks)
    return (losses / pairs.clamp_min(1)).sum()


def run_forward(net, batch, use_self):
    z, a, state, _, _, task = batch
    z = z.float() if z.device.type == "cpu" else z
    a, s = a.unsqueeze(2), state[:, None].expand(-1, 5, -1)
    with torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=z.device.type == "cuda"):
        if use_self:
            # Stop gradient on generated history; fit the same true next-state targets.
            with torch.no_grad():
                history, _ = net.rollout_kv_cached(z[:, 0], a, s, task_id=task)
            history = history[:, :-1]
        else:
            history = z[:, :-1]
        latent, value = net(history.flatten(1, 2), a, s, task_id=task)
    return latent.reshape_as(z[:, 1:]), value.squeeze(-1).float()


def local_terms(zhat, p, batch, denom):
    z, _, _, truth, valid, _ = batch
    visual = (zhat.float() - z[:, 1:].float()).abs().mean((-1, -2))
    regression = F.smooth_l1_loss(p, truth, reduction="none")
    return ((visual * valid).sum(0) / denom).sum(), ((regression * valid).sum(0) / denom).sum()


def slice_batch(batch, start, stop, device):
    return tuple(t[start:stop].to(device, non_blocking=True) for t in batch)


def prefetched(loader):
    # Overlap mmap reads and pinned host batching with the current GPU update.
    iterator = iter(loader)
    with ThreadPoolExecutor(max_workers=1) as pool:
        upcoming = pool.submit(next, iterator, None)
        while True:
            batch = upcoming.result()
            if batch is None:
                return
            upcoming = pool.submit(next, iterator, None)
            yield batch


def backward_batch(net, batch, micro, use_self, device):
    truth, valid, tasks = (t.to(device, non_blocking=True) for t in batch[3:])
    denom = valid.sum(0).clamp_min(1)
    size = len(truth)
    # Preserve all cross-microbatch ranking pairs. A no-grad prediction pass gives
    # the exact gradient of the full-batch scalar rank loss w.r.t. each score.
    if micro < size:
        with torch.no_grad():
            vals = [run_forward(net, slice_batch(batch, i, i+micro, device), use_self)[1]
                    for i in range(0, size, micro)]
        scores = torch.cat(vals).requires_grad_()
        rank = global_rank(scores, truth, valid, tasks)
        rank_grad, = torch.autograd.grad(rank, scores)
    loss_sums = torch.zeros(2, device=device)
    for i in range(0, size, micro):
        part = slice_batch(batch, i, i+micro, device)
        zhat, p = run_forward(net, part, use_self)
        visual, huber = local_terms(zhat, p, part, denom)
        if micro >= size:
            rank = global_rank(p, truth, valid, tasks)
            rank_term = rank
        else:
            rank_term = (p * rank_grad[i:i+micro]).sum()
        (visual + huber + rank_term).backward()
        loss_sums += torch.stack((visual.detach(), huber.detach()))
    return [*loss_sums.tolist(), rank.detach().item()]


def update(net, optimizer, batch, micro, use_self, device):
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            values = backward_batch(net, batch, micro, use_self, device)
            break
        except torch.cuda.OutOfMemoryError:
            if micro <= 1 or device.type != "cuda":
                raise
        # Clear gradients only after the failed backward stack has unwound.
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        micro = max(1, micro // 2)
        print(f"Retrying the same optimizer update with microbatch {micro}", flush=True)
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True).item()
    if not (norm > 0 and all(math.isfinite(x) for x in values)):
        raise ValueError("Non-finite loss or zero gradient")
    optimizer.step()
    return values, norm, micro


@torch.inference_mode()
def validate(net, data, micro, device, limit=None, *, with_rank=False):
    net.eval()
    error_sum = 0.0
    count = 0
    rank_sums = torch.zeros(5, dtype=torch.float64, device=device)
    pair_counts = torch.zeros(5, dtype=torch.long, device=device)
    # Match deployment: all five predictions are conditioned on imagined history.
    indices = epoch_batches(data, 0)
    if limit:
        indices = indices[:limit]
    for batch in DataLoader(data, batch_sampler=indices, num_workers=0):
        predictions = []
        for i in range(0, len(batch[0]), micro):
            z, a, s, y, mask, tasks = slice_batch(batch, i, i+micro, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, values = net.rollout_kv_cached(z[:, 0].to(torch.bfloat16 if device.type=="cuda" else torch.float32),
                                                  a.unsqueeze(2), s, task_id=tasks)
            scores = values.squeeze(-1).T.float()
            err = F.smooth_l1_loss(scores, y, reduction="none")
            error_sum += (err * mask).sum().item()
            count += mask.sum().item()
            if with_rank:
                predictions.append(scores)
        if with_rank:
            truth, valid, tasks = (x.to(device) for x in batch[3:])
            losses, pairs = rank_totals(torch.cat(predictions), truth, valid, tasks)
            rank_sums += losses
            pair_counts += pairs
    net.train()
    if not count or not math.isfinite(error_sum):
        raise ValueError("No finite validation targets")
    if not with_rank:
        return error_sum / count
    rank = (rank_sums / pair_counts.clamp_min(1)).sum().item()
    if not math.isfinite(rank):
        raise ValueError("Non-finite validation ranking loss")
    return dict(validation_huber=error_sum / count, validation_rank=rank,
                valid_targets=count, ranking_pairs=pair_counts.sum().item())


def train(stage, epochs, initial_micro, workers, job=JOB):
    job = Path(job).resolve()
    directory = job / stage
    index = read_json(directory / "features.json")
    trainset, valset = Windows(index["train"]), Windows(index["validation"])
    pilot = stage == "long100"
    task_count = 10 if pilot else 40
    if set(trainset.by_task) != set(range(task_count)) or set(valset.by_task) != set(range(task_count)):
        raise ValueError("Missing task coverage")
    steps_epoch = math.ceil(len(trainset) / 32)
    goal = 100 if pilot else epochs * steps_epoch
    config = dict(stage=stage, features_sha256=sha256(directory/"features.json"), source_sha256=index["source_sha256"],
                  tasks=task_count, suites=index["suites"], epochs=None if pilot else epochs, target_updates=goal,
                  seed=7, effective_batch=32, lr=.0005, weight_decay=.04, visual_weight=1, rank_weight=1,
                  margin=1, gradient_clip=1, teacher_self="alternating updates; detached self-generated history",
                  action_state="LIBERO OSC 7D actions and 8D state; newly trained input projections")
    ck = directory / "last.pt"
    best_path = directory / "best.pt"
    summary = directory / "training.json"
    if summary.exists():
        done = read_json(summary)
        if done["config"] != config or done["updates"] != goal or not ck.is_file() or not best_path.is_file():
            raise ValueError("Different completed training run in output directory")
        return
    saved = torch.load(ck, weights_only=True, map_location="cpu", mmap=True) if ck.exists() else None
    if saved is not None:
        if config != saved["config"] or not 0 <= saved["step"] <= goal:
            raise ValueError("Training configuration or update count differs from checkpoint")
        if saved["best"] is not None and not best_path.is_file():
            raise FileNotFoundError(f"Missing best validation checkpoint: {best_path}")
        # The final checkpoint commits the report before publishing its JSON copy.
        if saved["step"] == goal and "result" in saved:
            save_json(summary, saved["result"])
            return
    device = torch.device("cuda:0")
    torch.manual_seed(7)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    net = predictor(task_count, pretrained=saved is None, job=job).to(device).train()
    opt = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=.0005, weight_decay=.04, fused=True)
    step, micro, best, train_seconds, warm_seconds, warm_count = 0, initial_micro, None, 0., 0., 0
    changed = 0.
    history = []
    peak_gib = 0.
    if saved is not None:
        net.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        step, micro, best, changed = (saved[k] for k in ("step","micro","best","changed"))
        micro = min(initial_micro, micro)
        train_seconds, warm_seconds, warm_count = (saved[k] for k in ("train_seconds","warm_seconds","warm_count"))
        history = saved.get("validation_history", [])
        peak_gib = saved.get("peak_allocated_gib", 0.)
        if "rng_cpu" in saved:
            torch.set_rng_state(saved["rng_cpu"])
            torch.cuda.set_rng_state(saved["rng_cuda"], device)
        del saved
    # best.pt is written before last.pt. A restart between those writes must
    # retain the already published best model even if this epoch is replayed.
    if best_path.is_file():
        previous = torch.load(best_path, weights_only=True, map_location="cpu", mmap=True)
        if previous["config"] != config:
            raise ValueError("Best validation checkpoint belongs to a different run")
        best = previous["validation_huber"]
        del previous
    torch.cuda.reset_peak_memory_stats()
    def checkpoint(result=None):
        data = dict(config=config, step=step, micro=micro, best=best, changed=changed, train_seconds=train_seconds,
                    warm_seconds=warm_seconds, warm_count=warm_count, model=net.state_dict(), optimizer=opt.state_dict(),
                    rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(device),
                    validation_history=history, peak_allocated_gib=max(peak_gib, torch.cuda.max_memory_allocated()/2**30))
        if result is not None:
            data["result"] = result
        atomic_write(ck, lambda path: torch.save(data, path))
    while step < goal:
        epoch, start = divmod(step, steps_epoch)
        batches = epoch_batches(trainset, epoch)[start:]
        # Large visual batches would exhaust the small /dev/shm common inside pods.
        # A single-process pinned loader also retains mmap's OS page-cache sharing.
        loader = DataLoader(trainset, batch_sampler=batches, num_workers=workers, pin_memory=True,
                            generator=torch.Generator().manual_seed(7 + epoch),
                            **({"prefetch_factor": 1} if workers else {}))
        for batch in prefetched(loader):
            torch.cuda.synchronize()
            before = time.perf_counter()
            old = net.value_mlp.weight.detach().clone()
            values, norm, micro = update(net, opt, batch, micro, step % 2 == 1, device)
            changed = max(changed, (net.value_mlp.weight.detach() - old).abs().max().item())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - before
            step += 1
            train_seconds += elapsed
            if step > 10:
                warm_seconds += elapsed
                warm_count += 1
            if step == 1 or step % 10 == 0:
                print(json.dumps(dict(update=step, target=goal, visual=values[0], huber=values[1], rank=values[2],
                                      grad_norm=norm, microbatch=micro, seconds=elapsed,
                                      peak_gib=torch.cuda.max_memory_allocated()/2**30)), flush=True)
            if step == goal or step % steps_epoch == 0:
                metrics = validate(net, valset, micro, device, limit=4 if pilot else None, with_rank=True)
                history.append(dict(epoch=epoch + 1, update=step, **metrics))
                metric = metrics["validation_huber"]
                if best is None or metric < best:
                    best = metric
                    atomic_write(best_path, lambda p: torch.save(dict(config=config, step=step,
                                **metrics, model=net.state_dict()), p))
                print(json.dumps(dict(update=step, **metrics, best=best)), flush=True)
            if step == goal:
                break
            if step % steps_epoch == 0 or step % (25 if pilot else 200) == 0:
                checkpoint()
    if not changed > 0:
        raise ValueError("No recorded progress-head parameter update")
    # Old final checkpoints predate validation history and the embedded report.
    if not history or history[-1]["update"] != step:
        metrics = validate(net, valset, micro, device, limit=4 if pilot else None, with_rank=True)
        history.append(dict(epoch=math.ceil(step / steps_epoch), update=step, **metrics))
    result = dict(status="COMPLETE", config=config, updates=step, microbatch=micro, train_windows=len(trainset),
                  validation_windows=len(valset), best_validation_huber=best, maximum_head_update=changed,
                  validation_history=history, last_validation_rank=history[-1]["validation_rank"],
                  training_compute_seconds=train_seconds, warm_seconds_per_update=warm_seconds/max(1,warm_count),
                  peak_allocated_gib=max(peak_gib, torch.cuda.max_memory_allocated()/2**30), gpu=torch.cuda.get_device_name(0))
    checkpoint(result)
    save_json(summary, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["long100","joint40"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--microbatch", type=int, choices=[1,2,4,8,16,32], default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--job-dir", type=Path, default=JOB, help="Run directory containing the stage's features.json")
    a = p.parse_args()
    if a.epochs < 1 or a.workers < 0:
        p.error("Epochs must be positive and workers nonnegative")
    train(a.stage, a.epochs, a.microbatch, a.workers, job=a.job_dir)
