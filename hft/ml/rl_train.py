"""PPO search for the reinforcement-learning trader (16 pre-registered configurations).

Implements docs/specs/2026-11-11-rl-daytrader-design.md: train on 2016-2021, keep the
best checkpoint by the 2022 inner validation, score the three-seed ensemble on
2023-2024 against holding and the baselines. The test years are never loaded here.

    python -m hft.ml.rl_train --data data/ml --out artifacts/rl-search --device cuda
    python -m hft.ml.rl_train --data data/ml --out artifacts/rl-search --summarize
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluate import report
from .rl_env import (
    EPISODE_DAYS,
    MAX_STEPS_A,
    DailyEnv,
    IntradayEnv,
    build_daily,
    build_intraday,
)

PRIOR_TRIALS = 108  # ML day-trader variants before this search
YEARS = {"train": (2016, 2021), "inner": (2022, 2022), "validation": (2023, 2024)}
NEWS_VARIANT = Path("artifacts/ml-news-search/7f7b87ac489e/metrics-validation.json")
SETTINGS = {
    "A": {"decisions": 50_000_000, "gamma": 0.999},
    "B": {"decisions": 20_000_000, "gamma": 0.99},
}
PPO = {
    "envs": 4096,
    "epochs": 4,
    "minibatches": 8,
    "lr": 2.5e-4,
    "clip": 0.2,
    "lambda": 0.95,
    "entropy": (0.01, 0.001),
    "value": 0.5,
    "grad_norm": 0.5,
    "eval_every": 10,
    "cost_range": (1.0, 3.0),
}


def grid():
    """The 16 pre-registered configurations, in a fixed order (trials 109-124)."""
    out = []
    for track, encoder, rich, reward in itertools.product(
        ("A", "B"), ("conv", "gru"), (False, True), ("log", "dsr")
    ):
        name = f"{track}-{encoder}-{'rich' if rich else 'base'}-{reward}"
        out.append({"id": name, "track": track, "encoder": encoder, "rich": rich, "reward": reward})
    return out


class Policy(nn.Module):
    """Encoder over the window, joined with the other inputs; policy and value heads."""

    def __init__(self, channels, extra, encoder, width=64):
        super().__init__()
        self.kind = encoder
        if encoder == "conv":
            self.encoder = nn.Sequential(
                nn.Conv1d(channels, width, 5, padding=2),
                nn.ReLU(),
                nn.Conv1d(width, width, 5, padding=2),
                nn.ReLU(),
            )
            encoded = 2 * width
        else:
            self.encoder = nn.GRU(channels, width, batch_first=True)
            encoded = width
        self.body = nn.Sequential(
            nn.Linear(encoded + extra, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh()
        )
        self.policy = nn.Linear(128, 2)
        self.value = nn.Linear(128, 1)
        nn.init.orthogonal_(self.policy.weight, 0.01)

    def forward(self, window, extra, mask):
        window = window.float()
        if self.kind == "conv":
            h = self.encoder(window.transpose(1, 2))
            h = torch.cat([h[..., -1], h.mean(-1)], dim=-1)
        else:
            _, last = self.encoder(window)
            h = last[-1]
        z = self.body(torch.cat([h, extra], dim=-1))
        logits = self.policy(z).masked_fill(~mask, -1e9)
        return logits, self.value(z).squeeze(-1)


def _year_mask(dates, split):
    low, high = YEARS[split]
    years = np.asarray(dates).astype("datetime64[Y]").astype(int) + 1970
    return (years >= low) & (years <= high)


def prepare(inputs, config):
    """Environment tensors per split (train, inner, validation); never later years."""
    frame = inputs.frame
    out = {}
    if config["track"] == "A":
        data, stats = build_intraday(inputs, _year_mask(frame.dates, "train"), rich=config["rich"])
        out["train"] = data
        for split in ("inner", "validation"):
            out[split], _ = build_intraday(
                inputs, _year_mask(frame.dates, split), rich=config["rich"], stats=stats
            )
    else:
        from dataclasses import replace

        data, _ = build_daily(inputs, np.datetime64("2024-12-31"), rich=config["rich"])
        for split in ("train", "inner", "validation"):
            mask = _year_mask(data.dates, split)
            # Episodes end where `valid` ends, so no split's episode runs into the next;
            # look-back windows may still read earlier days.
            out[split] = (replace(data, valid=data.valid & mask[:, None]), mask)
    return out


# Rollouts and PPO -----------------------------------------------------------------


def _env(config, tensors, device):
    cls = IntradayEnv if config["track"] == "A" else DailyEnv
    return cls(tensors, reward=config["reward"], device=device)


def _reset_train(env, config, rows, n, rng, device):
    low, high = PPO["cost_range"]
    cost = torch.as_tensor(rng.uniform(low, high, n), dtype=torch.float32, device=device)
    if config["track"] == "A":
        items = torch.as_tensor(rng.choice(rows, n), device=device)
        return env.reset(items, cost)
    # Track B: random (symbol, start) with a tradable start inside the training years.
    starts = rows[rng.integers(0, len(rows), n)]
    symbol = torch.as_tensor(starts[:, 1], device=device)
    start = torch.as_tensor(starts[:, 0], device=device)
    return env.reset(symbol, start, EPISODE_DAYS, cost)


def _train_rows(config, data):
    if config["track"] == "A":
        return np.arange(len(data.symbol))
    daily, mask = data
    valid = daily.valid & mask[:, None]
    return np.argwhere(valid)  # (day, symbol)


def train_seed(config, prepared, device, seed, *, decisions=None, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    track = config["track"]
    settings = SETTINGS[track]
    decisions = decisions or settings["decisions"]
    if track == "A":
        train_t = prepared["train"].to(device)
        rows = _train_rows(config, prepared["train"])
    else:
        daily, mask = prepared["train"]
        train_t = daily.to(device)
        rows = _train_rows(config, prepared["train"])
    env = _env(config, train_t, device)
    steps = MAX_STEPS_A if track == "A" else EPISODE_DAYS
    n = PPO["envs"]
    window_shape, extra_dim = env.obs_window, env.obs_extra
    policy = Policy(window_shape[1], extra_dim, config["encoder"]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=PPO["lr"], eps=1e-5)
    iterations = max(1, decisions // (n * steps))
    best, best_score, history = None, -np.inf, []
    started = time.time()
    for it in range(iterations):
        frac = it / max(iterations - 1, 1)
        for group in optimizer.param_groups:
            group["lr"] = PPO["lr"] * (1 - frac)
        entropy_coef = PPO["entropy"][0] + frac * (PPO["entropy"][1] - PPO["entropy"][0])
        window, extra, mask = _reset_train(env, config, rows, n, rng, device)
        buf = {
            k: []
            for k in ("window", "extra", "mask", "action", "logp", "value", "reward", "active")
        }
        with torch.no_grad():
            for _ in range(steps):
                logits, value = policy(window, extra, mask)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                reward, _, active = env.step(action)
                for key, val in (
                    ("window", window.half()),
                    ("extra", extra),
                    ("mask", mask),
                    ("action", action),
                    ("logp", dist.log_prob(action)),
                    ("value", value),
                    ("reward", reward),
                    ("active", active),
                ):
                    buf[key].append(val)
                window, extra, mask = env.observe()
        data = {k: torch.stack(v) for k, v in buf.items()}  # (T, N, ...)
        active = data["active"].float()
        advantages = torch.zeros_like(data["reward"])
        last = torch.zeros(n, device=device)
        for t in reversed(range(steps)):
            next_value = data["value"][t + 1] * active[t + 1] if t + 1 < steps else 0.0
            delta = data["reward"][t] + settings["gamma"] * next_value - data["value"][t]
            nxt = active[t + 1] if t + 1 < steps else torch.zeros(n, device=device)
            last = delta + settings["gamma"] * PPO["lambda"] * nxt * last
            advantages[t] = last * active[t]
        returns = advantages + data["value"]
        keep = data["active"].reshape(-1)
        flat = {k: v.reshape(steps * n, *v.shape[2:])[keep] for k, v in data.items()}
        flat_adv, flat_ret = advantages.reshape(-1)[keep], returns.reshape(-1)[keep]
        total = len(flat_adv)
        # Normalized once over the whole batch: a tiny minibatch has no usable spread.
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        for _ in range(PPO["epochs"]):
            order = torch.randperm(total, device=device)
            for idx in torch.tensor_split(order, PPO["minibatches"]):
                logits, value = policy(flat["window"][idx], flat["extra"][idx], flat["mask"][idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(flat["action"][idx])
                adv = flat_adv[idx]
                ratio = torch.exp(logp - flat["logp"][idx])
                clipped = torch.clamp(ratio, 1 - PPO["clip"], 1 + PPO["clip"])
                loss_pi = -torch.min(ratio * adv, clipped * adv).mean()
                loss_v = 0.5 * ((value - flat_ret[idx]) ** 2).mean()
                loss = loss_pi + PPO["value"] * loss_v - entropy_coef * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), PPO["grad_norm"])
                optimizer.step()
        if (it + 1) % PPO["eval_every"] == 0 or it == iterations - 1:
            inner = evaluate([policy.state_dict()], config, prepared, "inner", device)
            score = float(np.mean(inner["daily"]))
            history.append({"iteration": it + 1, "inner_mean_daily": score})
            log(
                f"{config['id']} seed {seed} it {it + 1}/{iterations} inner {score * 1e4:+.2f} bp/day "
                f"{time.time() - started:.0f}s"
            )
            if score > best_score:
                best_score = score
                best = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    return best, {"best_inner_mean_daily": best_score, "history": history, "iterations": iterations}


# Evaluation -----------------------------------------------------------------------


def _policies(states, config, env, device):
    out = []
    for state in states:
        p = Policy(env.obs_window[1], env.obs_extra, config["encoder"]).to(device)
        p.load_state_dict(state)
        p.eval()
        out.append(p)
    return out


def _act(policies, window, extra, mask):
    probs = sum(torch.softmax(p(window, extra, mask)[0], -1) for p in policies) / len(policies)
    return probs.argmax(-1)


def _run_intraday(env, items, cost, act, chunk=8192):
    """Day net log returns, trade flags and entry/exit steps for every item."""
    out = {"log": [], "trades": [], "entry": [], "exit": [], "steps": []}
    for start in range(0, len(items), chunk):
        part = items[start : start + chunk]
        window, extra, mask = env.reset(part, cost)
        for k in range(MAX_STEPS_A):
            env.step(act(k, window, extra, mask, part))
            window, extra, mask = env.observe()
        out["log"].append(env.log_value.cpu())
        out["trades"].append(env.trades.cpu())
        out["entry"].append(env.entry_k.cpu())
        out["exit"].append(env.exit_k.cpu())
        out["steps"].append(env.steps.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def _portfolio_intraday(data, run, symbols):
    """Equal slots in all tradable symbols: a slot's day return is its net return."""
    net = np.expm1(run["log"])
    daily = np.zeros(len(data.dates))
    np.add.at(daily, data.date, net)
    return daily / symbols


@torch.no_grad()
def evaluate(states, config, prepared, split, device, cost_multiple=1.0, baselines=False, seed=0):
    track = config["track"]
    if track == "A":
        data = prepared[split]
        tensors = data.to(device)
        env = _env(config, tensors, device)
        policies = _policies(states, config, env, device)
        items = torch.arange(len(data.symbol), device=device)
        symbols = len(np.unique(data.symbol)) if len(data.symbol) else 1

        def agent(k, window, extra, mask, part):
            return _act(policies, window, extra, mask)

        run = _run_intraday(env, items, cost_multiple, agent)
        traded = run["trades"] > 0
        out = {
            "daily": _portfolio_intraday(data, run, symbols),
            "trades": np.expm1(run["log"][traded]),
            "dates": [str(d) for d in data.dates],
        }
        if baselines:
            out["baselines"] = _intraday_baselines(
                env, data, items, run, cost_multiple, symbols, seed
            )
        return out
    daily_data, mask = prepared[split]
    tensors = daily_data.to(device)
    env = _env(config, tensors, device)
    policies = _policies(states, config, env, device)
    return _evaluate_daily(env, daily_data, mask, cost_multiple, policies, baselines, seed)


def _intraday_baselines(env, data, items, run, cost, symbols, seed):
    rng = np.random.default_rng(seed)
    always = _run_intraday(
        env,
        items,
        cost,
        lambda k, w, e, m, part: torch.ones(len(part), dtype=torch.long, device=w.device),
    )
    # Random entries with the agent's own trade count and holding times.
    entry = np.full(len(items), -1)
    exit_ = np.full(len(items), -1)
    for i in np.flatnonzero(run["trades"] > 0):
        steps = int(run["steps"][i])
        hold = (run["exit"][i] if run["exit"][i] >= 0 else steps - 1) - run["entry"][i]
        latest = max(steps - 1 - hold - 3, 0)
        entry[i] = rng.integers(0, latest + 1)
        exit_[i] = entry[i] + hold if run["exit"][i] >= 0 else -1
    entry_t = torch.as_tensor(entry, device=items.device)
    exit_t = torch.as_tensor(exit_, device=items.device)

    def planned(k, window, extra, mask, part):
        idx = part
        e, x = entry_t[idx], exit_t[idx]
        return ((e >= 0) & (k >= e) & ((x < 0) | (k < x))).long()

    random = _run_intraday(env, items, cost, planned)
    return {
        "buy_open_sell_close": _portfolio_intraday(data, always, symbols),
        "random_matched": _portfolio_intraday(data, random, symbols),
    }


def _evaluate_daily(env, data, mask, cost, policies, baselines, seed):
    """One long episode per symbol across the split; equal slots in all symbols."""
    rows = np.flatnonzero(mask)
    symbols = np.flatnonzero(data.valid[rows].any(axis=0))
    first = np.array([rows[np.argmax(data.valid[rows, s])] for s in symbols])
    length = int(rows[-1] - first.min() + 1)
    device = env.device
    total_symbols = int(np.asarray(data.valid).shape[1])

    def run(choose):
        window, extra, m = env.reset(
            torch.as_tensor(symbols, device=device),
            torch.as_tensor(first, device=device),
            length,
            cost,
            record=True,
        )
        for _ in range(length):
            env.step(choose(window, extra, m))
            window, extra, m = env.observe()
            if not env.alive.any():
                break
        changes = torch.stack(env.daily).cpu().numpy()  # (steps, symbols)
        daily = np.zeros(len(rows))
        for j, s0 in enumerate(first):
            offset = int(np.searchsorted(rows, s0))
            series = np.expm1(changes[:, j])[: len(rows) - offset]
            daily[offset : offset + len(series)] += series
        trades = [(int(symbols[i]), k, n, v) for i, k, n, v in env.completed]
        return daily / total_symbols, trades

    daily, trades = run(lambda w, e, m: _act(policies, w, e, m))
    out = {
        "daily": daily,
        "trades": np.expm1([t[3] for t in trades]),
        "dates": [str(d) for d in data.dates[rows]],
    }
    if baselines:
        rng = np.random.default_rng(seed)
        always, _ = run(lambda w, e, m: torch.ones(len(w), dtype=torch.long, device=w.device))
        # Random trades with the agent's holding times, at random tradable starts.
        values = []
        opens = data.open
        last_row = rows[-1]
        for symbol, _, hold, _ in trades:
            usable = [
                d
                for d in rows
                if d + hold <= last_row and data.valid[d : d + hold + 1, symbol].all()
            ]
            if not usable:
                continue
            d = int(rng.choice(usable))
            p_in, p_out = float(opens[d, symbol]), float(opens[d + hold, symbol])

            def side(p, s=symbol):
                return cost * (max(float(data.cost_bps[s]), 0.5 / p * 100) + 1.0) / 1e4

            values.append((d, np.log(p_out / p_in) + np.log1p(-side(p_out)) - np.log1p(side(p_in))))
        random = np.zeros(len(rows))
        for d, v in values:
            random[np.searchsorted(rows, d)] += np.expm1(v) / total_symbols
        out["baselines"] = {"hold_every_session": always, "random_matched": random}
    return out


def pbo(matrix, blocks=16):
    """Probability of backtest overfitting (CSCV, Bailey et al.): `matrix` is (days,
    configurations) of daily returns over the selection period."""
    matrix = np.asarray(matrix, float)
    parts = np.array_split(np.arange(len(matrix)), blocks)
    n = matrix.shape[1]

    def sharpe(rows):
        x = matrix[rows]
        return x.mean(0) / (x.std(0) + 1e-12)

    below = total = 0
    for chosen in itertools.combinations(range(blocks), blocks // 2):
        inside = np.concatenate([parts[i] for i in chosen])
        outside = np.concatenate([parts[i] for i in range(blocks) if i not in chosen])
        best = int(np.argmax(sharpe(inside)))
        oos = sharpe(outside)
        rank = (oos < oos[best]).sum() + 1
        w = rank / (n + 1)
        below += np.log(w / (1 - w)) <= 0
        total += 1
    return below / total


# Search ---------------------------------------------------------------------------


def run_config(config, inputs, out, device, *, seeds=3, decisions=None, log=print):
    target = Path(out) / config["id"]
    if (target / "result.json").exists():
        log(f"{config['id']}: done already")
        return json.loads((target / "result.json").read_text())
    target.mkdir(parents=True, exist_ok=True)
    prepared = prepare(inputs, config)
    states, seed_info = [], []
    for seed in range(seeds):
        state, info = train_seed(config, prepared, device, seed, decisions=decisions, log=log)
        torch.save(state, target / f"seed{seed}.pt")
        states.append(state)
        single = evaluate([state], config, prepared, "validation", device)
        info["validation_total"] = float(np.prod(1 + single["daily"]) - 1)
        seed_info.append(info)
    result = {"config": config, "seeds": seed_info}
    for multiple, key in ((1.0, "validation"), (3.0, "validation_3x")):
        ev = evaluate(
            states, config, prepared, "validation", device, multiple, baselines=multiple == 1.0
        )
        holding = _holding_series(inputs, ev["dates"])
        result[key] = report(ev["daily"], holding, ev["trades"], trials=PRIOR_TRIALS + len(grid()))
        np.save(target / f"{key}-daily.npy", ev["daily"])
        if multiple == 1.0:
            result["dates"] = [ev["dates"][0], ev["dates"][-1]]
            result["baselines"] = {
                name: float(np.prod(1 + series) - 1) for name, series in ev["baselines"].items()
            }
            for name, series in ev["baselines"].items():
                np.save(target / f"baseline-{name}.npy", series)
    from .train import code_commit

    result["code"] = code_commit()
    result["data_hash"] = inputs.data_hash
    (target / "result.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    log(f"{config['id']}: validation {result['validation']['strategy']['total']:+.1%}")
    return result


def _holding_series(inputs, dates):
    from .train import _holding

    frame = inputs.frame
    days = np.isin(frame.dates, np.array(dates, dtype="datetime64[D]"))
    return _holding(inputs.root, inputs.symbols, frame, days, inputs.trainable)


def summarize(out):
    """Pass rule of the design over all finished configurations."""
    out = Path(out)
    results = {
        c["id"]: json.loads((out / c["id"] / "result.json").read_text())
        for c in grid()
        if (out / c["id"] / "result.json").exists()
    }
    if not results:
        raise ValueError("no finished configurations")
    ids = list(results)
    matrix = np.stack([np.load(out / i / "validation-daily.npy") for i in ids], axis=1)
    overfit = pbo(matrix) if len(ids) > 1 else None
    news = json.loads(NEWS_VARIANT.read_text()) if NEWS_VARIANT.exists() else None
    best = max(ids, key=lambda i: results[i]["validation"]["strategy"]["sharpe"])
    r = results[best]
    total = r["validation"]["strategy"]["total"]
    checks = {
        "edge_over_holding_interval_above_zero": r["validation"]["beats_holding"],
        "beats_baselines": all(total > v for v in r["baselines"].values()) and total > 0,
        "beats_news_model": news is None or total > news["strategy"]["total"],
        "deflated_sharpe_at_least_0.95": r["validation"]["deflated_sharpe"] >= 0.95,
        "pbo_below_0.2": overfit is not None and overfit < 0.2,
        "positive_at_3x_costs": r["validation_3x"]["strategy"]["total"] > 0,
    }
    summary = {
        "finished": len(ids),
        "best": best,
        "best_validation": r["validation"],
        "best_baselines": r["baselines"],
        "news_model_total": news["strategy"]["total"] if news else None,
        "pbo": overfit,
        "checks": checks,
        "passes": all(checks.values()) and len(ids) == len(grid()),
        "table": {
            i: {
                "total": results[i]["validation"]["strategy"]["total"],
                "sharpe": results[i]["validation"]["strategy"]["sharpe"],
                "trades": results[i]["validation"]["trades"],
                "per_trade": results[i]["validation"]["per_trade"],
                "total_3x": results[i]["validation_3x"]["strategy"]["total"],
                "holding": results[i]["validation"]["holding"]["total"],
            }
            for i in ids
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/ml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--configs", nargs="*", help="configuration ids (default: all 16)")
    parser.add_argument("--symbols", nargs="*", help="smoke runs only: a subset of symbols")
    parser.add_argument("--decisions", type=int, help="smoke runs only: decisions per seed")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--envs", type=int, help="smoke runs only: parallel episodes")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize:
        print(json.dumps(summarize(args.out), indent=2, default=float))
        return 0
    if (args.symbols or args.decisions or args.envs) and "smoke" not in args.out.name:
        parser.error("--symbols/--decisions/--envs are for smoke runs: use an --out named *smoke*")
    if args.envs:
        PPO["envs"] = args.envs
    from .train import load_inputs

    symbols = args.symbols and sorted(set(args.symbols) | {"SPY", "QQQ"})
    inputs = load_inputs(args.data, symbols)
    wanted = set(args.configs or [])
    for config in grid():
        if wanted and config["id"] not in wanted:
            continue
        run_config(
            config, inputs, args.out, args.device, seeds=args.seeds, decisions=args.decisions
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
