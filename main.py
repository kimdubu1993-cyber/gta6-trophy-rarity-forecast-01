"""Forecast GTA VI trophy earn rates at week one.

Reads every case in MAKEFUN_INPUT_DIR (default /input) and writes one
forecast per case to MAKEFUN_OUTPUT_DIR (default /output).

Model sketch: rates are diluted by each refresh of the tracked-player
pool, so a frozen history is forecast as last rate times a dilution
factor with a logit spread that grows with the horizon and shrinks
with the earner count. A moving history fits log-time growth curves
to the player count and to each trophy's earner count separately
(the player curve saturates in log time), shrinks each trophy's
earner slope toward its tier mean, and combines the fit residual
with horizon terms into the spread.
"""

import json
import math
import os
from datetime import datetime

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
EXPECTED_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(p):
    c = clamp_pct(p)
    return math.log(c / (100.0 - c))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    v = math.floor(x * 10000.0 + 0.5) / 10000.0
    if v == int(v):
        return int(v)
    return v


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ols_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def quantiles(median, spread):
    base = logit(median)
    out = []
    for z in Z:
        q = round4(inv_logit(base + z * spread))
        q = min(100.0, max(0.01, q))
        out.append(q)
    running = out[0]
    fixed = []
    for q in out:
        if q < running:
            q = running
        running = q
        fixed.append(q)
    return fixed


def forecast_case(case):
    levels = case.get("levels")
    if levels != EXPECTED_LEVELS:
        raise ValueError("unexpected levels: %r" % (levels,))
    history = case.get("history") or []
    if not history:
        raise ValueError("empty history")
    trophies = case["trophies"]
    n_troph = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0

    times = [parse_time(read["at"]) for read in history]
    t0 = times[0]
    u = [(t - t0).total_seconds() / 86400.0 for t in times]
    u_last = u[-1]
    u_target = u_last + horizon_days

    players = [float(read["players_tracked"]) for read in history]
    p_last = max(players[-1], 1.0)

    # Carry each trophy's last seen rate forward over null reads.
    rates = []
    for j in range(n_troph):
        col = []
        last_seen = None
        for read in history:
            v = read["rates"][j]
            if v is not None:
                last_seen = float(v)
            col.append(last_seen)
        rates.append(col)

    last_rate = [col[-1] for col in rates]
    earners_last = []
    for j in range(n_troph):
        if last_rate[j] is None:
            earners_last.append(None)
        else:
            earners_last.append(last_rate[j] * p_last / 100.0)

    frozen = all(p == players[0] for p in players)
    if frozen:
        for col in rates:
            seen = {v for v in col if v is not None}
            if len(seen) > 1:
                frozen = False
                break

    rows = []
    if frozen:
        for j in range(n_troph):
            if last_rate[j] is not None and last_rate[j] >= 100.0:
                rows.append({"id": trophies[j]["id"], "q": [100, 100, 100, 100, 100]})
                continue
            if last_rate[j] is None:
                rows.append({"id": trophies[j]["id"], "q": quantiles(0.01, 2.0)})
                continue
            med = clamp_pct(last_rate[j] * 0.23)
            e_last = earners_last[j]
            inner = (
                0.35 ** 2
                + (0.5 * math.sqrt(horizon_days)) ** 2
                + (0.5 / math.sqrt(e_last + 1.0)) ** 2
            )
            spread = min(2.5, 0.6 * math.sqrt(inner))
            rows.append({"id": trophies[j]["id"], "q": quantiles(med, spread)})
        return {"trophies": rows}

    # Moving history: saturating log-time player growth.
    log_u = [math.log(1.0 + t) for t in u]
    log_p = [math.log(p) if p > 0 else 0.0 for p in players]
    b_player, _ = ols_slope(log_u, log_p)
    b_player = min(3.0, max(0.0, b_player))
    ratio = (1.0 + u_target) / (1.0 + u_last)
    m_player = ratio ** b_player
    m_player = max(1.0, min(500.0, m_player))

    # Per-trophy earner growth in log time, shrunk toward the tier mean.
    slopes = []
    resid_sd = []
    usable_counts = []
    for j in range(n_troph):
        xs = []
        ys = []
        for i, read in enumerate(history):
            if rates[j][i] is None:
                continue
            e = rates[j][i] * players[i] / 100.0
            xs.append(log_u[i])
            ys.append(math.log(e + 0.5))
        usable_counts.append(len(xs))
        if len(xs) < 2:
            slopes.append(0.0)
            resid_sd.append(0.30)
            continue
        a, b = ols_slope(xs, ys)
        slopes.append(a)
        if len(xs) >= 3:
            resid = [y - (b + a * x) for x, y in zip(xs, ys)]
            mean_r = sum(resid) / len(resid)
            var = sum((r - mean_r) ** 2 for r in resid) / (len(resid) - 1)
            sd = math.sqrt(var) if var > 0 else 0.0
        else:
            sd = 0.0
        resid_sd.append(min(0.80, max(0.15, sd)))

    tiers = [t["tier"] for t in trophies]
    global_mean = sum(slopes) / len(slopes) if slopes else 0.0
    tier_mean = {}
    for tier in set(tiers):
        members = [slopes[k] for k in range(n_troph) if tiers[k] == tier]
        tier_mean[tier] = sum(members) / len(members) if members else global_mean

    sqrt_d = math.sqrt(horizon_days)
    for j in range(n_troph):
        tid = trophies[j]["id"]
        if last_rate[j] is not None and last_rate[j] >= 100.0:
            rows.append({"id": tid, "q": [100, 100, 100, 100, 100]})
            continue
        if last_rate[j] is None:
            rows.append({"id": tid, "q": quantiles(0.01, 2.0)})
            continue
        tier = tiers[j]
        members = sum(1 for k in range(n_troph) if tiers[k] == tier)
        anchor = tier_mean[tier] if members > 1 else global_mean
        n_j = usable_counts[j]
        a_star = (n_j * slopes[j] + 3.0 * anchor) / (n_j + 3.0)
        a_star = min(2.0, max(0.0, a_star))
        g = ratio ** a_star
        g = max(1.0, min(200.0, g))
        if last_rate[j] == 0:
            med = 0.01
        else:
            med = clamp_pct(100.0 * earners_last[j] * g / (p_last * m_player))
        sd = resid_sd[j]
        spread = math.sqrt(
            (0.35 + 0.25 * sqrt_d) ** 2
            + sd ** 2
            + (0.5 * sqrt_d) ** 2
            + 0.6 ** 2
        )
        spread = min(2.5, spread)
        rows.append({"id": tid, "q": quantiles(med, spread)})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
