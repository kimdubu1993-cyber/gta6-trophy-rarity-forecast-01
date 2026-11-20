"""Forecast GTA VI trophy earn rates at week one (v3).

Earner/player model with a dilution prior: a refresh of Exophase's page
brings in many new tracked players who hold fewer trophies, so rates fall
roughly in proportion to the player-count jump.  When the recorded history
is flat (no refresh seen yet) the median is the last rate scaled by a
fixed dilution factor; when the history moves, earner and player growth
are extrapolated separately in log space and the median is the ratio of
the two extrapolations.  Quantiles are normal on the logit scale.
Standard library only, deterministic, no network.
"""

import json
import math
import os
from datetime import datetime

Z_VALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)
EXPECTED_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
DILUTION = 0.20


def clamp_rate(value):
    """Clamp a percent into the logit-safe range [0.01, 99.99]."""
    if value < 0.01:
        return 0.01
    if value > 99.99:
        return 99.99
    return value


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round4(value):
    """Round to 4 decimals (half up); return an int when whole."""
    stepped = math.floor(value * 10000.0 + 0.5) / 10000.0
    whole = int(stepped)
    if stepped == whole:
        return whole
    return stepped


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least-squares line y ~ slope * x + intercept; returns (slope, intercept)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def sample_sd(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def forecast_case(case):
    if case["levels"] != EXPECTED_LEVELS:
        raise ValueError("unexpected levels: %r" % (case["levels"],))
    history = case["history"]
    if not history:
        raise ValueError("empty history")
    trophies = case["trophies"]
    n_troph = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0

    start = parse_time(history[0]["at"])
    times = []
    for read in history:
        delta = parse_time(read["at"]) - start
        times.append(delta.total_seconds() / 86400.0)
    u_last = times[-1]
    u_target = u_last + horizon_days

    counts = [read["players_tracked"] for read in history]
    p_last = max(counts[-1], 1)

    # Carry each trophy's last seen rate forward over null reads.
    carried = []  # carried[j][i] is trophy j's rate at read i, or None
    last_rate = []
    for j in range(n_troph):
        series = []
        current = None
        for read in history:
            raw = read["rates"][j]
            if raw is not None:
                current = raw
            series.append(current)
        carried.append(series)
        last_rate.append(series[-1])

    # Frozen when every player count is identical and every trophy's
    # observed rate never changes (nulls ignored). One read is frozen.
    frozen = all(c == counts[0] for c in counts)
    if frozen:
        for j in range(n_troph):
            seen = [read["rates"][j] for read in history
                    if read["rates"][j] is not None]
            if seen and any(r != seen[0] for r in seen):
                frozen = False
                break

    # Earner counts from carried rates.
    earners = []
    for j in range(n_troph):
        earners.append([
            (carried[j][i] * counts[i] / 100.0) if carried[j][i] is not None
            else None
            for i in range(len(history))
        ])

    # --- Branch B shared fits (computed up front; used only if moving) ---
    # Player growth: OLS of ln(P) on days since the first read.
    log_p = [math.log(max(c, 1)) for c in counts]
    slope_p, _ = ols(times, log_p)
    slope_p = min(3.0, max(0.0, slope_p))
    player_mult = math.exp(0.7 * slope_p * horizon_days)
    player_mult = max(1.0, min(500.0, player_mult))

    raw_slope = [None] * n_troph
    usable_n = [0] * n_troph
    resid_sd = [0.30] * n_troph
    for j in range(n_troph):
        xs = []
        ys = []
        for i in range(len(history)):
            if earners[j][i] is None:
                continue
            xs.append(math.log(1.0 + times[i]))
            ys.append(math.log(earners[j][i] + 0.5))
        usable_n[j] = len(xs)
        if len(xs) < 2:
            raw_slope[j] = 0.0
            resid_sd[j] = 0.30
            continue
        a_j, b_j = ols(xs, ys)
        raw_slope[j] = a_j
        resid = [y - (b_j + a_j * x) for x, y in zip(xs, ys)]
        sd = sample_sd(resid)
        resid_sd[j] = min(0.80, max(0.15, sd))

    present_slopes = [a for a in raw_slope if a is not None]
    global_mean = (sum(present_slopes) / len(present_slopes)
                   if present_slopes else 0.0)

    rows = []
    for j, trophy in enumerate(trophies):
        rate = last_rate[j]
        if rate is not None and rate >= 100:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        if rate is None:
            median = 0.01
            spread = 2.0
        elif frozen:
            median = clamp_rate(rate * DILUTION)
            e_last = earners[j][-1]
            spread = math.sqrt(
                0.35 ** 2
                + (0.5 * math.sqrt(horizon_days)) ** 2
                + (0.5 / math.sqrt(e_last + 1.0)) ** 2
            )
            spread = min(spread, 2.5)
        else:
            tier = trophy["tier"]
            same_tier = [raw_slope[k] for k in range(n_troph)
                         if trophies[k]["tier"] == tier
                         and raw_slope[k] is not None]
            if len(same_tier) > 1:
                tier_mean = sum(same_tier) / len(same_tier)
            else:
                tier_mean = global_mean
            n_j = usable_n[j]
            a_star = (n_j * raw_slope[j] + 3.0 * tier_mean) / (n_j + 3.0)
            a_star = min(2.0, max(0.0, a_star))
            growth = ((1.0 + u_target) / (1.0 + u_last)) ** a_star
            growth = max(1.0, min(200.0, growth))
            if rate == 0:
                median = 0.01
            else:
                median = clamp_rate(
                    100.0 * earners[j][-1] * growth / (p_last * player_mult)
                )
            root_d = math.sqrt(horizon_days)
            spread = math.sqrt(
                (0.35 + 0.25 * root_d) ** 2
                + resid_sd[j] ** 2
                + (0.5 * root_d) ** 2
                + 0.6 ** 2
            )
            spread = min(spread, 2.5)

        centre = logit(median)
        quantiles = []
        for z in Z_VALUES:
            q = round4(inv_logit(centre + z * spread))
            if q < 0.01:
                q = 0.01
            elif q > 100:
                q = 100
            quantiles.append(q)
        # Enforce non-decreasing order with a cumulative-max pass.
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})

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
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
