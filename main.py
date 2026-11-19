#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at the case target.

Per trophy, fit a saturating log-time growth curve to the logit of the
recorded earn rates, y = a + b * ln(1 + t), and emit five quantiles of the
rate at the target. The spread combines a random-walk term, the fit
residual, and a fixed floor for refresh / re-basing uncertainty: the
history Exophase shows is only as fresh as its last refresh, so a flat
history must not be read as zero uncertainty.

Standard library only. Deterministic: no randomness, no wall clock.
Reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR
(default /output), one <id>.out per <id>.in, sorted by file name.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z_VALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)
REFRESH_FLOOR = 0.8
# Raised from 0.6 after the practice runs: the house's first refresh
# (players tracked 638 -> 3,868, a 6.06x jump) cut the mean earn rate
# from 2.687% to 2.010%, a x0.748 dilution, about -0.29 logit on
# average -- and rare trophies dilute further, since their holder count
# grows slower than the player base, plausibly -0.6 to -1.1 logit.
# A 0.6 floor gives a 90-minute spread of only 0.613, so its 90% band
# (+/-1.01 logit) barely covers that tail. Pinball loss is asymmetric
# here: a trophy outside the band costs ~1.9 per logit at the outer
# quantiles, while a slightly wider band costs only ~0.1 per logit
# when the rate lands inside it. 0.8 lifts the 90-minute spread to
# 0.810 and the final's (6,510-minute horizon) spread from 1.22 to
# 1.33, where the random-walk term dominates -- cheap insurance.
MAX_SPREAD = 2.5
MAX_MOVE = 2.0
RATE_LO = 0.01
RATE_HI = 99.99
DEFAULT_MEDIAN_RATE = 1.0


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    if not isinstance(text, str):
        raise ValueError("not a timestamp string")
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def clamp_rate(rate):
    return min(max(float(rate), RATE_LO), RATE_HI)


def logit(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    value = min(max(value, -700.0), 700.0)
    return 100.0 / (1.0 + math.exp(-value))


def round_4(value):
    """Round to 4 decimals, half-up; emit whole numbers as ints."""
    rounded = math.floor(value * 10000.0 + 0.5) / 10000.0
    rounded = min(max(rounded, RATE_LO), RATE_HI)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def quantiles(median_logit, spread):
    out = []
    for z in Z_VALUES:
        out.append(round_4(inv_logit(median_logit + z * spread)))
    # Rounding and clamping are monotone, but enforce the contract.
    for k in range(1, len(out)):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return out


def random_walk_spread(horizon_minutes):
    try:
        horizon = float(horizon_minutes)
    except (TypeError, ValueError):
        horizon = 0.0
    if horizon < 0.0:
        horizon = 0.0
    return 0.5 * math.sqrt(horizon / 1440.0)


def fit_curve(pairs):
    """Least-squares fit of y = a + b * ln(1 + t) over (t_hours, y) pairs.

    Returns (a, b, residual_sd, used_fit). The fit is only used when
    there are at least 3 pairs with distinct t and non-constant y;
    otherwise b = 0 and a = the last y (persistence).
    """
    n = len(pairs)
    if n < 3:
        return pairs[-1][1] if pairs else logit(DEFAULT_MEDIAN_RATE), 0.0, 0.0, False
    xs = [math.log1p(t) for t, _ in pairs]
    ys = [y for _, y in pairs]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return ys[-1], 0.0, 0.0, False
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return ys[-1], 0.0, 0.0, False
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = mean_y - b * mean_x
    resid_sq = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    if n > 2:
        s_fit = math.sqrt(resid_sq / (n - 2))
    else:
        s_fit = 0.0
    return a, b, s_fit, True


def forecast_trophy(pairs, last_y, target_hours, horizon_minutes):
    """Median logit and spread for one trophy.

    pairs: valid (t_hours, y) history pairs, oldest first.
    last_y: logit of the last valid rate, or None when there is none.
    target_hours: hours from the first read's slot to the target.
    """
    rw = random_walk_spread(horizon_minutes)
    if last_y is None:
        median_logit = logit(DEFAULT_MEDIAN_RATE)
        spread = min(math.sqrt(rw * rw + REFRESH_FLOOR * REFRESH_FLOOR), MAX_SPREAD)
        return median_logit, spread
    a, b, s_fit, _used = fit_curve(pairs)
    median_logit = a + b * math.log1p(target_hours)
    move = median_logit - last_y
    if move > MAX_MOVE:
        median_logit = last_y + MAX_MOVE
    elif move < -MAX_MOVE:
        median_logit = last_y - MAX_MOVE
    spread = min(math.sqrt(rw * rw + s_fit * s_fit + REFRESH_FLOOR * REFRESH_FLOOR), MAX_SPREAD)
    return median_logit, spread


def persistence_rows(trophies, history, horizon_minutes):
    """Fallback forecast: last rate per trophy, random-walk spread only."""
    rw = random_walk_spread(horizon_minutes)
    rows = []
    for idx, trophy in enumerate(trophies):
        last_rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                last_rate = rates[idx]
                break
        if last_rate is None:
            median_logit = logit(DEFAULT_MEDIAN_RATE)
        else:
            median_logit = logit(last_rate)
        rows.append({"id": trophy["id"], "q": quantiles(median_logit, rw)})
    return rows


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_minutes = case.get("horizon_minutes", 0)
    try:
        target = parse_time(case["target"])
    except (KeyError, ValueError, TypeError):
        target = None

    if not history:
        # No reads at all: median 1.0% for every trophy, honest spread.
        rw = random_walk_spread(horizon_minutes)
        spread = min(math.sqrt(rw * rw + REFRESH_FLOOR * REFRESH_FLOOR), MAX_SPREAD)
        median_logit = logit(DEFAULT_MEDIAN_RATE)
        return [
            {"id": trophy["id"], "q": quantiles(median_logit, spread)}
            for trophy in trophies
        ]

    first_read = history[0]
    try:
        first_slot = parse_time(first_read.get("slot") or first_read.get("at"))
    except (ValueError, TypeError):
        first_slot = None
    if target is not None and first_slot is not None:
        target_hours = (target - first_slot).total_seconds() / 3600.0
    else:
        target_hours = 0.0
    if target_hours < 0.0:
        target_hours = 0.0

    # Valid (t_hours, y) pairs per trophy, t since the first read's slot.
    rows = []
    for idx, trophy in enumerate(trophies):
        pairs = []
        last_y = None
        for read in history:
            rates = read.get("rates") or []
            if idx >= len(rates):
                continue
            rate = rates[idx]
            if rate is None:
                continue
            try:
                rate_value = float(rate)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(rate_value):
                continue
            try:
                slot = parse_time(read.get("slot") or read.get("at"))
            except (ValueError, TypeError):
                continue
            if first_slot is None:
                continue
            t_hours = (slot - first_slot).total_seconds() / 3600.0
            if t_hours < 0.0:
                continue
            y = logit(rate_value)
            pairs.append((t_hours, y))
            last_y = y
        median_logit, spread = forecast_trophy(pairs, last_y, target_hours, horizon_minutes)
        rows.append({"id": trophy["id"], "q": quantiles(median_logit, spread)})
    return rows


def process_file(input_path, output_path):
    """Read one case, write one forecast; never raise for a bad case."""
    try:
        with open(input_path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        if not isinstance(case, dict):
            raise ValueError("case is not an object")
        trophies = case.get("trophies")
        if not isinstance(trophies, list) or not trophies:
            raise ValueError("case has no trophies")
        for trophy in trophies:
            if not isinstance(trophy, dict) or "id" not in trophy:
                raise ValueError("trophy row lacks an id")
        rows = forecast_case(case)
    except Exception:
        # Fall back to persistence on whatever we could salvage, so the
        # case still yields a valid file instead of a crash.
        try:
            with open(input_path, "r", encoding="utf-8") as handle:
                case = json.load(handle)
            trophies = case.get("trophies") if isinstance(case, dict) else None
            history = case.get("history") if isinstance(case, dict) else None
            horizon = case.get("horizon_minutes", 0) if isinstance(case, dict) else 0
            if not isinstance(trophies, list) or not trophies:
                return False
            if not isinstance(history, list):
                history = []
            rows = persistence_rows(trophies, history, horizon)
        except Exception:
            return False
    payload = {"trophies": rows}
    text = json.dumps(payload, separators=(",", ":")) + "\n"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return True


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    try:
        names = sorted(
            name for name in os.listdir(input_dir) if name.endswith(".in")
        )
    except OSError:
        return 0
    for name in names:
        stem = name[:-3]
        process_file(
            os.path.join(input_dir, name),
            os.path.join(output_dir, stem + ".out"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
