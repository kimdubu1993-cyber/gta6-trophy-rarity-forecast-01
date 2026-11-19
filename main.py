#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates at week one (v2).

For every trophy, fit a saturating log-time curve to the logit of its
earn rate over the case history, shrink the fitted slope toward the
global mean slope across trophies, and widen the spread with a large
refresh floor so a sudden Exophase recount (as on launch night, when
every rate but the one at 100% fell at once) stays inside the band.

Standard library only, deterministic, no network, no wall clock.
Reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR
(default /output), one <stem>.out per *.in file.
"""

import json
import math
import os
from datetime import datetime, timezone

Z_VALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)
RW_COEF = 0.5
REFRESH_FLOOR = 1.8
RESIDUAL_FLOOR = 0.25
SPREAD_CAP = 2.5
MOVE_CAP = 2.0
SHRINK_PRIOR = 3.0
NO_HISTORY_RATE = 1.0


def clamp_rate(rate):
    if rate is None:
        return None
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return min(99.99, max(0.01, value))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(logit_value):
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    exp_value = math.exp(logit_value)
    return 100.0 * exp_value / (1.0 + exp_value)


def round4(value):
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 10000.0
    return int(result) if float(result).is_integer() else result


def parse_time(text):
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def least_squares(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    return slope, intercept


def residual_sd(xs, ys, slope, intercept):
    n = len(xs)
    if n <= 2:
        return 0.0
    total = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return math.sqrt(total / (n - 2))


def forecast_trophy(pairs, target_x, horizon_minutes, last_logit,
                     global_slope):
    """pairs: list of (x, logit) history points for one trophy.
    Returns (median_logit, spread)."""
    horizon_days = max(0.0, horizon_minutes) / 1440.0
    walk = RW_COEF * math.sqrt(horizon_days)
    if last_logit is None:
        median = logit_of(NO_HISTORY_RATE)
        spread = min(SPREAD_CAP, math.sqrt(walk * walk
                                           + RESIDUAL_FLOOR ** 2
                                           + REFRESH_FLOOR ** 2))
        return median, spread
    if len(pairs) >= 3:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        distinct_times = len(set(xs)) >= 2
        varied_rates = max(ys) - min(ys) > 0.0
        fit = least_squares(xs, ys) if distinct_times and varied_rates else None
    else:
        fit = None
    if fit is not None:
        slope, intercept = fit
        weight = len(pairs)
        shrunk = (weight * slope + SHRINK_PRIOR * global_slope) / (
            weight + SHRINK_PRIOR)
        median = intercept + shrunk * target_x
        resid = residual_sd(xs, ys, slope, intercept)
    else:
        median = last_logit
        resid = 0.0
    if median > last_logit + MOVE_CAP:
        median = last_logit + MOVE_CAP
    elif median < last_logit - MOVE_CAP:
        median = last_logit - MOVE_CAP
    resid = max(resid, RESIDUAL_FLOOR)
    spread = min(SPREAD_CAP, math.sqrt(walk * walk + resid * resid
                                       + REFRESH_FLOOR ** 2))
    return median, spread


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon = case.get("horizon_minutes") or 0
    try:
        horizon = float(horizon)
    except (TypeError, ValueError):
        horizon = 0.0
    target = parse_time(case.get("target"))
    first_slot = None
    for read in history:
        if not isinstance(read, dict):
            continue
        slot = parse_time(read.get("slot")) or parse_time(read.get("at"))
        if slot is not None:
            first_slot = slot
            break
    per_trophy = []
    raw_slopes = []
    for index, trophy in enumerate(trophies):
        points = []
        last_logit = None
        for read in history:
            if not isinstance(read, dict):
                continue
            rates = read.get("rates") or []
            if index >= len(rates):
                continue
            rate = clamp_rate(rates[index])
            if rate is None:
                continue
            moment = parse_time(read.get("slot")) or parse_time(read.get("at"))
            if moment is None or first_slot is None:
                continue
            hours = (moment - first_slot).total_seconds() / 3600.0
            if hours < 0.0:
                continue
            points.append((math.log1p(hours), logit_of(rate)))
            last_logit = logit_of(rate)
        per_trophy.append((points, last_logit))
        if len(points) >= 3:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if len(set(xs)) >= 2 and max(ys) - min(ys) > 0.0:
                fit = least_squares(xs, ys)
                if fit is not None:
                    raw_slopes.append((fit[0], len(points)))
    if raw_slopes:
        total_weight = sum(w for _, w in raw_slopes)
        global_slope = sum(s * w for s, w in raw_slopes) / total_weight
    else:
        global_slope = 0.0
    if target is not None and first_slot is not None:
        target_hours = (target - first_slot).total_seconds() / 3600.0
        target_x = math.log1p(max(0.0, target_hours))
    else:
        target_x = 0.0
    rows = []
    for index, trophy in enumerate(trophies):
        points, last_logit = per_trophy[index]
        median, spread = forecast_trophy(points, target_x, horizon,
                                         last_logit, global_slope)
        quantiles = []
        previous = None
        for z in Z_VALUES:
            value = clamp_rate(inv_logit_pct(median + z * spread))
            value = round4(value)
            if previous is not None and value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"id": trophy.get("id"), "q": quantiles})
    return {"trophies": rows}


def fallback_case(case):
    """Persistence forecast used when the history is unusable."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon = case.get("horizon_minutes") or 0
    try:
        horizon = float(horizon)
    except (TypeError, ValueError):
        horizon = 0.0
    walk = RW_COEF * math.sqrt(max(0.0, horizon) / 1440.0)
    spread = min(SPREAD_CAP, math.sqrt(walk * walk + RESIDUAL_FLOOR ** 2
                                       + REFRESH_FLOOR ** 2))
    last_rates = {}
    for read in history:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates") or []
        for i, rate in enumerate(rates):
            clamped = clamp_rate(rate)
            if clamped is not None:
                last_rates[i] = clamped
    rows = []
    for index, trophy in enumerate(trophies):
        rate = last_rates.get(index, NO_HISTORY_RATE)
        median = logit_of(rate)
        quantiles = []
        previous = None
        for z in Z_VALUES:
            value = round4(clamp_rate(inv_logit_pct(median + z * spread)))
            if previous is not None and value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"id": trophy.get("id"), "q": quantiles})
    return {"trophies": rows}


def process_file(in_path, out_path):
    try:
        with open(in_path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        if not isinstance(case, dict) or not case.get("trophies"):
            return
        try:
            result = forecast_case(case)
        except Exception:
            result = fallback_case(case)
    except Exception:
        return
    payload = json.dumps(result, separators=(",", ":"), sort_keys=False)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(payload)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(in_dir)
                   if name.endswith(".in"))
    for name in names:
        stem = name[:-3]
        process_file(os.path.join(in_dir, name),
                     os.path.join(out_dir, stem + ".out"))


if __name__ == "__main__":
    main()
