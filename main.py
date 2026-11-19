"""Forecast GTA VI trophy earn rates on Exophase.

A dual-regime forecaster for make.fun's gta6-trophy-rarity-forecast-01. The program
reads a case (trophy earn-rate history plus players tracked) and forecasts each
trophy's earn rate at the case's target time as five quantiles (0.05, 0.25, 0.5,
0.75, 0.95), scored by mean pinball loss on the logit scale.

Two regimes, chosen from the history the program actually receives:

FLAT history (every read identical): Exophase's page is stale and a refresh is
pending. The model predicts the dilution jump: a refresh multiplies the tracked
player count (the kit documents roughly a 6x jump on launch night) while earners
grow modestly and unevenly -- early, high-rate trophies faster than late, rare
ones. A trophy at 100% stays at 100%.

VARYING history (reads change over time): fit per-trophy trends -- a robust
Theil-Sen slope of log(earners) against log-time plus a growth trend for players
tracked, shrunk toward the tier mean (hierarchical shrinkage). The target rate is
predicted earners divided by predicted tracked players.

The quantile spread combines a random-walk term (0.5 * sqrt(days)), the fit
residual, and refresh-jump uncertainty. All computation is deterministic: the
program uses no randomness, so two runs on the same case write identical bytes.

Standard library only. Reads $MAKEFUN_INPUT_DIR (else /input) and writes
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Random-walk spread of a rate's move, in logit units per square-root day.
WALK_SIGMA = 0.5
# The refresh prior for a stale page: a refresh multiplies tracked players by
# about this much (the kit documents 638 -> 3868 on launch night, a 6.06x jump).
REFRESH_MULTIPLIER = 6.06
# Logit spread used in the flat regime, where the refresh dominates uncertainty.
# Tuned locally against four constructed practice answers (heterogeneous growth,
# pure dilution, concentrated growth, uniform growth): 0.25 beats 0.30 on all
# four (loss 0.108-0.120 vs 0.113-0.122) while keeping 90% coverage at
# 0.91-0.98. A narrower spread scores better still on the constructed answers
# but drops coverage below 0.9, which is too fragile if the real answer's
# growth is more dispersed than the constructions.
FLAT_SPREAD = 0.25
# Release moment the log-time axis is measured from (2026-11-19T00:00:00Z).
RELEASE_EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()


def clamp_rate(pct):
    """Clamp a percent into the range the logit can take."""
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    """logit of a percent, after clamping into [0.01, 99.99]."""
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of_logit(value):
    """Inverse logit: a logit value back to a percent in (0, 100)."""
    return 100.0 / (1.0 + math.exp(-value))


def round_4dp(x):
    """Round to 4 decimals, half up (as JavaScript's Math.round rounds it)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_instant(text):
    """Parse an ISO 8601 UTC timestamp to epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def quantiles_from(median_pct, spread):
    """Five quantiles around a median percent with a logit spread.

    Returns [q05, q25, q50, q75, q95], each in [0, 100] and non-decreasing.
    """
    centre = logit_of(median_pct)
    out = []
    for z in Z_SCORES:
        q = pct_of_logit(centre + z * spread)
        out.append(round_4dp(min(100.0, max(0.0, q))))
    # Rounding can in principle break monotonicity by a hair; enforce it.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def history_is_flat(history):
    """True when every read shows the same tracked count and the same rates."""
    if len(history) < 2:
        return True
    first = history[0]
    for read in history[1:]:
        if read["players_tracked"] != first["players_tracked"]:
            return False
        if read["rates"] != first["rates"]:
            return False
    return True


def rate_percentiles(rates):
    """Each trophy's percentile rank by its current rate (0 = lowest, 1 = highest).

    Trophies sharing a rate share the average percentile of their rank span.
    """
    n = len(rates)
    order = sorted(range(n), key=lambda i: (rates[i] if rates[i] is not None else -1.0))
    pct = [0.0] * n
    i = 0
    while i < n:
        j = i
        val = rates[order[i]]
        while j + 1 < n and rates[order[j + 1]] == val:
            j += 1
        # Ranks i..j share this rate; average percentile over that span.
        avg = ((i + j) / 2.0) / max(1, n - 1)
        for k in range(i, j + 1):
            pct[order[k]] = avg
        i = j + 1
    return pct


def flat_regime_rows(trophies, last_read):
    """Forecast rows for a flat (stale-page) history.

    A refresh is pending: tracked players jump by the refresh prior while earners
    grow modestly, fastest for the highest-rate (earliest) trophies. The 100%
    trophy stays at 100%.
    """
    rates = last_read["rates"]
    pcts = rate_percentiles(rates)
    rows = []
    for i, trophy in enumerate(trophies):
        rate = rates[i]
        if rate is None:
            rate = 0.0
        if rate >= 99.99:
            # Everybody tracked holds it; a refresh leaves it at 100%.
            rows.append({"id": trophy["id"], "q": quantiles_from(99.99, 0.05)})
            continue
        # Earner growth: only the highest-rate trophies grow meaningfully.
        # The cubic keeps growth near 1.0 for most trophies and lets the top
        # few reach ~1.8x, matching the launch-night pattern where growth
        # concentrates in early, widely-held trophies.
        growth = 1.0 + 0.8 * (pcts[i] ** 3)
        median = rate * growth / REFRESH_MULTIPLIER
        rows.append({"id": trophy["id"], "q": quantiles_from(median, FLAT_SPREAD)})
    return rows


def theil_sen(xs, ys):
    """Robust line fit: median pairwise slope and median intercept.

    Returns (slope, intercept). With fewer than two distinct x values the slope
    is 0 and the intercept is the median y.
    """
    n = len(xs)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if xs[j] != xs[i]:
                slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
    if not slopes:
        ordered = sorted(ys)
        mid = ordered[n // 2] if n % 2 == 1 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
        return 0.0, mid
    slopes.sort()
    m = len(slopes)
    slope = slopes[m // 2] if m % 2 == 1 else (slopes[m // 2 - 1] + slopes[m // 2]) / 2.0
    intercepts = sorted(y - slope * x for x, y in zip(xs, ys))
    intercept = intercepts[n // 2] if n % 2 == 1 else (intercepts[n // 2 - 1] + intercepts[n // 2]) / 2.0
    return slope, intercept


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    return ordered[n // 2] if n % 2 == 1 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def varying_regime_rows(case):
    """Forecast rows for a history with real variation.

    Per trophy, a Theil-Sen trend of log(earners) against log-time, shrunk
    toward the tier mean; a matching trend for players tracked. The target rate
    is predicted earners over predicted tracked players.
    """
    trophies = case["trophies"]
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target_epoch = parse_instant(case["target"])

    # Time axis: log(1 + hours since release), as the house curve baseline uses.
    def log_time(epoch):
        hours = max(0.0, (epoch - RELEASE_EPOCH) / 3600.0)
        return math.log1p(hours)

    # Cap the fit to at most 120 reads (evenly spaced) so the pairwise
    # Theil-Sen stays fast inside the 8-second cap.
    reads = history
    if len(reads) > 120:
        step = len(reads) / 120.0
        reads = [reads[int(k * step)] for k in range(120)]
        if reads[-1] is not history[-1]:
            reads.append(history[-1])

    xs = [log_time(parse_instant(r["at"])) for r in reads]
    x_target = log_time(target_epoch)
    n_trophies = len(trophies)

    # Trend for players tracked.
    log_n = [math.log(max(1, r["players_tracked"])) for r in reads]
    n_slope, n_intercept = theil_sen(xs, log_n)
    n_target = math.exp(n_intercept + n_slope * x_target)

    # Per-trophy earner trends.
    slopes = []
    intercepts = []
    residuals = []
    for i in range(n_trophies):
        ex = []
        ey = []
        for k, read in enumerate(reads):
            rate = read["rates"][i]
            if rate is None:
                continue
            earners = rate * read["players_tracked"] / 100.0
            ex.append(xs[k])
            ey.append(math.log(max(0.5, earners)))
        if len(ex) >= 2:
            s, b = theil_sen(ex, ey)
            resid = median_of([abs(y - (b + s * x)) for x, y in zip(ex, ey)])
        else:
            s, b, resid = 0.0, (ey[0] if ey else 0.0), 0.0
        slopes.append(s)
        intercepts.append(b)
        residuals.append(resid)

    # Tier-mean slopes for hierarchical shrinkage.
    tier_slopes = {}
    for i, trophy in enumerate(trophies):
        tier_slopes.setdefault(trophy["tier"], []).append(slopes[i])
    tier_mean = {t: median_of(v) for t, v in tier_slopes.items()}

    # Random-walk spread over the horizon.
    walk = WALK_SIGMA * math.sqrt(max(horizon_days, 0.0))

    rows = []
    for i, trophy in enumerate(trophies):
        # Shrink the trophy's slope toward its tier mean; the weight grows
        # with the number of usable reads for that trophy.
        usable = sum(1 for r in reads if r["rates"][i] is not None)
        weight = usable / (usable + 5.0)
        slope = weight * slopes[i] + (1.0 - weight) * tier_mean[trophy["tier"]]
        log_e = intercepts[i] + slope * x_target
        earners_target = math.exp(log_e)
        median = 100.0 * earners_target / max(1.0, n_target)
        median = min(100.0, max(0.0, median))
        # Spread: random walk, fit residual and refresh uncertainty combined.
        spread = math.sqrt(walk * walk + residuals[i] * residuals[i] + 0.01)
        rows.append({"id": trophy["id"], "q": quantiles_from(median, spread)})
    return rows


def forecast_case(case):
    """Forecast one case, dispatching on whether its history is flat."""
    if case["levels"] != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (case["levels"], LEVELS))
    if not case["history"]:
        raise ValueError("case has no history")
    if history_is_flat(case["history"]):
        rows = flat_regime_rows(case["trophies"], case["history"][-1])
    else:
        rows = varying_regime_rows(case)
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
