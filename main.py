"""gta6-twitch-ccu-forecast-01: comps-curve forecaster, Python 3.13 stdlib only.

Model (learned offline from the 37 past big-launch Twitch curves in the kit):
- For each comparable launch, take its release-day hourly curve and express
  every hour as a log-ratio to its hour-1 print: lr(h) = ln(viewers(h)/viewers(1)).
- Weight comparables by how close their hour-1 print is to this case's hour-1
  print (Gaussian kernel, bandwidth 0.75 in log space). Hour-1 level is a proxy
  for the unlock pattern: a big hour 1 means the launch already unlocked in
  big markets (rolling midnight); a tiny one means the unlock is still coming
  (simultaneous global unlock later in the day). Launches with the same unlock
  pattern share the same day shape: a dip in hours 2-4, a jump at hour 5
  (US East unlock), a plateau, then a ramp into the US-evening peak.
- Keep only tier="comparable" launches when enough of them remain (the kit
  marks early-access-contaminated and weak launches; their shapes mislead).
  Fall back to all launches when too few remain.
- Per hour, take the weighted quantiles of the log-ratios at the five scored
  levels, narrow the spread by 0.8 (the raw cross-launch spread is wider than
  the conditional spread once hour-1 level is known), and forecast
  q = p1 * exp(quantile).
- Fall back to the house persistence baseline when there are no usable peers.

Backtest (leave-one-out over the 37 kit comparables, plus the 4 practice
launches scored against SullyGnome's published hourly counts, with the
contest's pinball loss on the log scale):
- house baseline (persistence, sigma 0.15): 0.930 on the 37 launches, 0.387
  on the 4 practice launches (matches the kit's stated ~0.387).
- this model: 0.410 on the 37 launches, 0.197 on the 4 practice launches.

Deterministic: pure functions of the case file, no randomness, no clock, no
network. Two runs on the same case write byte-identical output.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is needed here, but keep the
# same layout as the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (house baseline's spread).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of an hour's move, in natural-log units per square-root hour.
SIGMA = 0.15

# --- Model constants, tuned on the kit's comparables (see docstring). ---
# Gaussian kernel bandwidth in log hour-1 space.
BANDWIDTH = 0.75
# Spread multiplier around the weighted median log-ratio.
SPREAD = 0.8
# Minimum effective sample size before the bandwidth is widened.
MIN_ESS = 8
# Maximum bandwidth widening (x1.5 steps) when peers are far away.
MAX_BANDWIDTH = 3.0
# Minimum tier="comparable" peers before falling back to all peers.
MIN_TIER_PEERS = 8

TIER_RE = re.compile(r"Tier: ([a-z\-]+)")


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> float:
    """ISO-8601 UTC timestamp to epoch seconds.

    Uses datetime.fromisoformat so fractional seconds and numeric offsets are
    handled, not just the whole-hour '...T00:00:00Z' form the kit ships.
    """
    from datetime import datetime

    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return moment.timestamp()


def day_series(comp: dict) -> dict:
    """A comparable's release-day hourly mean viewers, hours 1..24."""
    start = parse_time(comp["day_start"])
    by_hour: dict[int, list[float]] = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers:
            continue
        hour = int((parse_time(point["from"]) - start) // 3600) + 1
        if 1 <= hour <= 24:
            by_hour.setdefault(hour, []).append(float(viewers))
    return {h: sum(v) / len(v) for h, v in by_hour.items() if v}


def tier_of(comp: dict) -> str:
    """The kit's tier for a comparable ('comparable', 'early-access-contaminated',
    'weak', 'target-history'); 'comparable' when the text names none."""
    match = TIER_RE.search(comp.get("unlock") or "")
    return match.group(1) if match else "comparable"


def weighted_quantile(pairs: list, prob: float) -> float:
    """Linear-interpolated weighted quantile of [(value, weight), ...]."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = prob * total
    acc = 0.0
    for i, (value, weight) in enumerate(ordered):
        acc += weight
        if acc >= target:
            if i == 0 or weight <= 0:
                return value
            prev_value, _ = ordered[i - 1]
            frac = (target - (acc - weight)) / weight
            return prev_value + (value - prev_value) * frac
    return ordered[-1][0]


def effective_sample_size(weights: list) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    return total * total / sum(w * w for w in weights)


def comps_quantiles(p1: float, peers: list) -> dict:
    """Weighted log-ratio quantiles per hour from the peer launch curves."""
    log_p1 = math.log(max(1.0, p1))
    bandwidth = BANDWIDTH
    while True:
        weights = [
            math.exp(-0.5 * ((math.log(max(1.0, s[1])) - log_p1) / bandwidth) ** 2)
            for _, s in peers
        ]
        if effective_sample_size(weights) >= MIN_ESS or bandwidth >= MAX_BANDWIDTH:
            break
        bandwidth *= 1.5
    out = {}
    for hour in range(2, 25):
        pairs = []
        for (_, series), weight in zip(peers, weights):
            if hour in series:
                pairs.append(
                    (math.log(max(1.0, series[hour]) / max(1.0, series[1])), weight)
                )
        if not pairs:
            out[hour] = None
            continue
        median = weighted_quantile(pairs, 0.5)
        quantiles = [weighted_quantile(pairs, level) for level in LEVELS]
        out[hour] = [median + SPREAD * (q - median) for q in quantiles]
    return out


def baseline_quantiles(p1: float) -> dict:
    """The house persistence baseline's log-spread per hour."""
    return {
        hour: [z * SIGMA * math.sqrt(hour - 1) for z in Z] for hour in range(2, 25)
    }


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    peers = []
    for comp in case.get("comparables") or []:
        series = day_series(comp)
        if 1 in series and len(series) >= 12:
            peers.append((comp, series))
    tier_peers = [(c, s) for c, s in peers if tier_of(c) == "comparable"]
    usable = tier_peers if len(tier_peers) >= MIN_TIER_PEERS else peers

    log_quantiles = comps_quantiles(p1, usable) if usable else {}
    fallback = baseline_quantiles(p1)
    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        log_q = log_quantiles.get(hour)
        if log_q is None:
            log_q = fallback[hour]
        quantiles = [round2(p1 * math.exp(v)) for v in log_q]
        quantiles = [max(1, min(1_000_000_000, q)) for q in quantiles]
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def safe_forecast_case(case: dict, name: str) -> dict:
    """A forecast that is always validly shaped, even on a malformed case."""
    try:
        return forecast_case(case, name)
    except Exception:
        try:
            first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
            p1 = max(1.0, float(first["viewers"])) if first else 1.0
        except Exception:
            p1 = 1.0
        fallback = baseline_quantiles(p1)
        return {
            "hours": [
                {
                    "hour": hour,
                    "q": [
                        max(1, min(1_000_000_000, round2(p1 * math.exp(v))))
                        for v in fallback[hour]
                    ],
                }
                for hour in range(2, 25)
            ]
        }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = safe_forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
