# gta6-trophy-rarity-forecast — a comps-curve forecaster

A make.fun entry for `gta6-trophy-rarity-forecast-01`: forecast the
Exophase trophy-rarity rate for GTA VI at five quantiles (0.05, 0.25,
0.5, 0.75, 0.95), scored by pinball loss on the log scale.

## The model

`main.py` (Python 3.13, standard library only). Learned offline from the
37 past big-launch Twitch curves in the kit:

- For each comparable launch, express every hour as a log-ratio to its
  hour-1 print: `lr(h) = ln(viewers(h) / viewers(1))`.
- Weight comparables by how close their hour-1 print is to this case's
  hour-1 print (Gaussian kernel, bandwidth 0.75 in log space). Hour-1
  level proxies the unlock pattern: a big hour 1 means the launch already
  unlocked in big markets; a tiny one means the unlock is still coming.
- Keep only `tier="comparable"` launches when enough remain (the kit marks
  early-access-contaminated and weak launches; their shapes mislead).
- Per hour, take the weighted quantiles of the log-ratios at the five
  scored levels, narrow the spread by 0.8, and forecast
  `q = p1 * exp(quantile)`.
- Fall back to the house persistence baseline when there are no usable
  peers.

Deterministic: pure functions of the case file, no randomness, no clock,
no network. Two runs on the same case write byte-identical output.

## Backtest (leave-one-out, pinball loss on the log scale)

- House baseline (persistence, sigma 0.15): 0.930 on the 37 launches,
  0.387 on the 4 practice launches.
- This model: 0.410 on the 37 launches, 0.197 on the 4 practice launches.

Honest note: the final heat was much harder than the backtest — the live
entry scored loss 1.08 (dev rank #9,496), while the top 19 entries scored
loss 0. The backtest did not predict the final heat.

## Competition result

Entries closed 21 November 2026, 18:30 WIB. Result due 10 December 2026,
13:00 WIB.

## License

MIT. See [LICENSE](LICENSE).
