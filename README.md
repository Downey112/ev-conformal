# When is a prediction interval "valid"? Stress-testing conformal forecasts of EV charging demand

An independent re-examination of [*Probabilistic Forecasting of Electric Vehicle Charging in a Time-varying Hierarchical Infrastructure*](https://arxiv.org/abs/2610.13588) (Principato, Amara-Ouali, Goude, Hamrouche and Poggi, October 2026).

The paper forecasts daily energy demand at 1,398 public charging stations in Scotland, rolled up to 32 local authorities and a national total, and wraps the forecasts in conformal prediction intervals. Stations open, close and go offline, so the hierarchy changes every day. Its headline method, CP-MNR-Nested⋆, is reported as the most reliable.

By Muhammad Luqman Hakim. I reimplemented the methods in Python from the authors' released R code, tested them in a controlled simulation, and reran them on the full Scottish data, including 14 months the paper did not evaluate.

## Findings

**1. The paper's headline metric rewards over-coverage.**
The paper scores a method by its *validity rate*: the share of nodes whose coverage on the test set reaches 90%. A method calibrated at exactly 90% scores only about 53% on this metric, because roughly half its nodes land just below target by chance. So the 56% the paper reports for the simpler CP-MNR method is what correct calibration looks like, and Nested⋆ scores higher by producing wider intervals.

**2. A one-line fix to the simpler method matches the headline method, with narrower intervals.**
The paper defines the calibration quantile with the standard finite-sample `n+1` correction. The released code uses the plain empirical quantile. Restoring the correction gives, at station level on the paper's own test period:

| Method | Coverage | Validity rate | Mean interval length (kWh) |
|---|---|---|---|
| CP-MNR, released code | 0.903 | 55.2% | 390 |
| CP-MNR, quantile as defined in the paper | 0.922 | 72.9% | 413 |
| CP-MNR-Nested⋆, released code | 0.922 | 72.5% | 486 |
| CP-MNR-Nested⋆, exact interval routine | 0.912 | 67.1% | 487 |

The corrected CP-MNR matches Nested⋆ on coverage and validity with intervals about 15% shorter. Its cost: it returns an infinite interval on 2.4% of station-days, where a station has too few calibration days. Excluding those, its validity rate is still 70.6%.

**3. The released interval routine for Nested⋆ is shifted by one order statistic.**
A brute-force unit test shows that the sweep-line routine returns bounds one endpoint lower, at both ends, than the set it is meant to compute. The effect is negligible with 100 or more calibration days, but it accounts for part of Nested⋆'s edge on the real data (72.5% against 67.1% validity with the exact routine).

**4. The independence assumption the authors worry about is not what breaks coverage.**
The theory for Nested⋆ assumes station activity is independent of demand, which cannot hold because inactivity is defined by zero demand. In simulation, making activity strongly dependent on demand left average coverage unchanged for every method. What does fail, for every method, is coverage conditional on how much of the network is active.

**5. Coverage degrades outside the paper's test window.**
From August 2024 to October 2025, station coverage of the released CP-MNR falls to 0.888 and its validity rate to 37.6%, below the 52% an exactly calibrated method would score.

**6. A sharp regime shift costs sharpness more than coverage.**
Glasgow's demand roughly halved after tariffs were introduced on 4 April 2023. Misses cluster in the following two weeks; after that the intervals stay wider than needed.

## Replication check

Station-level validity rates on the paper's test period (February to August 2024):

| Method | Paper | This repository |
|---|---|---|
| CP | 38.1% | 36.4% |
| CP-MNR | 56.2% | 55.2% |
| CP-MNR-Nested⋆ | 73.6% | 72.5% |
| Adaptive CP-MNR | 57.6% | 57.9% |

These agree closely even though my base forecasts are much lighter than the paper's (see Limitations). The national level, a single node, does not replicate and is too noisy to compare.

## Figures

Validity rate in simulation. CP-MNR with the corrected quantile (green) is calibrated at almost exactly 90% and still scores near 55%:

![Validity rate in simulation](results/fig_validity_vs_rho.png)

Coverage conditional on how many stations are off. No method holds 90% across regimes:

![Coverage by activity regime](results/fig_regimes_rho0.6.png)

Monthly coverage on the Scottish data, February 2024 to October 2025:

![Monthly coverage](results/fig_real_monthly.png)

Glasgow City around the tariff change:

![Glasgow tariff change](results/fig_tariff_glasgow.png)

## Method

**Simulation.** A synthetic hierarchy of 1 national node, 3 regions and 12 stations with independent days, so exchangeability holds by construction. One parameter, `rho`, sets the correlation between a station's activity and its own unpredictable demand. Each configuration is run 300 times; standard errors on coverage are at most 0.004.

**Real data.** The authors' processed ChargePlace Scotland tables (October 2022 to October 2025). The protocol follows theirs: 14-day test windows, models refit for each window, calibration on a random 20% of historical weeks plus the week before the test window, 90% target. Results are averaged over 5 random calibration splits.

**Methods compared.** Plain split conformal (CP), CP-MNR, CP-MNR-Nested⋆ and Adaptive CP-MNR, each transcribed from the released R code, plus two corrected variants: CP-MNR with the `n+1` quantile and Nested⋆ with an exact interval routine.

**Scaling.** The authors report needing close to 36 GB of RAM. This runs on an 8 GB laptop because the reconciliation step never forms a 1,431 x 1,431 projection matrix: for a three-level hierarchy the normal equations reduce, by the Woodbury identity, to a 33 x 33 solve per activity pattern (`evconf/hierarchy.py`).

## Limitations

- **Lighter base forecasts.** I use gradient boosting for stations and regions and ridge regression for the national series, in place of the paper's average of random forest, GAM, XGBoost and TabICL. Differences from the paper could partly come from this.
- **Five calibration splits, not ten.**
- **The simulation is small and stylised**: 12 stations, 15% of station-days inactive (the real figure is 46%), no time dependence.
- **The two code discrepancies are reported to the authors but not yet confirmed by them** ([issue #1](https://github.com/PrincipatoG/Probabilistic-Hierarchical-EV/issues/1)). They are differences between the released code and, respectively, the paper's stated definition and the set the routine is meant to compute.
- **I did not reproduce** the paper's 0% national validity for the adaptive method; here it was valid in 3 of 5 splits.

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # bin/ instead of Scripts/ on macOS and Linux
.venv/Scripts/python -m pytest tests

# simulation (about 5 minutes on 8 cores)
.venv/Scripts/python scripts/run_sim.py --reps 300
.venv/Scripts/python scripts/plot_sim.py

# real data (about 15 minutes)
# download Data.zip from https://github.com/PrincipatoG/AOAS-EV/releases/latest and extract
# dataset_scotland_main.csv, dataset_region_main.csv and dataset_address_main.csv into data/raw/Data/
.venv/Scripts/python scripts/run_real.py --name extended --seeds 5 --end 2025-10-26
.venv/Scripts/python scripts/run_real.py --name tariff --seeds 5 --test-start 2023-03-13 --end 2023-08-06
.venv/Scripts/python scripts/analyse_real.py
```

## Layout

| Path | Contents |
|---|---|
| `evconf/hierarchy.py` | Summing matrix, activity patterns, fast refined projection |
| `evconf/conformal.py` | The conformal procedures and the two corrected variants |
| `evconf/simulate.py` | Synthetic hierarchy and one simulation replication |
| `evconf/data.py`, `evconf/realdata.py` | Data loading, features, rolling-window experiment |
| `scripts/` | Run and plotting scripts |
| `tests/` | Unit tests, including the brute-force check of the interval routine |
| `results/` | Summary tables (`sim.csv`, `real_summary.csv`, `real_glasgow.csv`) and figures |

## Credits

Methods and data are due to the paper's authors; their code is at [PrincipatoG/Probabilistic-Hierarchical-EV](https://github.com/PrincipatoG/Probabilistic-Hierarchical-EV) (MIT licence). The data comes from ChargePlace Scotland via the [GridCharge](https://huggingface.co/datasets/djordjebatic/GridCharge) dataset and is not redistributed here.
