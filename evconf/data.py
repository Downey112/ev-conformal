"""Load the authors' processed ChargePlace Scotland tables into dense arrays.

Everything is (day, series) shaped. Features follow the "global" formulas of the
authors' Model_training/R scripts (calendar, weather, equipment mix, log lags),
recomputed here from the raw columns so no future information leaks in.
"""
import os

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Data")
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "panel.npz")

_COMMON = ["Date", "Consumed_kWh", "n_cp_lag1", "Weekday_Holiday", "Posan",
           "max_cp_30days", "max_paid_cp_30days", "n_cp_paid_lag1", "tmpf_max", "relh_min"]
_EQUIP = ["nbr_cp_infrastructure", "Rapide", "Lent", "UltraRapide"]
_WEEKDAYS = {"Tuesday", "Wednesday", "Thursday", "Friday"}


def _ffill(a):
    """Forward-fill NaN along axis 0 (zoo::na.locf(na.rm = FALSE))."""
    idx = np.where(np.isnan(a), 0, np.arange(a.shape[0])[:, None])
    np.maximum.accumulate(idx, axis=0, out=idx)
    return np.take_along_axis(a, idx, axis=0)


def _lag(a, k):
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def _wide(df, key, col, dates, keys):
    return df.pivot(index="Date", columns=key, values=col).reindex(index=dates, columns=keys).to_numpy(float)


def _features(df, key, dates, keys, extra):
    """Returns Y (T, k) and the feature dict {name: (T, k)}."""
    W = lambda c: _wide(df, key, c, dates, keys)
    Y = W("Consumed_kWh")
    logy = np.log1p(Y)
    lags = [_ffill(_lag(logy, k)) for k in range(1, 8)]
    ncp_lag1 = _ffill(W("n_cp_lag1"))
    paid_lag1 = _ffill(W("n_cp_paid_lag1"))
    max_cp, max_paid = W("max_cp_30days"), W("max_paid_cp_30days")
    max_cp = np.where(np.isnan(max_cp), ncp_lag1, max_cp)
    max_paid = np.where(np.isnan(max_paid), paid_lag1, max_paid)
    with np.errstate(invalid="ignore", divide="ignore"):
        paid_share = max_paid / max_cp
        lag_mean = np.nanmean(np.stack(lags), 0)
    wd = df.drop_duplicates("Date").set_index("Date")["Weekday_Holiday"].reindex(dates)
    wd = wd.where(~wd.isin(_WEEKDAYS), "Weekday")
    wd_code = pd.Categorical(wd).codes.astype(float)[:, None] * np.ones_like(Y)
    F = {
        "weekday_holiday": wd_code,
        "posan": W("Posan"),
        "paid": np.where(np.isnan(paid_share), np.nan, paid_share != 0),
        "log_max_cp_30days": np.log1p(max_cp),
        "tmpf_max": W("tmpf_max"),
        "relh_min": W("relh_min"),
        "log_lag_mean_1_7": lag_mean,
        "log_ncp_lag1": np.log1p(ncp_lag1),
        "log_lag_1": lags[0],
        "log_lag_7": lags[6],
    }
    if "nbr_cp_infrastructure" in df:
        infra = W("nbr_cp_infrastructure")
        with np.errstate(invalid="ignore", divide="ignore"):
            F["share_slow"] = W("Lent") / infra
            F["share_rapid"] = (W("Rapide") + W("UltraRapide")) / infra
    for c in extra:
        F[c.lower()] = W(c)
    return Y, F


def build_panel(force=False):
    """Parse the CSVs once and cache the dense panel."""
    if os.path.exists(CACHE) and not force:
        return dict(np.load(CACHE, allow_pickle=True))
    nat = pd.read_csv(os.path.join(RAW, "dataset_scotland_main.csv"), usecols=_COMMON)
    reg = pd.read_csv(os.path.join(RAW, "dataset_region_main.csv"), usecols=_COMMON + _EQUIP + ["Region"])
    sta = pd.read_csv(os.path.join(RAW, "dataset_address_main.csv"),
                      usecols=_COMMON + _EQUIP + ["Region", "Station.ID", "Latitude", "Longitude"])
    dates = np.sort(nat.Date.unique())
    regions = np.sort(reg.Region.unique())
    meta = sta.drop_duplicates("Station.ID").sort_values(["Region", "Station.ID"])
    stations = meta["Station.ID"].to_numpy()
    region_of = np.searchsorted(regions, meta.Region.to_numpy())

    nat["key"] = 0
    Yn, Fn = _features(nat, "key", dates, [0], [])
    Yr, Fr = _features(reg, "Region", dates, regions, [])
    Ys, Fs = _features(sta, "Station.ID", dates, stations, ["Latitude", "Longitude"])
    Fr["region"] = np.arange(len(regions))[None, :] * np.ones_like(Yr)

    out = dict(dates=dates.astype("datetime64[D]"), regions=regions, stations=stations,
               region_of=region_of, Y=np.concatenate([Yn, Yr, Ys], 1))
    for lvl, F in (("nat", Fn), ("reg", Fr), ("sta", Fs)):
        out[f"X_{lvl}"] = np.stack(list(F.values()), -1).astype(np.float32)
        out[f"names_{lvl}"] = np.array(list(F))
    np.savez_compressed(CACHE, **out)
    return out
