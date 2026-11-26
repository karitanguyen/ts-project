"""
01_eda_starter.py -- starting point for the copper-gold ratio / 10Y yield project.

Needs: weekly_clean.csv in the same folder; numpy, pandas, matplotlib, statsmodels.
Run:   python 01_eda_starter.py        (figures are saved to ./figures and shown on screen)

Sections
  1. Load data and plot the levels (log ratio r_t and 10Y yield y_t)
  2. ACF/PACF of levels vs first differences, plus ADF tests
  3. Rolling 52-week correlation of the two differenced series
  4. Pre-whitening: fit an AR(p) to the differenced ratio, filter the differenced yield, CCF
  5. Same steps on 2000-2021 and 2022-2026
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg

warnings.filterwarnings("ignore", category=FutureWarning)   # harmless statsmodels notice about adfuller's return type
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
P = 10                              # AR order for pre-whitening (check that the residual ACF is flat)
SPLIT = pd.Timestamp("2021-12-31")  # end of the first sub-period
MAXLAG = 12                         # CCF lags shown on each side


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=130)


# ---------------------------------------------------------------- 1. data and levels
w = pd.read_csv("weekly_clean.csv", parse_dates=["date"]).set_index("date")
r, y = w["log_ratio"], w["yield_10y_pct"]          # levels
dr, dy = w["d_log_ratio"].dropna(), w["d_yield"].dropna()   # first differences (weekly changes)
print(f"{len(w)} weekly levels, {w.index.min().date()} to {w.index.max().date()}; {len(dr)} weekly changes")

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(r); ax[0].set_ylabel("log(copper/gold)")
ax[1].plot(y); ax[1].set_ylabel("10Y yield (%)")
for a in ax:
    a.axvline(pd.Timestamp("2022-01-01"), color="red", ls=":")
save(fig, "01_levels.png")

# ---------------------------------------------------------------- 2. ACF/PACF and stationarity
fig, ax = plt.subplots(4, 2, figsize=(11, 10))
for row, (s, name) in enumerate([(r, "log ratio (levels)"), (y, "yield (levels)"),
                                 (dr, "diff log ratio"), (dy, "diff yield")]):
    plot_acf(s, lags=40, ax=ax[row, 0], title=f"ACF: {name}")
    plot_pacf(s, lags=40, ax=ax[row, 1], method="ywm", title=f"PACF: {name}")
save(fig, "02_acf_pacf.png")

print("\nADF test (H0: unit root)")
for s, name in [(r, "log ratio"), (y, "yield"), (dr, "diff log ratio"), (dy, "diff yield")]:
    print(f"  {name:15s} p = {adfuller(s, autolag='AIC')[1]:.4f}")

# ---------------------------------------------------------------- 3. rolling correlation
roll = dr.rolling(52).corr(dy)
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(roll); ax.axhline(0, color="black", lw=0.7); ax.axvline(pd.Timestamp("2022-01-01"), color="red", ls=":")
ax.set_ylabel("52-week correlation"); ax.set_title("Rolling correlation of weekly changes")
save(fig, "03_rolling_corr.png")
print("\nMean rolling correlation by year:\n", roll.groupby(roll.index.year).mean().round(2).dropna().to_string())


# ---------------------------------------------------------------- 4. pre-whitening and CCF
def prewhiten(a, b, p=P):
    """Fit AR(p) to a; return a's residuals and b filtered with the same AR coefficients."""
    fit = AutoReg(a.values, lags=p, trend="c").fit()
    phi = fit.params[1:]                                        # AR coefficients phi_1..phi_p
    a_res = pd.Series(fit.resid, index=a.index[p:])             # whitened ratio changes
    bc = b - b.mean()
    b_filt = bc.copy()                                          # b_t - phi_1 b_{t-1} - ... - phi_p b_{t-p}
    for i, ph in enumerate(phi, start=1):
        b_filt = b_filt - ph * bc.shift(i)
    return a_res, b_filt.loc[a_res.index], phi


def ccf(a_res, b_filt, maxlag=MAXLAG):
    """corr(a_res[t-k], b_filt[t]).  k > 0: the ratio leads the yield by k weeks.  k = 0: same week."""
    ks = np.arange(-maxlag, maxlag + 1)
    return pd.Series([a_res.shift(k).corr(b_filt) for k in ks], index=ks)


def ccf_plot(c, n, title, name):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.vlines(c.index, 0, c.values, color="C0"); ax.axhline(0, color="black", lw=0.7)
    band = 1.96 / np.sqrt(n)
    ax.axhline(band, color="C0", ls="--"); ax.axhline(-band, color="C0", ls="--")
    ax.set(xlabel="lag k  (k > 0: ratio leads yield by k weeks)", ylabel="CCF", title=title)
    save(fig, name)


a_res, b_filt, phi = prewhiten(dr, dy)
print(f"\nAR({P}) coefficients sum to {phi.sum():.3f}; residual ACF should be flat:")
fig, ax = plt.subplots(figsize=(8, 3)); plot_acf(a_res, lags=30, ax=ax, title=f"ACF of AR({P}) residuals"); save(fig, "04_prewhiten_resid_acf.png")
c = ccf(a_res, b_filt); n = len(a_res)
ccf_plot(c, n, "Pre-whitened CCF, full sample", "05_ccf_full.png")
print(f"Full sample, n = {n}, 95% band = +/-{1.96 / np.sqrt(n):.3f}")
print("  k = 0..6:", c.loc[0:6].round(3).tolist())

# ---------------------------------------------------------------- 5. split-sample check
rows = {}
for label, mask in [("2000-2021", dr.index <= SPLIT), ("2022-2026", dr.index > SPLIT)]:
    ar_, br_, _ = prewhiten(dr[mask], dy[mask])
    cs = ccf(ar_, br_); rows[label] = cs
    ccf_plot(cs, len(ar_), f"Pre-whitened CCF, {label}", f"06_ccf_{label}.png")
    print(f"{label}, n = {len(ar_)}, band = +/-{1.96 / np.sqrt(len(ar_)):.3f}; k = 0..6:", cs.loc[0:6].round(3).tolist())

plt.show()
