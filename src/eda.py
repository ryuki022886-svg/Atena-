"""EDA for ETT oil-temperature forecasting PoC.

Reproduces the six findings from spec section 3 (autocorrelation, load-feature
correlation, periodicity, distribution shift, data quality, persistence
baseline) for the requested ETT datasets, printing summary numbers and saving
figures under figures/<dataset>/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET = "OT"

# Informer-paper convention: 12/4/4 "months" of 30 days each, not calendar months.
MONTH_HOURS = 30 * 24


def load_data(name: str, data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{name}.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def steps_per_hour(df: pd.DataFrame) -> int:
    freq_minutes = df.index.to_series().diff().dropna().mode()[0].total_seconds() / 60
    return round(60 / freq_minutes)


def section_3_1_autocorrelation(df: pd.DataFrame, sph: int, outdir: Path) -> dict:
    """OT self-autocorrelation at increasing lags."""
    ot = df[TARGET]
    lags_h = [1, 6, 24, 168, 720]
    results = {h: ot.autocorr(lag=h * sph) for h in lags_h}

    max_lag_h = 24 * 30  # 30 days
    lags = np.arange(1, max_lag_h * sph + 1, sph)
    acf_vals = [ot.autocorr(lag=int(l)) for l in lags]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags / sph, acf_vals)
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("autocorrelation of OT")
    ax.set_title("3.1 OT self-autocorrelation")
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "3_1_autocorrelation.png", dpi=120)
    plt.close(fig)

    return results


def section_3_2_load_feature_correlation(df: pd.DataFrame, sph: int, outdir: Path) -> pd.Series:
    """Pearson correlation of each load feature with OT, plus lag-correlation search."""
    corr = df[LOAD_COLS].corrwith(df[TARGET]).sort_values(ascending=False)

    max_lag_h = 24
    lag_range = range(-max_lag_h, max_lag_h + 1)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for ax, col in zip(axes.flat, LOAD_COLS):
        vals = [df[col].shift(l * sph).corr(df[TARGET]) for l in lag_range]
        ax.plot(list(lag_range), vals)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_title(col)
    fig.suptitle("3.2 Lag correlation of load features with OT (negative lag = feature leads OT)")
    fig.supxlabel("lag (hours)")
    fig.tight_layout()
    fig.savefig(outdir / "3_2_load_lag_correlation.png", dpi=120)
    plt.close(fig)

    return corr


def section_3_3_periodicity(df: pd.DataFrame, outdir: Path) -> dict:
    hourly = df.groupby(df.index.hour)[TARGET].mean()
    weekday = df.groupby(df.index.dayofweek)[TARGET].mean()
    monthly = df.groupby(df.index.month)[TARGET].mean()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    hourly.plot(ax=axes[0], marker="o")
    axes[0].set_title(f"OT mean by hour-of-day (amplitude={hourly.max()-hourly.min():.2f})")
    axes[0].set_xlabel("hour")

    weekday.plot(ax=axes[1], marker="o")
    axes[1].set_title(f"OT mean by weekday (amplitude={weekday.max()-weekday.min():.2f})")
    axes[1].set_xlabel("day of week (0=Mon)")

    monthly.plot(ax=axes[2], marker="o")
    axes[2].set_title(f"OT mean by month (amplitude={monthly.max()-monthly.min():.2f})")
    axes[2].set_xlabel("month")

    fig.suptitle("3.3 Periodicity of OT")
    fig.tight_layout()
    fig.savefig(outdir / "3_3_periodicity.png", dpi=120)
    plt.close(fig)

    return {
        "hourly_amplitude": hourly.max() - hourly.min(),
        "weekday_amplitude": weekday.max() - weekday.min(),
        "monthly_amplitude": monthly.max() - monthly.min(),
    }


def section_3_4_distribution_shift(df: pd.DataFrame, sph: int, outdir: Path) -> pd.DataFrame:
    """Informer-convention 12/4/4 "month" (30-day) train/val/test split."""
    n = len(df)
    train_end = 12 * MONTH_HOURS * sph
    val_end = 16 * MONTH_HOURS * sph
    test_end = min(20 * MONTH_HOURS * sph, n)

    splits = {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:test_end],
    }
    summary = pd.DataFrame(
        {
            "start": [s.index.min() for s in splits.values()],
            "end": [s.index.max() for s in splits.values()],
            "ot_mean": [s[TARGET].mean() for s in splits.values()],
            "ot_std": [s[TARGET].std() for s in splits.values()],
        },
        index=splits.keys(),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_mean = df[TARGET].resample("MS").mean()
    monthly_mean.plot(ax=ax, marker="o")
    for name, s in splits.items():
        ax.axvspan(s.index.min(), s.index.max(), alpha=0.15, label=name)
    ax.legend()
    ax.set_title("3.4 Monthly mean OT with train/val/test split (distribution shift)")
    fig.tight_layout()
    fig.savefig(outdir / "3_4_distribution_shift.png", dpi=120)
    plt.close(fig)

    return summary


def _max_consecutive_run(s: pd.Series) -> int:
    same_as_prev = s.eq(s.shift())
    groups = (~same_as_prev).cumsum()
    return int(same_as_prev.groupby(groups).sum().max()) + 1


def section_3_5_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in LOAD_COLS + [TARGET]:
        rows.append(
            {
                "column": col,
                "max_consecutive_identical": _max_consecutive_run(df[col]),
                "n_negative": int((df[col] < 0).sum()),
                "n_zero": int((df[col] == 0).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def section_3_6_persistence_baseline(df: pd.DataFrame, sph: int, outdir: Path) -> pd.Series:
    horizons_h = [1, 6, 24, 96, 336]
    mae = {}
    for h in horizons_h:
        steps = h * sph
        pred = df[TARGET].shift(steps)
        mae[h] = (df[TARGET] - pred).abs().mean()
    mae = pd.Series(mae, name="persistence_MAE")

    fig, ax = plt.subplots(figsize=(6, 4))
    mae.plot(ax=ax, marker="o")
    ax.set_xlabel("horizon (hours)")
    ax.set_ylabel("MAE")
    ax.set_title("3.6 Persistence baseline MAE by horizon")
    fig.tight_layout()
    fig.savefig(outdir / "3_6_persistence_baseline.png", dpi=120)
    plt.close(fig)

    return mae


def run_eda(name: str, data_dir: Path, fig_dir: Path) -> None:
    outdir = fig_dir / name
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(name, data_dir)
    sph = steps_per_hour(df)

    print(f"\n{'='*60}\n{name}  (rows={len(df)}, steps/hour={sph})\n{'='*60}")

    print("\n[3.1] OT self-autocorrelation by lag (hours):")
    for h, v in section_3_1_autocorrelation(df, sph, outdir).items():
        print(f"  lag={h:>4}h : corr={v:.3f}")

    print("\n[3.2] Load feature correlation with OT (sorted):")
    print(section_3_2_load_feature_correlation(df, sph, outdir).round(3).to_string())

    print("\n[3.3] Periodicity amplitude (max-min of group means):")
    for k, v in section_3_3_periodicity(df, outdir).items():
        print(f"  {k}: {v:.3f}")

    print("\n[3.4] Distribution shift (12/4/4-month split, Informer convention):")
    print(section_3_4_distribution_shift(df, sph, outdir).round(3).to_string())

    print("\n[3.5] Data quality:")
    print(section_3_5_data_quality(df).to_string())

    print("\n[3.6] Persistence baseline MAE by horizon:")
    print(section_3_6_persistence_baseline(df, sph, outdir).round(3).to_string())

    print(f"\nFigures saved to {outdir}/")


def main():
    parser = argparse.ArgumentParser(description="ETT EDA (spec section 3)")
    parser.add_argument(
        "--datasets", nargs="+", default=["ETTh1", "ETTh2"],
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    for name in args.datasets:
        run_eda(name, args.data_dir, args.fig_dir)


if __name__ == "__main__":
    main()
