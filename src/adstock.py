"""
Adstock Transformation Module
=============================
Implements geometric adstock decay to model the carryover effect of advertising.

Theory:
    Advertising impact doesn't vanish after one week. The geometric adstock model
    captures this lingering effect:

        Adstocked(t) = Spend(t) + λ × Adstocked(t-1)

    where λ (decay rate) controls how quickly the effect fades.
    - λ = 0 → no carryover (impact dies immediately)
    - λ = 1 → infinite memory (impact never fades)
"""

import numpy as np
import pandas as pd


def apply_adstock(series: pd.Series, decay: float) -> np.ndarray:
    """
    Apply geometric adstock transformation to a spend series.

    Parameters
    ----------
    series : pd.Series
        Raw weekly spend values.
    decay : float
        Decay rate (0 to 1). Higher values mean longer carryover.

    Returns
    -------
    np.ndarray
        Adstocked spend values.
    """
    adstocked = np.zeros(len(series))
    adstocked[0] = series.iloc[0]
    for t in range(1, len(series)):
        adstocked[t] = series.iloc[t] + decay * adstocked[t - 1]
    return adstocked


def find_optimal_decay(
    spend_series: pd.Series,
    revenue_series: pd.Series,
    decay_range: np.ndarray = None,
) -> tuple:
    """
    Grid search for the decay rate that maximizes correlation
    between adstocked spend and revenue.

    Parameters
    ----------
    spend_series : pd.Series
        Raw weekly spend for one channel.
    revenue_series : pd.Series
        Weekly revenue.
    decay_range : np.ndarray, optional
        Array of decay values to test. Defaults to 0.00–0.95 in steps of 0.05.

    Returns
    -------
    tuple
        (best_decay, best_correlation, all_results)
        where all_results is a list of (decay, correlation) pairs.
    """
    if decay_range is None:
        decay_range = np.arange(0.0, 1.0, 0.05)

    results = []
    for decay in decay_range:
        adstocked = apply_adstock(spend_series, decay)
        correlation = np.corrcoef(adstocked, revenue_series)[0, 1]
        results.append((round(decay, 2), round(correlation, 4)))

    best = max(results, key=lambda x: x[1])
    return best[0], best[1], results
