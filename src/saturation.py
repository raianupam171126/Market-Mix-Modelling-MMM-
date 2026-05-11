"""
Saturation Curves Module
========================
Implements the Hill function to model diminishing returns in advertising.

Theory:
    Doubling spend does NOT double impact. The Hill function captures this
    S-shaped response:

        S(x) = x^α / (x^α + K^α)

    where:
    - K (half-saturation): the spend level at which you get 50% of max effect
    - α (shape): steepness of the curve
      - α < 1 → concave (quick saturation)
      - α > 1 → S-shaped (slow start, then rapid, then saturation)
"""

import numpy as np
import pandas as pd


def hill_transform(
    x: np.ndarray, K: float = 0.5, alpha: float = 2.0
) -> np.ndarray:
    """
    Apply Hill saturation transformation.

    Parameters
    ----------
    x : np.ndarray
        Input values (typically adstocked spend).
    K : float
        Half-saturation point (0 to 1, after normalization).
    alpha : float
        Shape parameter controlling curve steepness.

    Returns
    -------
    np.ndarray
        Saturated values between 0 and 1.
    """
    x_norm = x / x.max() if x.max() > 0 else x
    return x_norm**alpha / (x_norm**alpha + K**alpha)


def apply_saturation(
    df: pd.DataFrame,
    adstock_cols: list,
    params: dict,
) -> pd.DataFrame:
    """
    Apply Hill saturation to multiple adstocked columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing adstocked columns.
    adstock_cols : list
        List of adstocked column names.
    params : dict
        Saturation parameters per channel.
        Example: {'tv': {'K': 0.5, 'alpha': 2.5}, ...}

    Returns
    -------
    pd.DataFrame
        DataFrame with new saturated columns added.
    """
    df = df.copy()
    for col in adstock_cols:
        channel = col.replace("_adstock", "")
        p = params.get(channel, {"K": 0.5, "alpha": 2.0})
        df[f"{channel}_saturated"] = hill_transform(
            df[col].values, K=p["K"], alpha=p["alpha"]
        )
    return df
