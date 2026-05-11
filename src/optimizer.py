"""
Budget Optimization Module
==========================
Reallocates marketing budget across channels to maximize predicted revenue
using SLSQP (Sequential Least Squares Quadratic Programming).

Approach:
    Given a fixed total budget, find the allocation across channels that
    maximizes the model's predicted revenue, subject to:
    - Total spend = current total (budget-neutral)
    - Each channel gets at least 5% of total (no channel goes to zero)
    - Each channel gets at most 50% of total (no single-channel dependency)
"""

import numpy as np
from scipy.optimize import minimize


def predict_revenue_from_allocation(
    spend_allocation: np.ndarray,
    decay_rates: list,
    max_adstocks: list,
    saturation_params: list,
    coefficients: list,
    base_revenue: float,
    control_contribution: float,
) -> float:
    """
    Predict weekly revenue for a given spend allocation.

    Parameters
    ----------
    spend_allocation : np.ndarray
        Proposed weekly spend per channel.
    decay_rates : list
        Adstock decay rate per channel.
    max_adstocks : list
        Max adstock values from training data (for normalization).
    saturation_params : list
        List of (K, alpha) tuples per channel.
    coefficients : list
        OLS coefficients for each saturated channel.
    base_revenue : float
        Model intercept (base sales).
    control_contribution : float
        Sum of control variable contributions.

    Returns
    -------
    float
        Predicted revenue (negative, for minimization).
    """
    total_media_contribution = 0.0

    for i, (spend, decay, max_adstock, (K, alpha), coef) in enumerate(
        zip(spend_allocation, decay_rates, max_adstocks, saturation_params, coefficients)
    ):
        # Steady-state adstock approximation
        adstocked = spend / (1 - decay)

        # Normalize and apply Hill saturation
        x_norm = adstocked / max_adstock if max_adstock > 0 else 0
        saturated = x_norm**alpha / (x_norm**alpha + K**alpha)

        total_media_contribution += coef * saturated

    predicted = base_revenue + control_contribution + total_media_contribution
    return -predicted  # Negative because we minimize


def optimize_budget(
    current_spends: np.ndarray,
    decay_rates: list,
    max_adstocks: list,
    saturation_params: list,
    coefficients: list,
    base_revenue: float,
    control_contribution: float,
    min_share: float = 0.05,
    max_share: float = 0.50,
) -> dict:
    """
    Find optimal budget allocation using SLSQP.

    Parameters
    ----------
    current_spends : np.ndarray
        Current weekly spend per channel.
    decay_rates, max_adstocks, saturation_params, coefficients, base_revenue,
    control_contribution : see predict_revenue_from_allocation.
    min_share : float
        Minimum fraction of total budget per channel (default 5%).
    max_share : float
        Maximum fraction of total budget per channel (default 50%).

    Returns
    -------
    dict
        Keys: 'optimized_spends', 'current_revenue', 'optimized_revenue',
              'revenue_lift_pct', 'total_budget'
    """
    total_budget = current_spends.sum()

    # Constraints: total spend = budget
    constraints = {"type": "eq", "fun": lambda x: x.sum() - total_budget}

    # Bounds: each channel between min_share and max_share of total
    bounds = [(total_budget * min_share, total_budget * max_share)] * len(current_spends)

    args = (
        decay_rates,
        max_adstocks,
        saturation_params,
        coefficients,
        base_revenue,
        control_contribution,
    )

    result = minimize(
        predict_revenue_from_allocation,
        x0=current_spends,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    current_rev = -predict_revenue_from_allocation(current_spends, *args)
    optimized_rev = -result.fun

    return {
        "optimized_spends": result.x,
        "current_revenue": current_rev,
        "optimized_revenue": optimized_rev,
        "revenue_lift_pct": (optimized_rev - current_rev) / current_rev * 100,
        "total_budget": total_budget,
        "success": result.success,
    }
