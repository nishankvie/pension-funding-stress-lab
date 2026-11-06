"""Check the economic assumptions against history. The yearly simulation is added in Block 5."""

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12

# A value is only called "estimated" if the two halves of the history broadly agree. These limits
# are a judgement call, recorded here so that the decision for each parameter is the same rule
MAX_VOLATILITY_RATIO = 1.25
MAX_CORRELATION_GAP = 0.20

# Order of the variables in the correlation matrix
RISK_FACTORS = ["rate", "inflation", "spread", "equity"]


def yearly_changes(market):
    """Return 12-month changes, matching the yearly steps of the simulation."""
    changes = pd.DataFrame({
        "rate": market["rate"].diff(MONTHS_PER_YEAR),
        "inflation": market["inflation"].diff(MONTHS_PER_YEAR),
        "equity": np.log(market["equity_index"]).diff(MONTHS_PER_YEAR),
    })
    return changes.dropna()


def split_in_half(changes):
    """Return the first and second half of the history."""
    middle = len(changes) // 2
    return changes.iloc[:middle], changes.iloc[middle:]


def volatility_row(changes, column, parameter):
    """Return one calibration row for a volatility, with its stability verdict."""
    first, second = split_in_half(changes)
    full_value = changes[column].std()
    first_value = first[column].std()
    second_value = second[column].std()
    ratio = max(first_value, second_value) / min(first_value, second_value)
    return {
        "parameter": parameter,
        "full_sample": full_value,
        "first_half": first_value,
        "second_half": second_value,
        "stable": bool(ratio <= MAX_VOLATILITY_RATIO),
    }


def correlation_row(changes, column_a, column_b, parameter):
    """Return one calibration row for a correlation, with its stability verdict."""
    first, second = split_in_half(changes)
    full_value = changes[column_a].corr(changes[column_b])
    first_value = first[column_a].corr(first[column_b])
    second_value = second[column_a].corr(second[column_b])
    return {
        "parameter": parameter,
        "full_sample": full_value,
        "first_half": first_value,
        "second_half": second_value,
        "stable": bool(abs(first_value - second_value) <= MAX_CORRELATION_GAP),
    }


def calibration_table(market):
    """Return what history says about each parameter that can be measured directly."""
    changes = yearly_changes(market)
    rows = [
        volatility_row(changes, "rate", "rate_volatility"),
        volatility_row(changes, "inflation", "inflation_volatility"),
        volatility_row(changes, "equity", "equity_volatility"),
        correlation_row(changes, "rate", "inflation", "rate_inflation"),
        correlation_row(changes, "rate", "equity", "rate_equity"),
        correlation_row(changes, "inflation", "equity", "inflation_equity"),
    ]
    return pd.DataFrame(rows)


def mean_reversion_estimate(series):
    """Return the yearly pull back towards the average, and the implied long-run level."""
    # Regress each month on the month before: value(t+1) = a + b x value(t)
    current = series.iloc[:-1].to_numpy()
    following = series.iloc[1:].to_numpy()
    b, a = np.polyfit(current, following, 1)
    return {"yearly_speed": 1 - b ** MONTHS_PER_YEAR, "long_run_level": a / (1 - b)}


def mean_reversion_table(market):
    """Return mean reversion estimates for the full history and each half, to show stability."""
    rows = []
    for column in ["rate", "inflation"]:
        series = market[column].dropna()
        first, second = split_in_half(series)
        for label, part in [("full sample", series), ("first half", first),
                            ("second half", second)]:
            rows.append({"variable": column, "sample": label, **mean_reversion_estimate(part)})
    return pd.DataFrame(rows)


def correlation_matrix(correlations):
    """Return the 4 x 4 correlation matrix built from the six pairwise values in the config."""
    matrix = np.eye(len(RISK_FACTORS))
    for i, first in enumerate(RISK_FACTORS):
        for j, second in enumerate(RISK_FACTORS):
            if i < j:
                matrix[i, j] = correlations[f"{first}_{second}"]
                matrix[j, i] = matrix[i, j]
    return matrix


def check_correlation_matrix(matrix):
    """Raise an error unless the matrix is one that real variables could actually have."""
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Correlation matrix is not symmetric.")
    if not np.allclose(np.diag(matrix), 1.0):
        raise ValueError("Correlation matrix must have 1 on its diagonal.")
    if np.abs(matrix).max() > 1:
        raise ValueError("Correlations must be between -1 and 1.")
    # Some sets of correlations contradict each other. Cholesky needs every eigenvalue above zero
    smallest_eigenvalue = np.linalg.eigvalsh(matrix).min()
    if smallest_eigenvalue <= 0:
        raise ValueError(f"Correlations contradict each other (eigenvalue {smallest_eigenvalue}).")
