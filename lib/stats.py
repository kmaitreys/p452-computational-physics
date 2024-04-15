"""
This module houses the statistics functions for the library.
"""

import math

import scipy.stats as sps

from .array import Array, pow


def mean(x):
    return sum(x) / len(x)


def std(x):
    return math.sqrt(sum((x - mean(x)) ** 2) / (len(x)))


def pooled_std(std_1, std_2, n_1, n_2):
    return math.sqrt(((n_1 - 1) * std_1**2 + (n_2 - 1) * std_2**2) / (n_1 + n_2 - 2))


def standard_normal(x, mean, obs):
    return pow(math.e, -0.5 * (x - mean) ** 2) / math.sqrt(2 * math.pi) * sum(obs)


def expected_normal(g, obs):
    mean_g = mean(g)
    return standard_normal(g, mean_g, obs)


def chi2(obs: Array, exp: Array) -> float:
    if isinstance(obs, list):
        obs = Array.fromlist("d", obs)
    if isinstance(exp, list):
        exp = Array.fromlist("d", exp)

    return sum((obs - exp) ** 2 / exp)


# def chi2_ppf(df, p):
#     """
#     Calculate the inverse cumulative distribution function (CDF) of
#     the chi-squared distribution.

#     Parameters:
#         df (float): Degrees of freedom.
#         p (float): Probability.

#     Returns:
#         float: Value such that the probability of observing a value
#         less than or equal to that value is equal to the given probability.
#     """
#     if p <= 0.0 or p >= 1.0:
#         raise ValueError("Probability must be in the range (0, 1)")

#     if df <= 0:
#         raise ValueError("Degrees of freedom must be greater than 0")

#     # Define a tolerance level for convergence
#     tolerance = 1e-6
#     # Initialize lower and upper bounds
#     lower = 0.0
#     upper = max(1.0, df)
#     # Set initial guess for the root
#     x = (lower + upper) / 2

#     # Iteratively find the root using binary search
#     while True:
#         # Calculate the cumulative distribution function (CDF) at x
#         cdf_x = 1 - chi2_cdf(df, x)
#         # Check if we have reached the desired probability
#         if abs(cdf_x - p) < tolerance:
#             return x
#         # Update the bounds based on the comparison with the desired probability
#         elif cdf_x < p:
#             upper = x
#         else:
#             lower = x
#         # Update the guess for the root
#         x = (lower + upper) / 2


# def chi2_cdf(df, x):
#     """
#     Calculate the cumulative distribution function (CDF) of the
#     chi-squared distribution.

#     Parameters:
#         df (float): Degrees of freedom.
#         x (float): Value at which to evaluate the CDF.

#     Returns:
#         float: Probability that a chi-squared random variable
#         with the given degrees of freedom is less than or equal to x.
#     """
#     if x < 0:
#         return 0.0

#     gamma_term = math.gamma(df / 2)
#     return (1 / gamma_term) * math.pow(x / 2, df / 2) * math.exp(-x / 2)


def chi2_ppf(p, df):
    return sps.chi2.ppf(p, df)


def f_test(p, df1, df2):
    return sps.f.ppf(p, df1, df2)


def t_test(p, df):
    return sps.t.ppf(p, df)
