import numpy as np

from . import utils as ut


def get_flux(geometry, source_function, tau_freq, solid_angle):

    if (
        geometry == "uniform_shock_slab"
        or geometry == "uniform_face_on_slab"
    ):
        return source_function * (1 - np.exp(-tau_freq)) * solid_angle

    elif geometry == "uniform_sphere":
        min_tau_freq = 1e-2
        tau_freq = np.array(tau_freq)
        stable_region = tau_freq > min_tau_freq
        with np.errstate(divide="ignore", invalid="ignore"):
            flux = (
                2
                * np.pi
                * source_function
                / tau_freq**2
                * (tau_freq**2 / 2 - 1 + (tau_freq + 1) * np.exp(-tau_freq))
            )
        flux_freq_taylor = (
            2
            * np.pi
            * source_function
            * (tau_freq / 3 - tau_freq**2 / 8 + tau_freq**3 / 30 - tau_freq**4 / 144)
        )
        flux_freq = np.where(stable_region, flux, flux_freq_taylor)
        assert np.all(np.isfinite(flux_freq))
        return flux_freq * solid_angle / np.pi


def get_escape_probability(geometry, tau_freq):
    if geometry == "taylor":
        tau_epsilon = 5e-5
        min_tau = -1

        tau_freq = np.atleast_1d(np.array(tau_freq))
        probability = np.ones_like(tau_freq) * np.inf
        stable_optical_depth_region = tau_freq > tau_epsilon
        probability[stable_optical_depth_region] = beta_analytical(
            tau_freq[stable_optical_depth_region]
        )
        small_tau_region = np.abs(tau_freq) < tau_epsilon
        probability[small_tau_region] = beta_taylor(tau_freq[small_tau_region])

        negative_tau_region = (tau_freq >= min_tau) & (tau_freq < -tau_epsilon)
        probability[negative_tau_region] = beta_analytical(
            tau_freq[negative_tau_region]
        )

        uncertain_negative_tau_region = tau_freq < min_tau
        probability[uncertain_negative_tau_region] = beta_analytical(
            np.abs(tau_freq[uncertain_negative_tau_region])
        )

        assert np.all(np.isfinite(probability))
        return probability

    elif geometry == "uniform_face_on_slab":
        theta = np.linspace(0, np.pi / 2, 200)
        tau_grid = np.logspace(-3, 2, 1000)
        min_tau_nu = np.min(tau_grid)

        integral_term_grid = np.array(
            [ut.integral_term(theta, tau) for tau in tau_grid]
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            probability = (
                ut.interpolated_integral_term(
                    theta, tau_freq, tau_grid, integral_term_grid
                )
                / tau_freq
            )

        probability = np.where(tau_freq < min_tau_nu, 1, probability)
        assert np.all(np.isfinite(probability))
        return probability


def beta_analytical(geometry, tau_freq):
    if geometry == "uniform_sphere":
        tau_freq = np.array(tau_freq)
        return (
            1.5
            / tau_freq
            * (
                1
                - 2 / tau_freq**2
                + (2 / tau_freq + 2 / tau_freq**2) * np.exp(-tau_freq)
            )
        )
    elif geometry == "uniform_shock_slab":
        return (1 - np.exp(-3 * tau_freq)) / (3 * tau_freq)


def beta_taylor(geometry, tau_freq):
    if geometry == "uniform_sphere":
        return (
            1
            - 0.375 * tau_freq
            + 0.1 * tau_freq**2
            - 0.0208333 * tau_freq**3
            + 0.00357143 * tau_freq**4
        )
    elif geometry == "uniform_shock_slab":
        return (
            1
            - (3 * tau_freq) / 2
            + (3 * tau_freq**2) / 2
            - (9 * tau_freq**3) / 8
            + (27 * tau_freq**4) / 40
            - (27 * tau_freq**5) / 80
        )
