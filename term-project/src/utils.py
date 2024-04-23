import numpy as np

from . import constants as cc


def integral_term(theta, tau):
    return np.trapz(
        (1 - np.exp(-tau / np.cos(theta))) * np.cos(theta) * np.sin(theta),
        theta,
    )


def interpolated_integral_term(theta, tau, tau_grid, integral_term_grid):
    interp = np.interp(x=tau, xp=tau_grid, fp=integral_term_grid, left=0, right=0.5)
    return np.where(tau < np.min(tau_grid), tau, interp)


def einst_B21(einst_A21, nu):
    return cc.SPEED_OF_LIGHT**2 / (2 * cc.PLANK_CONSTANT * nu**3) * einst_A21


def einst_B12(einst_A21, nu, g1, g2):
    return g2 / g1 * einst_B21(einst_A21, nu)


def plank_freq(freq, temp):
    temp = np.array(temp)
    return (
        2
        * cc.PLANK_CONSTANT
        * freq**3
        / cc.SPEED_OF_LIGHT**2
        * (np.exp(cc.PLANK_CONSTANT * freq / (cc.BOLTZMANN_CONSTANT * temp)) - 1) ** -1
    )

def get_background(z=0):
    temp_cmb = 2.73 * (1 + z)

    def cmb(nu):
        return plank_freq(freq=nu, temp=temp_cmb)

    return cmb

def no_background(nu):
    return np.zeros_like(nu)

def get_std_deviation(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))

def get_relative_difference(a, b):
    abs_diff = np.abs(a - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_diff = np.where((a == 0) & (b == 0), 0, np.where(a == 0, 1, abs_diff / a))
    
    assert not np.any(np.isnan(relative_diff))
    return np.abs(relative_diff)

def get_frequency_interval(velocity_interval, frequency):
    return frequency * velocity_interval / cc.SPEED_OF_LIGHT