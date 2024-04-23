import warnings

import numpy as np
import scipy.linalg as la

from . import constants as cc
from . import escape as esc
from . import spec as sp
from . import utils as ut
from .molecule import EmittingMolecule


class RateMatrix:
    def __init__(
        self, molecule, collisional_partner_densities, kinetic_temp, mode="standard"
    ):
        self.molecule = molecule
        self.collisional_partner_densities = collisional_partner_densities
        self.kinetic_temp = kinetic_temp
        assert mode in ["standard", "ALI"]
        if mode == "standard":
            self.radiative_rates = self.radiative_rates_standard
        elif mode == "ALI":
            self.radiative_rates = self.radiative_rates_ALI

    def empty_rate_matrix(self):
        return np.zeros((self.molecule.n_levels, self.molecule.n_levels))

    def radiative_rates_standard(self, j_bar_arr):
        excite_rates = np.array(
            [
                line.B12 * j_bar_arr[i]
                for i, line in enumerate(self.molecule.radiative_transitions)
            ]
        )
        dexcite_rates = np.array(
            [
                line.A21 + line.B21 * j_bar_arr[i]
                for i, line in enumerate(self.molecule.radiative_transitions)
            ]
        )

        return excite_rates, dexcite_rates

    def radiative_rates_ALI(self, beta_arr, i_ext_arr):
        excite_rates = np.array(
            [
                line.B12 * i_ext * beta
                for line, i_ext, beta in zip(
                    self.molecule.radiative_transitions, i_ext_arr, beta_arr
                )
            ]
        )

        dexcite_rates = np.array(
            [
                line.A21 * beta + line.B21 * i_ext * beta
                for line, i_ext, beta in zip(
                    self.molecule.radiative_transitions, i_ext_arr, beta_arr
                )
            ]
        )

        return excite_rates, dexcite_rates

    def generate_radiative_rates(self, matrix, radiative_rates):
        excite_rates, dexcite_rates = radiative_rates

        for up, down, line in zip(
            excite_rates, dexcite_rates, self.molecule.radiative_transitions
        ):
            matrix[line.low.number, line.up.number] += down
            matrix[line.up.number, line.up.number] += -down
            matrix[line.up.number, line.low.number] += up
            matrix[line.low.number, line.low.number] += -up

    def generate_collisional_rates(self, matrix):
        for partner, density in self.collisional_partner_densities.items():
            for transition in self.molecule.collisional_transitions[partner]:
                coeffs = transition.coeffs(self.kinetic_temp)
                k_12, k_21 = coeffs["k_12"], coeffs["k_21"]
                matrix[transition.up.number, transition.low.number] += k_12 * density
                matrix[transition.low.number, transition.low.number] += -k_12 * density
                matrix[transition.low.number, transition.up.number] += k_21 * density
                matrix[transition.up.number, transition.up.number] += -k_21 * density

    def solve(self, **kwargs):
        matrix = self.empty_rate_matrix()
        self.generate_radiative_rates(matrix, self.radiative_rates(**kwargs))
        self.generate_collisional_rates(matrix)
        matrix[0, :] = np.ones(self.molecule.n_levels)
        b = np.zeros(self.molecule.n_levels)
        b[0] = 1

        fractional_population = la.solve(matrix, b)
        assert np.all(fractional_population >= 0)

        return fractional_population


class Cloud:
    relative_convergence = 1e-2
    min_iter = 30
    max_iter = 1000

    relaxation_factor = 0.3

    geometries = ["uniform_sphere", "uniform_face_on_slab", "uniform_shock_slab"]
    line_profiles = ["gaussian", "rectangular"]

    def __init__(
            self,
            file_path,
            geometry,
            background,
            kinetic_temp,
            collisional_partner_densities,
            total_column_density,
            line_profile,
            velocity_width,
            partition_function = None,
            verbose = False,
    ):
        self.emitting_molecule = EmittingMolecule.from_LAMDA_datafile(
            file_path,
            line_profile,
            velocity_width,
            partition_function,
        )
        self.geometry = geometry
        self.background = background
        self.kinetic_temp = kinetic_temp
        self.collisional_partner_densities = collisional_partner_densities
        self.total_column_density = total_column_density
        self.rate_matrix = RateMatrix(
            self.emitting_molecule,
            self.collisional_partner_densities,
            self.kinetic_temp,
            "ALI",
        )
        self.verbose = verbose
    
    def get_escape_probabilities(self, level_populations):
        beta = []
        for line in self.emitting_molecule.rad_transitions:
            N_1 = self.total_column_density * level_populations[line.low.number]
            N_2 = self.total_column_density * level_populations[line.up.number]
            beta_freq_arr = esc.get_escape_probability(
                 self.geometry, line.tau_nu_array(N_1, N_2),
            )
            averaged_beta = line.line_profile.average_over_nu_array(beta_freq_arr)
            beta.append(averaged_beta)
        
        return np.array(beta)

    def solve_radiative_transfer(self):
        ...
    
    def get_line_fluxes(self, solid_angle):
        ...

    def display_results(self):
        ...
    
