import warnings

import numpy as np
import scipy.linalg as la

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
                k_12, k_21 = coeffs["K12"], coeffs["K21"]
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
        if line_profile == "gaussian":
            line_profile_type = sp.GaussianLineProfile
        elif line_profile == "rectangular":
            line_profile_type = sp.RectangularLineProfile
        self.emitting_molecule = EmittingMolecule.from_LAMDA_datafile(
            file_path,
            line_profile_type,
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
    
    def get_escape_probability_forall(self, level_populations):
        beta = []
        for line in self.emitting_molecule.radiative_transitions:
            N_1 = self.total_column_density * level_populations[line.low.number]
            N_2 = self.total_column_density * level_populations[line.up.number]
            beta_freq_arr = esc.get_escape_probability(
                 self.geometry, line.tau_nu_array(N_1, N_2),
            )
            averaged_beta = line.line_profile.average_over_nu_array(beta_freq_arr)
            beta.append(averaged_beta)
        
        return np.array(beta)

    def solve_radiative_transfer(self):
        beta_lines = np.ones(self.emitting_molecule.n_rad_transitions)
        i_ext_lines = np.array(
            [
                self.background(line.nu0)
                for line in self.emitting_molecule.radiative_transitions
            ]
        )
        level_populations = self.rate_matrix.solve(
            beta_arr=beta_lines, i_ext_arr=i_ext_lines
        )
        excitation_temp_residual = np.ones(
            self.emitting_molecule.n_rad_transitions
        ) * np.inf
        excitation_temp_old = 0
        counter = 0

        while (
            np.any(excitation_temp_residual > self.relative_convergence) or counter < self.min_iter
        ):
            counter += 1
            if counter % 10 == 0 and self.verbose:
                print(f"Iteration {counter:4d}")
            if counter > self.max_iter:
                raise RuntimeError("Max iterations reached")
            new_level_populations = self.rate_matrix.solve(
                beta_arr=beta_lines, i_ext_arr=i_ext_lines
            )
            excitation_temp = self.emitting_molecule.get_Tex(new_level_populations)
            
            excitation_temp_residual = ut.get_relative_difference(
                excitation_temp, excitation_temp_old
            )
            if self.verbose:
                print(f"Maximum relative excitation temperature residual: {np.max(excitation_temp_residual):.2e}")
            
            excitation_temp_old = excitation_temp.copy()

            level_populations = (
                self.relaxation_factor * new_level_populations
                + (1 - self.relaxation_factor) * level_populations
            )
            beta_lines = self.get_escape_probability_forall(level_populations)
        if self.verbose:
            print(f"Converged in {counter} iterations")
        
        self.tau_nu0 = self.emitting_molecule.get_tau_nu0(
            self.total_column_density, level_populations
        )
        if np.any(self.tau_nu0 < 0):
            negative_lines = np.where(self.tau_nu0 < 0)[0]
            negative_transitions = [
                self.emitting_molecule.radiative_transitions[line]
                for line in negative_lines
            ]
            warnings.warn(
                f"Negative optical depths for transitions {negative_transitions}"
            )
            for line, transition in zip(
                negative_lines, negative_transitions
            ):
                print(f"Transition {transition} has negative optical depth = {self.tau_nu0[line]}")

        self.level_populations = level_populations
        self.excitation_temp = self.emitting_molecule.get_Tex(level_populations)


    def get_line_fluxes(self, solid_angle):
        self.obs_line_fluxes = []
        self.obs_line_spectra = []

        for i, line in enumerate(self.emitting_molecule.radiative_transitions):
            freq_arr = line.line_profile.freq_array
            x1 = self.level_populations[line.low.number]
            x2 = self.level_populations[line.up.number]
            source_function = ut.plank_freq(freq_arr, self.excitation_temp[i])
            tau_freq = line.tau_nu_array(
                N1 = x1 * self.total_column_density,
                N2 = x2 * self.total_column_density,
            )
            line_flux_freq = esc.get_flux(
                self.geometry,
                source_function,
                tau_freq,
                solid_angle,
            ) 
            self.obs_line_spectra.append(line_flux_freq)
            line_flux = np.trapz(line_flux_freq, freq_arr)
            self.obs_line_fluxes.append(line_flux)

    def display_results(self):
        print("\n")
        print(
            "  up   low      nu [GHz]    T_ex [K]      poplow         popup"
            + "         tau_nu0"
        )
        for i, line in enumerate(self.emitting_molecule.radiative_transitions):
            rad_trans_string = (
                "{:>4d} {:>4d} {:>14.6f} {:>10.2f} {:>14g} {:>14g} {:>14g}"
            )
            rad_trans_format = (
                line.up.number,
                line.low.number,
                line.nu0 / 1e9,
                self.excitation_temp[i],
                self.level_populations[line.low.number],
                self.level_populations[line.up.number],
                self.tau_nu0[i],
            )
            print(rad_trans_string.format(*rad_trans_format))
        print("\n")
    
