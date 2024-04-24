# Minimum example of the usage of pythonradex

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from . import cloud
from . import constants as cc
from . import utils as ut
from .lte import LTE


def simulate_spectra(lte=False):
    if lte:
        simulate_lte()
    else:
        simulate_non_lte()


def simulate_lte():
    spec = LTE(
        species="co",
        part_name="/home/kmaitreys/Documents/college/10-2024-spring/p452-computational-physics/term-project/src/resources/cdms_partition_functions.dat",
        idx_obs=0,
        CDMS_file="/home/kmaitreys/Documents/college/10-2024-spring/p452-computational-physics/term-project/src/resources/co.cat",
    )
    print(spec)



def simulate_non_lte():
    data_filepath = "/home/kmaitreys/Documents/pegasis-projects/synthspec/pythonradex-master/examples/co.dat"  # relative or absolute path to the LAMDA datafile
    geometry = "uniform_sphere"
    # spectral radiance of the background in units of [W/m2/Hz/sr].
    # This is simply a function of nu. Here using a pre-defined function, namely the
    # Planck function at 2.73 K (CMB), but one can define its own if wished
    ext_background = ut.get_background()
    Tkin = 150  # kinetic temperature of the colliders
    # number density of the collsion partners in [m-3]
    coll_partner_densities = {
        "para-H2": 100 / (1e-2) ** 3,
        "ortho-H2": 250 / (1e-2) ** 3,
    }
    Ntot = 1e18 / (1e-2) ** 2  # column density in [m-2], i.e. here 1e16 cm-2
    line_profile = "gaussian"  # type of the line profile
    width_v = 2 * 1e3  # witdh of the line in m/s

    example_cloud = cloud.Cloud(
        file_path=data_filepath,
        geometry=geometry,
        background=ext_background,
        kinetic_temp=Tkin,
        collisional_partner_densities=coll_partner_densities,
        total_column_density=Ntot,
        line_profile=line_profile,
        velocity_width=width_v,
        verbose=False,
    )
    example_cloud.solve_radiative_transfer()
    example_cloud.display_results()  # outputs a table with results for all transitions

    # examples of how to access the results directly:
    # excitation temperature of second (as listed in the LAMDA file) transition:
    print("Tex for second transition: {:g} K".format(example_cloud.excitation_temp[1]))
    # fractional population of 4th level:
    print(
        "fractional population of 4th level: {:g}".format(
            example_cloud.level_populations[3]
        )
    )
    # optical depth of lowest transition at line center:
    print("optical depth lowest transition: {:g}".format(example_cloud.tau_nu0[0]))

    # compute the flux observed for a target at 20 pc with a radius of 3 au
    d_observer = 20 * cc.PARSEC
    source_radius = 3 * cc.AU
    solid_angle = source_radius**2 * np.pi / d_observer**2
    # a list of observed fluxes for all transitions:
    example_cloud.get_line_fluxes(solid_angle=solid_angle)
    # for flux in example_nebula.obs_line_fluxes:
    #     print(f"observed flux: {flux} W/m2")
    # Create figure and axis
    fig, ax = plt.subplots()
    ax.set_xlabel("velocity [km/s]")
    ax.set_ylabel("flux [W/m2]")

    def update(frame):
        ax.clear()
        spectra = example_cloud.obs_line_spectra[frame]
        x = np.linspace(-350, 350, len(spectra))
        spectra[spectra == 0] = 1e-100    
        ax.plot(x, spectra, label=f"Synthetic spectra for line {example_cloud.emitting_molecule.radiative_transitions[frame].name}")
        ax.legend()
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(example_cloud.obs_line_spectra), interval=200)

    # Save the animation as a GIF
    ani.save('animated_plot.gif', writer='imagemagick')

    for i, spectra in enumerate(example_cloud.obs_line_spectra):
        if i % 4 == 0:
            # Start x-axis label at -10 km/s
            x = np.linspace(-350, 350, len(spectra))
            spectra[spectra == 0] = 1e-100    
            plt.plot(x, spectra, label=f"Synthetic spectra for line {example_cloud.emitting_molecule.radiative_transitions[i].name}")
            plt.xlabel("velocity [km/s]")
            plt.ylabel("flux [W/m2]")
            plt.legend()

    plt.title("Observed spectra")
    plt.show()


    print(
        "observed flux of second transition: {:g} W/m2".format(
            example_cloud.obs_line_fluxes[1]
        )
    )


if __name__ == "__main__":
    simulate_spectra()
