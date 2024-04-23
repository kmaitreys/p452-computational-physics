# Minimum example of the usage of pythonradex

import matplotlib.pyplot as plt
import numpy as np

from . import cloud
from . import constants as cc
from . import utils as ut

data_filepath = "/home/kmaitreys/Documents/pegasis-projects/synthspec/pythonradex-master/examples/co.dat"  # relative or absolute path to the LAMDA datafile
geometry = "uniform_shock_slab"
# spectral radiance of the background in units of [W/m2/Hz/sr].
# This is simply a function of nu. Here using a pre-defined function, namely the
# Planck function at 2.73 K (CMB), but one can define its own if wished
ext_background = ut.get_background()
Tkin = 150  # kinetic temperature of the colliders
# number density of the collsion partners in [m-3]
coll_partner_densities = {"para-H2": 100 / (1e-2) ** 3, "ortho-H2": 250 / (1e-2) ** 3}
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
print("fractional population of 4th level: {:g}".format(example_cloud.level_populations[3]))
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

for spectra in example_cloud.obs_line_spectra:
    plt.plot(spectra)

plt.show()


print(
    "observed flux of second transition: {:g} W/m2".format(
        example_cloud.obs_line_fluxes[1]
    )
)
