import sys

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style()

from Materials import Material, Concrete
from Sections import Rectangular, Section, HollowRectangular

# frp = Material('FRP', 600e6, 80e9, 'black')
concrete = Concrete('Concrete', 37e9, 40e6, 'gray', confining_steel=False, frp=frp)
steel = Material('Steel', 200e9,500e6, 'blue')  # Pa


# def neutral_axis_optim(section: Section, y_neutral: float) -> Tuple[float, float]:
#     step = 0.001
#
#     # Find bounds of whole section
#     minY = 99999
#     maxY = -99999
#     for material, polygon in section.polygons.items():
#         minY = min(polygon.bounds[1], minY)
#         maxY = max(polygon.bounds[3], maxY)
#
#     #ns = [i * step for i in range(1, int(height / step))]
#     # e_cu: Concrete(Material)
#     ys = np.arange(minY, maxY, step)
#
#     total_force = 0
#     total_moment = 0
#     axial_force = 2780000 # N
#     for y in ys:
#         slice_polygon = Polygon(rectangle(2, step, -1, y))
#         #y_centroid = slice_polygon.centroid.y
#         y_centroid = y - step / 2
#         # for material, polygon in test_concrete_section.polygons.items():
#         #     slice = slice_polygon.intersection(polygon)
#             #y_centroid = slice.centroid.y
#         strain_slice = concrete.ultimate_strain / y_neutral * (y_neutral - y_centroid)
#         # steel stress
#         if abs(strain_slice) < steel.characteristic_strength / steel.youngs_modulus:
#                 steel_stress = strain_slice / steel.youngs_modulus
#         else:
#             steel_stress = steel.characteristic_strength * abs(strain_slice) / strain_slice
#         # concrete stress
#         if concrete.conf_steel:
#             if strain_slice < 0:
#                 concrete_stress = 0
#             elif strain_slice < concrete.top_strain:
#                 concrete_stress = concrete.youngs_modulus * strain_slice - (
#                             concrete.youngs_modulus - concrete.confined_youngs_modulus) ** 2 / (
#                                               4 * concrete.characteristic_strength) * strain_slice ** 2
#             elif strain_slice < concrete.ultimate_strain:
#                 concrete_stress = concrete.characteristic_strength + concrete.confined_youngs_modulus * strain_slice
#             else:
#                 concrete_stress = 0
#         else:
#             if strain_slice < 0:
#                 concrete_stress = 0
#             elif strain_slice < concrete.critical_strain:
#                 concrete_stress = concrete.design_strength * (
#                             1 - (1 - strain_slice / concrete.critical_strain) ** concrete.n)
#             elif strain_slice < concrete.ultimate_strain:
#                 concrete_stress = concrete.design_strength
#             else:
#                 concrete_stress = 0
#         # Areas
#         areas = section.get_areas_of_slice(y, step)
#         # steel area
#         steel_area = areas[steel]
#         # concrete area
#         concrete_area = areas[concrete]
#         # steel force
#         steel_force = steel_stress * steel_area
#         # concrete force
#         concrete_force = concrete_stress * concrete_area
#         # Slice force
#         slice_force = steel_force + concrete_force
#         # Total force (section forces)
#         total_force += slice_force
#         # Sum force (external forces, i.e. Axial load included)
#         sum_force = total_force - axial_force
#         # Total moment
#         total_moment += slice_force * (y_neutral - y_centroid)
#     return (sum_force, total_moment)
#

# TESTS

test_section = HollowRectangular(concrete, 0.600, 0.600, 0.34, 0.34, 0.130)
test_section.add_reinforcements(steel, 0.016, [(0.065,0.065) , (0.535,0.065) , (0.065,0.535) , (0.535,0.535) , (0.1825,0.065) , (0.065,0.1825) , (0.535,0.1825) , (0.1825,0.535) , (0.3,0.065) , (0.065,0.3) , (0.3,0.535) , (0.535,0.3) , (0.4175,0.065) , (0.065,0.4175) , (0.4175,0.535) , (0.535,0.4175)
])
# test_concrete_section = Rectangular(concrete, 0.600, 0.600)
# test_concrete_section.add_reinforcements(steel, 0.02, [(0.065,0.065) , (0.535,0.065) , (0.065,0.535) , (0.535,0.535)])

test_section.generate_slices()
test_section.calculate_neutral_axis(670e3, concrete, steel)
print(test_section.neutral_axis)
print(test_section.slices.dtypes)
#test_concrete_section.slices.to_csv("output.csv")

fig, axs = plt.subplots(1, 4, sharey=True, squeeze=True)
axs[0].invert_yaxis()

test_section.plot(axs[0])


plt.show()

sys.exit()





# test_concrete_section.add_reinforcements(steel, 0.016, [(0.065, 0.049), (0.3, 0.049), (0.535, 0.049)])

test_section.plot(sections_ax)
#  (steel, 0.016, [(0.065, 0.049), (0.3, 0.0535), (0.5875, 0.049)])

ys = np.arange(0, 1, 0.001)
plt.plot([test_concrete_section.get_areas_of_slice(y, 0.01)[concrete] for y in ys], ys)


optim_results = scipy.optimize.root_scalar(
    lambda x: neutral_axis_optim(test_concrete_section, x)[0],
    x0=0.01,
    x1=0.02
)
neutral_axis = optim_results.root
print(neutral_axis)

sum_force, total_moment = neutral_axis_optim(test_concrete_section, neutral_axis)
print(f"Sum force: {sum_force:.2f} N, Total moment: {total_moment:.2f} Nm")

# print(total_moment)

# xs = np.linspace(0.0, 0.02, 20)
# # plt.clf()
# plt.plot(xs, [neutral_axis_optim(x) for x in xs], 'r.')
# #print([neutral_axis_optim(x) for x in xs])
#
# print(test_concrete_section.get_area_of_slice(1, 0.01))
# #test_concrete_section.plot()





#sections_ax.set(xlim=[0, 0.5], ylim=[0, 0.5])
sections_ax.invert_yaxis()
#sections_ax.set_xlim(0.0, 0.6)
#sections_ax.set_ylim(0.0, 0.6)
plt.show()


# print(test_concrete_section.get_area_of_slice(1, 0.01))

#test_concrete_section.plot()

# print(neutral_axis_optim(test_concrete_section, 0.021289166))
# print(neutral_axis_optim(test_concrete_section, 0.02))
# print(neutral_axis_optim(test_concrete_section, 0.01))
# print(neutral_axis_optim(test_concrete_section, 0.007))

# xs = np.linspace(0, 0.04, 40)
# plt.plot(xs, [neutral_axis_optim(test_concrete_section, x) for x in xs])
# plt.show()
