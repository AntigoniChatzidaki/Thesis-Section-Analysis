from typing import List, Set, Tuple, Dict, Union

import numpy as np
import pandas as pd
import scipy
import shapely.geometry
from shapely.geometry import MultiPoint, Polygon, MultiPolygon, LinearRing
from descartes import PolygonPatch
import matplotlib.pyplot as plt

from Materials import Material, Concrete, Steel


def plot_polygon(polygon: Union[Polygon, MultiPolygon], ax: plt.Axes, color: str):
    if type(polygon) is shapely.geometry.MultiPolygon:
        for p in polygon.geoms:
            plot_polygon(p, ax, color)
        return
    # ext_x, ext_y = [list(x) for x in polygon.exterior.xy]
    # int_x = []
    # int_y = []
    # for ring in polygon.interiors:
    #     x, y = ring.xy
    #     int_x += list(x)
    #     int_y += list(y)
    # ax.fill(ext_x + int_x, ext_y + int_y, color=color)
    patch = PolygonPatch(polygon, facecolor=color, linewidth=0)
    ax.add_patch(patch)


def second_moment_of_area(ring: LinearRing) -> Dict[str, float]:
    xc, yc = ring.centroid.coords[0]
    Ix, Iy, Ixy = 0, 0, 0
    coords = ring.coords
    for (x1, y1), (x2, y2) in zip(coords, coords[1:] + [coords[0]]):
        x1 -= xc
        x2 -= xc
        y1 -= yc
        y2 -= yc
        v = x1 * y2 - x2 * y1
        Ix += v * (y1 ** 2 + y1 * y2 + y2 ** 2)
        Iy += v * (x1 ** 2 + x1 * x2 + x2 ** 2)
        Ixy += v * (x1 * y2 + 2 * x1 * y1 + 2 * x2 * y2 + x2 * y1)
    Ix /= 12
    Iy /= 12
    Ixy /= 24
    return {'Ixx': np.abs(Ix), 'Iyy': np.abs(Iy), 'Ixy': np.abs(Ixy)}


def rectangle(width, height, offset_x, offset_y) -> List[Tuple]:
    return [
        (offset_x, offset_y),
        (offset_x + width, offset_y),
        (offset_x + width, offset_y + height),
        (offset_x, offset_y + height)
    ]


def t_section(flange_width, flange_height, web_width, web_height, offset_x, offset_y) -> List[Tuple]:
    flange_ext = (flange_width - web_width) / 2
    return [
        (offset_x, offset_y),
        (offset_x + flange_width, offset_y),
        (offset_x + flange_width, offset_y + flange_height),
        (offset_x + flange_width - flange_ext, offset_y + flange_height),
        (offset_x + flange_width - flange_ext, offset_y + flange_height + web_height),
        (offset_x + flange_ext, offset_y + flange_height + web_height),
        (offset_x + flange_ext, offset_y + flange_height),
        (offset_x, offset_y + flange_height)
    ]


def I_section(top_flange_width, top_flange_height, web_width, web_height, offset_x, offset_y, bottom_flange_width=None,
              bottom_flange_height=None) -> List[Tuple]:
    top_flange_ext = (top_flange_width - web_width) / 2
    bottom_flange_ext = (bottom_flange_width - web_width) / 2
    return [
        (offset_x, offset_y),
        (offset_x + top_flange_width, offset_y),
        (offset_x + top_flange_width, offset_y + top_flange_height),
        (offset_x + top_flange_width - top_flange_ext, offset_y + top_flange_height),
        (offset_x + top_flange_width - top_flange_ext, offset_y + top_flange_height + web_height),
        (offset_x + top_flange_width - top_flange_ext + bottom_flange_ext, offset_y + top_flange_height + web_height),
        (offset_x + top_flange_width - top_flange_ext + bottom_flange_ext,
         offset_y + top_flange_height + web_height + bottom_flange_height),
        (offset_x + (top_flange_ext - bottom_flange_ext),
         offset_y + top_flange_height + web_height + bottom_flange_height),
        (offset_x + (top_flange_ext - bottom_flange_ext), offset_y + top_flange_height + web_height),
        (
            offset_x + bottom_flange_ext + (top_flange_ext - bottom_flange_ext),
            offset_y + top_flange_height + web_height),
        (offset_x + top_flange_ext, offset_y + top_flange_height),
        (offset_x, offset_y + top_flange_height)
    ]


def circle(diameter, offset_x=0.0, offset_y=0.0, sides=360) -> List[Tuple]:
    r = diameter / 2
    return [
        (r * np.cos(2 * np.pi * (t / sides)) + offset_x,
         r * np.sin(2 * np.pi * (t / sides)) + offset_y)
        for t in range(sides)
    ]


class Section:
    polygons: Dict[Material, Union[Polygon, MultiPolygon]]  # from materials to polygons

    concrete: Concrete = None

    @property
    def reinf_steel(self):
        if self.concrete is None:
            return None
        return self.concrete.reinf_steel

    @property
    def conf_steel(self):
        if self.concrete is None:
            return None
        return self.concrete.conf_steel

    def update_concrete(self):
        all_concretes = [material for material in self.materials if issubclass(material.__class__, Concrete)]
        if len(all_concretes) == 0:
            self.concrete = None
        elif len(all_concretes) == 1:
            self.concrete = all_concretes[0]
        else:
            raise RuntimeError("More than one concrete per section is not supported")

    # Properties need to be calculated on request
    neutral_axis: float = None
    slices: pd.DataFrame = None

    def __init__(self, polygons: Dict):
        self.polygons = polygons
        self.update_concrete()

    def __add__(self, other):
        merged_polygons = {}
        for material in self.materials | other.materials:
            if material in self.materials and material in other.materials:
                merged_polygons[material] = self.polygons[material].union(other.polygons[material])
            elif material in self.materials:
                merged_polygons[material] = self.polygons[material]
            else:
                merged_polygons[material] = other.polygons[material]
        return Section(merged_polygons)

    @property
    def materials(self) -> Set:
        return set(self.polygons.keys())

    def get_y_limits(self):
        # Find bounds of whole section
        min_y = 999
        max_y = -999
        for material, polygon in self.polygons.items():
            min_y = min(polygon.bounds[1], min_y)
            max_y = max(polygon.bounds[3], max_y)
        return min_y, max_y

    def get_areas_of_slice(self, offset_y: float, step: float) -> Dict[Material, float]:
        slice = Polygon(rectangle(2e3, step, -1e3, offset_y))
        out = {}
        for material, polygon in self.polygons.items():
            if type(polygon) is shapely.geometry.MultiPolygon:
                out[material] = sum([
                    slice.intersection(geom).area
                    for geom in polygon.geoms
                ])
            else:
                out[material] = slice.intersection(polygon).area
        return out

    def plot(self, ax):
        for material, polygon in self.polygons.items():
            plot_polygon(polygon.buffer(0), ax, material.color)

    def add_reinforcements(self, reinf_material, diameter, points: List[Tuple]):
        reinforcement = MultiPoint(points).buffer(diameter / 2)
        for material, polygon in self.polygons.items():
            self.polygons[material] = polygon.difference(reinforcement)
        if reinf_material in self.polygons:
            self.polygons[reinf_material] = self.polygons[reinf_material].union(reinforcement)
        else:
            self.polygons[reinf_material] = reinforcement
        self.update_concrete()

    def generate_slices(self, step=0.5e-3):
        min_y, max_y = self.get_y_limits()

        heights = np.arange(min_y + step, max_y + step, step)
        mid_heights = [y - step / 2 for y in heights]
        areas = {mat: [] for mat in self.materials}
        area_slices = [self.get_areas_of_slice(y - step, step) for y in heights]
        for slice in area_slices:
            for mat, area in slice.items():
                areas[mat].append(area)

        self.slices = pd.DataFrame({
            'height': heights,
            'mid_height': mid_heights,
            'concrete_area': areas.get(self.concrete, np.zeros(len(heights))),
            'reinf_steel_area': areas.get(self.reinf_steel, np.zeros(len(heights))),
            'conf_steel_area': areas.get(self.conf_steel, np.zeros(len(heights))),
        })
        # dict = {'height': heights, 'mid_height': mid_heights}
        # if self.concrete is not None:
        #     dict |= {'concrete_area': self.polygons[self.concrete].area}
        # if self.reinf_steel is not None:
        #     dict |= {'reinf_steel_area': self.polygons[self.reinf_steel].area}
        # if self.reinf_steel is not None:
        #     dict |= {'confining_steel_area': self.polygons[self.reinf_steel].area}
        # self.slices = pd.DataFrame(dict)

    def calculate_forces(self, axial_force: float, confined_factor_h_a: float = 0, confined_factor_h_c: float = 0.0,
                         outer_thickness: float = 0.0, outer_depth: float = 0.0):
        if self.slices is None:
            raise RuntimeError("Need to generate slices first")
        if self.neutral_axis is None:
            raise RuntimeError("Need to define a neutral axis")

        def strain_fun(slice):
            return self.concrete.ultimate_strain / self.neutral_axis * (self.neutral_axis - slice.mid_height)

        def reinf_steel_stress_fun(slice):
            if slice.reinf_steel_area > 0:
                if abs(slice.strain) < self.reinf_steel.characteristic_strength / 1.15 / self.reinf_steel.youngs_modulus:
                    steel_stress = slice.strain * self.reinf_steel.youngs_modulus
                else:
                    steel_stress = self.reinf_steel.characteristic_strength / 1.15 * abs(slice.strain) / slice.strain
            else:
                steel_stress = 0
            return steel_stress

        def conf_steel_stress_fun(slice):
            if slice.conf_steel_area > 0:
                if abs(slice.strain) < self.conf_steel.characteristic_strength / self.conf_steel.youngs_modulus:
                    steel_stress = slice.strain * self.conf_steel.youngs_modulus
                else:
                    steel_stress = self.conf_steel.characteristic_strength * abs(slice.strain) / slice.strain
            else:
                steel_stress = 0
            return steel_stress

        def concrete_stress_fun(slice):
            if slice.strain < 0:
                concrete_stress = 0
            elif slice.strain < self.concrete.critical_strain:
                concrete_stress = self.concrete.design_strength * (
                        1 - (1 - slice.strain / self.concrete.critical_strain) ** self.concrete.n)
            else:  # slice.strain < concrete.ultimate_strain:
                concrete_stress = self.concrete.design_strength
            return concrete_stress

        self.slices['strain'] = self.slices.apply(strain_fun, axis=1)
        if self.concrete is not None:
            self.slices['concrete_stress'] = self.slices.apply(concrete_stress_fun, axis=1)
        if self.reinf_steel is not None:
            self.slices['reinf_steel_stress'] = self.slices.apply(reinf_steel_stress_fun, axis=1)
        if self.conf_steel is not None:
            self.slices['conf_steel_stress'] = self.slices.apply(conf_steel_stress_fun, axis=1)

        if self.conf_steel is None:
            self.slices['concrete_force'] = self.slices.concrete_stress * self.slices.concrete_area
            self.slices['reinf_steel_force'] = self.slices.reinf_steel_stress * self.slices.reinf_steel_area
            self.slices['conf_steel_force'] = self.slices.conf_steel_stress * self.slices.conf_steel_area

        else:
            self.slices['concrete_force'] = self.slices.concrete_stress * self.slices.concrete_area * \
                                            (1 + confined_factor_h_c * outer_thickness * outer_depth *
                                             self.slices.conf_steel_stress / self.slices.concrete_stress)
            self.slices['reinf_steel_force'] = self.slices.reinf_steel_stress * self.slices.reinf_steel_area
            self.slices['conf_steel_force'] = (self.slices.conf_steel_stress * self.slices.conf_steel_area *
                                              confined_factor_h_a)

        self.slices[
            'total_force'] = self.slices.concrete_force + self.slices.reinf_steel_force + self.slices.conf_steel_force
        min_y, max_y = self.get_y_limits()
        mid_height = (max_y + min_y) / 2
        self.slices['total_moment'] = self.slices.total_force * (mid_height - self.slices.mid_height)
        # self.slices['steel_moment'] = self.slices.steel_force * (mid_height - self.slices.mid_height)

        total_force = self.slices.total_force.sum()
        sum_force = total_force - axial_force

        return total_force, sum_force

    def calculate_neutral_axis(self, axial_force: float, confined_factor_h_a: float = 0.0,
                               confined_factor_h_c: float = 0.0, outer_thickness: float = 0.0,
                               outer_depth: float = 0.0):
        max_y = self.get_y_limits()

        # Returns total force for an assumed neutral axis
        def optim_fun(guess: float) -> float:
            self.neutral_axis = guess
            total_force, sum_force = self.calculate_forces(axial_force, confined_factor_h_a,
                               confined_factor_h_c, outer_thickness,
                               outer_depth)
            return sum_force

        optim_results = scipy.optimize.root_scalar(
            optim_fun,
            x0=0.3,
            x1=0.2
        )
        self.neutral_axis = optim_results.root

        # if optim_results.root < max_y:
        #     self.neutral_axis = optim_results.root
        # else:
        #     self.neutral_axis = max_y

    @property
    def second_moments_of_area(self) -> Dict[Material, Dict[str, float]]:
        def _net_second_moment(polygon: Polygon):
            exterior = second_moment_of_area(polygon.exterior)
            interiors = [second_moment_of_area(interior) for interior in polygon.interiors]
            return {
                key: exterior[key] - sum([interior[key] for interior in interiors])
                for key in exterior.keys()
            }

        out = {}
        for material, polygons in self.polygons.items():
            if type(polygons) is MultiPolygon:
                moments = [_net_second_moment(polygon) for polygon in polygons.geoms]
                out[material] = moments[0]
                for moment in moments[1:]:
                    for key, val in moment.items():
                        out[material][key] += val
            else:
                out[material] = _net_second_moment(polygons)
        return out

    @property
    def characteristic_load(self) -> float:
        if self.concrete is None:
            concrete_area = 0
            concrete_characteristic_strength = 0
        else:
            concrete_area = self.polygons[self.concrete].area
            concrete_characteristic_strength = self.concrete.characteristic_strength
        if self.conf_steel is None:
            conf_steel_area = 0
            conf_steel_characteristic_strength = 0
        else:
            conf_steel_area = self.polygons[self.conf_steel].area
            conf_steel_characteristic_strength = self.conf_steel.characteristic_strength
        if self.reinf_steel is None:
            reinf_steel_area = 0
            reinf_steel_characteristic_strength = 0
        else:
            reinf_steel_area = self.polygons[self.reinf_steel].area
            reinf_steel_characteristic_strength = self.reinf_steel.characteristic_strength

        return (concrete_area * concrete_characteristic_strength
                + conf_steel_area * conf_steel_characteristic_strength
                + reinf_steel_area * reinf_steel_characteristic_strength)


class HollowRectangular(Section):
    def __init__(self, material: Material,
                 outer_width: float, outer_height: float,
                 inner_width: float, inner_height: float,
                 top_cover: float, offset_x: float = 0, offset_y: float = 0):
        left_cover = (outer_width - inner_width) / 2

        polygons = {material: Polygon(
            rectangle(outer_width, outer_height, offset_x, offset_y),
            [rectangle(inner_width, inner_height, left_cover + offset_x, top_cover + offset_y)]
        )}
        super().__init__(polygons)


class Rectangular(Section):
    def __init__(self, material: Material,
                 width: float, height: float, offset_x: float = 0, offset_y: float = 0):
        polygons = {material: Polygon(
            rectangle(width, height, offset_x, offset_y))}
        super().__init__(polygons)


class TSection(Section):
    def __init__(self, material: Material,
                 flange_width: float, flange_height: float,
                 web_width: float, web_height: float, offset_x: float, offset_y: float):
        polygons = {material: Polygon(
            t_section(flange_width, flange_height, web_width, web_height, offset_x, offset_y)
        )}
        super().__init__(polygons)


class ISection(Section):
    def __init__(self, material: Material,
                 top_flange_width: float, top_flange_height: float,
                 web_width: float, web_height: float, offset_x: float, offset_y: float, bottom_flange_width: float,
                 bottom_flange_height: float):
        polygons = {material: Polygon(
            I_section(top_flange_width, top_flange_height,
                      web_width, web_height,
                      offset_x, offset_y,
                      bottom_flange_width, bottom_flange_height)
        )}
        super().__init__(polygons)


class HollowCircular(Section):
    def __init__(self, material: Material,
                 outer_diameter: float, thickness: float,
                 offset_x: float, offset_y: float):
        inner_diameter = outer_diameter - 2 * thickness
        # put cylinder in the middle, otherwise add offset

        polygons = {material: Polygon(
            circle(outer_diameter, offset_x, offset_y),
            [circle(inner_diameter, offset_x, offset_y)]
        )}
        super().__init__(polygons)
