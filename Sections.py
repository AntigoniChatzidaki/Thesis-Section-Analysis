from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd
import scipy
import shapely.geometry
from shapely.geometry import MultiPoint, Polygon, MultiPolygon
from descartes import PolygonPatch
import matplotlib.pyplot as plt

from Materials import Material, Concrete


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
    materials: List[Material]
    polygons: Dict[Material, Polygon]  # from materials to polygons

    # Properties need to be calculated on request
    neutral_axis: float = None
    slices: pd.DataFrame = None

    def __init__(self, polygons: Dict):
        self.polygons = polygons
        self.materials = list(polygons.keys())

    def __add__(self, other):
        return Section(self.polygons | other.polygons)

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
            self.materials.append(reinf_material)

    def generate_slices(self, step=0.5e-3):
        min_y, max_y = self.get_y_limits()

        heights = np.arange(min_y + step, max_y, step)
        mid_heights = [y - step / 2 for y in heights]
        areas = {mat: [] for mat in self.materials}
        area_slices = [self.get_areas_of_slice(y, step) for y in heights]
        for slice in area_slices:
            for mat, area in slice.items():
                areas[mat].append(area)
        self.slices = pd.DataFrame(
            {
                'height': heights,
                'mid_height': mid_heights,
                **{
                    mat.__class__.__name__ + "_area": area
                    for mat, area in areas.items()
                }
            }
        )

    def calculate_forces(self, axial_force: float, concrete: Concrete, steel: Material):
        if self.slices is None:
            raise RuntimeError("Need to generate slices first")
        if self.neutral_axis is None:
            raise RuntimeError("Need to define a neutral axis")

        def strain_fun(slice):
            return concrete.ultimate_strain / self.neutral_axis * (self.neutral_axis - slice.mid_height)

        def steel_stress_fun(slice):
            if slice.Steel_area > 0:
                if abs(slice.strain) < steel.strength / 1.15 / steel.youngs_modulus:
                    steel_stress = slice.strain * steel.youngs_modulus
                else:
                    steel_stress = steel.strength / 1.15 * abs(slice.strain) / slice.strain
            else:
                steel_stress = 0
            return steel_stress

        def concrete_stress_fun(slice):
            if concrete.confining_steel is not None:
                if slice.strain < 0:
                    concrete_stress = 0
                elif slice.strain < concrete.top_strain:
                    concrete_stress = concrete.youngs_modulus * slice.strain - (
                                concrete.youngs_modulus - concrete.confined_youngs_modulus) ** 2 / (
                                                  4 * concrete.strength) * slice.strain ** 2
                elif slice.strain < concrete.ultimate_strain:
                    concrete_stress = concrete.strength + concrete.confined_youngs_modulus * slice.strain
                else:
                    concrete_stress = 0
            else:
                if slice.strain < 0:
                    concrete_stress = 0
                elif slice.strain < concrete.critical_strain:
                    concrete_stress = concrete.design_strength * (
                                1 - (1 - slice.strain / concrete.critical_strain) ** concrete.n)
                elif slice.strain < concrete.ultimate_strain:
                    concrete_stress = concrete.design_strength
                else:
                    concrete_stress = 0
            return concrete_stress

        self.slices['strain'] = self.slices.apply(strain_fun, axis=1)
        self.slices['steel_stress'] = self.slices.apply(steel_stress_fun, axis=1)
        self.slices['concrete_stress'] = self.slices.apply(concrete_stress_fun, axis=1)
        self.slices['steel_force'] = self.slices.steel_stress * self.slices.Steel_area
        self.slices['concrete_force'] = self.slices.concrete_stress * self.slices.Concrete_area
        self.slices['total_force'] = self.slices.steel_force + self.slices.concrete_force
        min_y, max_y = self.get_y_limits()
        mid_height = (max_y + min_y) / 2
        self.slices['total_moment'] = self.slices.total_force * (mid_height - (self.slices.mid_height))
        self.slices['steel_moment'] = self.slices.steel_force * (mid_height - (self.slices.mid_height))


        total_force = self.slices.total_force.sum()
        sum_force = total_force - axial_force

        return total_force, sum_force

    def calculate_neutral_axis(self, axial_force: float, concrete: Concrete, steel: Material):
        # Returns total force for an assumed neutral axis
        def optim_fun(guess: float) -> float:
            self.neutral_axis = guess
            total_force, sum_force = self.calculate_forces(axial_force, concrete, steel)
            return sum_force

        optim_results = scipy.optimize.root_scalar(
            optim_fun,
            x0=0.01,
            x1=0.02
        )
        self.neutral_axis = optim_results.root


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