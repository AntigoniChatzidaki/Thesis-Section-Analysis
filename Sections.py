from typing import List, Set, Tuple, Dict, Union

import numpy as np
import pandas as pd
from scipy import optimize
import shapely.geometry
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon, MultiPolygon, LinearRing
from descartes import PolygonPatch
import matplotlib.pyplot as plt

from Materials import Material, Concrete, Steel


def plot_polygon(
        polygon: Union[Polygon, MultiPolygon], ax: plt.Axes, color: str,
        offset: Tuple[float] = (0.0, 0.0), scale: Tuple[float] = (1.0, 1.0)
):
    if type(polygon) is shapely.geometry.MultiPolygon:
        for p in polygon.geoms:
            plot_polygon(p, ax, color, offset, scale)
        return
    # ext_x, ext_y = [list(x) for x in polygon.exterior.xy]
    # int_x = []
    # int_y = []
    # for ring in polygon.interiors:
    #     x, y = ring.xy
    #     int_x += list(x)
    #     int_y += list(y)
    # ax.fill(ext_x + int_x, ext_y + int_y, color=color)
    trans_polygon = affinity.scale(polygon, scale[0], scale[1])
    trans_polygon = affinity.translate(trans_polygon, offset[0], offset[1])
    patch = PolygonPatch(trans_polygon, facecolor=color, linewidth=0)
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

    def plot(self, ax, offset: Tuple[float] = (0.0, 0.0), scale: Tuple[float] = (1.0, 1.0)):
        for material, polygon in self.polygons.items():
            plot_polygon(polygon.buffer(0), ax, material.color, offset, scale)

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
        print(min_y, max_y)

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

    @property
    def characteristic_force_plastic(self):
        characteristic_force_plastic = self.slices.concrete_area.sum() * self.concrete.characteristic_strength
        if self.conf_steel is not None:
            characteristic_force_plastic += self.slices.conf_steel_area.sum() * self.conf_steel.characteristic_strength
        if self.reinf_steel is not None:
            characteristic_force_plastic += self.slices.reinf_steel_area.sum() * self.reinf_steel.characteristic_strength
        return characteristic_force_plastic

    @property
    def EI_eff_confined(self):
        second_moments_of_area = self.second_moments_of_area
        EI_eff = self.concrete.youngs_modulus * second_moments_of_area[self.concrete]['Ixx'] * 0.6
        if self.conf_steel is not None:
            EI_eff += self.conf_steel.youngs_modulus * second_moments_of_area[self.conf_steel]['Ixx']
        if self.reinf_steel is not None:
            EI_eff += self.reinf_steel.youngs_modulus * second_moments_of_area[self.reinf_steel]['Ixx']
        return EI_eff


    def confinement_and_second_order_effect(self, length: float, axial_force: float, max_moment_m_2: float, moment_m_1: float):
        min_y, max_y = self.get_y_limits()
        height = max_y - min_y
        second_moments_of_area = self.second_moments_of_area
        self.axial_force = axial_force
        reinf_steel_to_concrete_ratio = np.nan_to_num((self.slices.reinf_steel_area.sum()) /self.slices.concrete_area.sum())
        if self.conf_steel is not None: # i.e. the section is confined
            calibration_coef_K_o = 0.9
            correction_coef_K_e_II = 0.5
            EI_eff_II_second_order = calibration_coef_K_o * correction_coef_K_e_II * self.concrete.youngs_modulus * \
                                     second_moments_of_area[self.concrete]['Ixx']
            if self.conf_steel is not None:
                EI_eff_II_second_order += calibration_coef_K_o * self.conf_steel.youngs_modulus * \
                                          second_moments_of_area[self.conf_steel]['Ixx']
            if self.reinf_steel is not None:
                EI_eff_II_second_order += calibration_coef_K_o * self.reinf_steel.youngs_modulus * \
                                          second_moments_of_area[self.reinf_steel]['Ixx']
            if axial_force > 0:
                eccentricity_of_loading = (max_moment_m_2 / axial_force)
            else:
                eccentricity_of_loading = 0
            self.eccentricity_depth = eccentricity_of_loading / height
            critical_force = self.EI_eff_confined * np.pi**2 / length**2
            eff_critical_force = EI_eff_II_second_order * np.pi**2 / length**2
            self.elastic_buckling_load_to_applied_ratio = eff_critical_force / axial_force
            self.relative_slenderness = np.sqrt(self.characteristic_force_plastic / critical_force)

            if self.eccentricity_depth <= 0.1:
                confined_factor_h_a_0 = min( 0.25 * (3 + 2 * self.relative_slenderness), 1)
                confined_factor_h_c_0 = max( 4.9 - 18.5 * self.relative_slenderness + 17*self.relative_slenderness**2,0)
                self.confined_factor_h_a = confined_factor_h_a_0 + (1+ confined_factor_h_a_0) * 10 * self.eccentricity_depth
                self.confined_factor_h_c = confined_factor_h_c_0 * (1 - 10 * self.eccentricity_depth)
            else:
                self.confined_factor_h_a = 1
                self.confined_factor_h_c = 0
            if axial_force <=0:
                beta_conf = max(0.66 + 0.44 * max_moment_m_2 / moment_m_1, 0.44)
            else:
                beta_conf = 1
            coefficient_K = max(beta_conf/(1 + axial_force/ eff_critical_force), 1)
            if reinf_steel_to_concrete_ratio < 0.3:
                equivalent_member_imperfection_edo = length / 300
            else:
                equivalent_member_imperfection_edo = length / 200
            self.second_order_moment = abs(coefficient_K * axial_force * equivalent_member_imperfection_edo) # imperfection
            self.first_order_moment_modified = abs(coefficient_K * max_moment_m_2)
            self.second_order_effect_design_moment_M_ed = (self.second_order_moment
                                                               + self.first_order_moment_modified)
        else:
            radius_of_gyration_concrete = np.sqrt(second_moments_of_area[self.concrete]['Ixx'] / self.slices.concrete_area.sum())
            radius_of_gyration_reinf_steel = np.sqrt(second_moments_of_area[self.reinf_steel]['Ixx']/ self.slices.reinf_steel_area.sum())
            self.slenderness_unconf = length / radius_of_gyration_concrete
            beta_unconf = 0.35 + self.concrete.characteristic_strength/200e6 - self.slenderness_unconf/150
            eff_creep_ratio_phi_eff = self.concrete.moment_sls_uls * self.concrete.phi_t
            factor_accounting_for_creep_K_phi = max(1 + beta_unconf * eff_creep_ratio_phi_eff, 1)
            relative_axial_force_n = axial_force / (self.slices.concrete_area.sum() * self.concrete.design_strength)
            omega = ((self.slices.reinf_steel_area.sum() * self.reinf_steel.characteristic_strength/1.15) /
                    (self.slices.concrete_area.sum() * self.concrete.design_strength))
            self.slenderness_limit_unconf = (20 * (1/ (1+0.2*eff_creep_ratio_phi_eff))*(np.sqrt(1+2*omega))*(1.7-moment_m_1/max_moment_m_2))/np.sqrt(relative_axial_force_n)
            n_u = 1 + omega
            n_bal = 0.4 # EC2 5.8.8.3 (3)
            effect_of_creep_factor_K_r = min ((n_u - relative_axial_force_n)/(n_u - n_bal), 1)
            e_yd = self.reinf_steel.characteristic_strength/1.15 / self.reinf_steel.youngs_modulus
            curvature_1_ro = e_yd /(0.45 *(height/2 + radius_of_gyration_reinf_steel))
            curvature_1_r = effect_of_creep_factor_K_r * factor_accounting_for_creep_K_phi * curvature_1_ro
            c_factor = 10  # EC2 - 5.8.8.2.(4)
            self.deflection_e_2 = curvature_1_r * length **2 / c_factor
            self.second_order_moment = abs(axial_force * self.deflection_e_2)
            self.first_order_moment_modified = abs(max(0.4*max_moment_m_2, 0.6*max_moment_m_2 + 0.4*moment_m_1))
            self.second_order_effect_design_moment_M_ed = self.first_order_moment_modified + self.second_order_moment
            if axial_force / self.characteristic_force_plastic < 0.1:
                self.factor_a = 1
            elif  axial_force / self.characteristic_force_plastic < 0.7:
                self.factor_a = (1+(1.5-1)/(0.7-0.1)*((axial_force / self.characteristic_force_plastic)-0.1))
            elif axial_force / self.characteristic_force_plastic < 1.0:
                self.factor_a = (1.5+(2-1.5)/(1-0.7)*((axial_force / self.characteristic_force_plastic)-0.7))
            else:
                self.factor_a = 2
            print(axial_force / self.characteristic_force_plastic)
            # effect of creep


    def calculate_forces(self, axial_force: float, outer_thickness: float = 0.0):
        min_y, max_y = self.get_y_limits()
        mid_height = (max_y + min_y) / 2
        height = max_y - min_y

        if self.slices is None:
            raise RuntimeError("Need to generate slices first")
        if self.neutral_axis is None:
            raise RuntimeError("Need to define a neutral axis")

        def strain_fun(slice):
            return self.concrete.ultimate_strain / self.neutral_axis * (self.neutral_axis - slice.mid_height)

        def reinf_steel_stress_fun(slice):
            if slice.reinf_steel_area > 0:
                if abs(slice.strain) < self.reinf_steel.characteristic_strength / 1.15 / self.reinf_steel.youngs_modulus:
                    reinf_steel_stress = slice.strain * self.reinf_steel.youngs_modulus
                else:
                    reinf_steel_stress = self.reinf_steel.characteristic_strength / 1.15 * abs(slice.strain) / slice.strain
            else:
                reinf_steel_stress = 0
            return reinf_steel_stress

        def conf_steel_stress_fun(slice):
            if slice.conf_steel_area > 0:
                if abs(slice.strain) < self.conf_steel.characteristic_strength / self.conf_steel.youngs_modulus:
                    conf_steel_stress = slice.strain * self.conf_steel.youngs_modulus
                else:
                    conf_steel_stress = self.conf_steel.characteristic_strength * abs(slice.strain) / slice.strain
            else:
                conf_steel_stress = 0
            return conf_steel_stress

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
        else:
            self.slices['reinf_steel_stress'] = 0
        if self.conf_steel is not None:
            self.slices['conf_steel_stress'] = self.slices.apply(conf_steel_stress_fun, axis=1)
        else:
            self.slices['conf_steel_stress'] = 0

        if self.conf_steel is None:
            if self.concrete is not None:
                self.slices['concrete_force'] = np.nan_to_num(self.slices.concrete_stress * self.slices.concrete_area)
            if self.reinf_steel is not None:
                self.slices['reinf_steel_force'] = np.nan_to_num(self.slices.reinf_steel_stress * self.slices.reinf_steel_area)
            self.slices['conf_steel_force'] = 0

        else:
            # if axial_force > 0:
            #     eccentricity_of_loading = (max_moment / axial_force)
            # else:
            #     eccentricity_of_loading = 0
            # self.eccentricity_depth = eccentricity_of_loading / height
            # critical_force = self.EI_eff_confined * np.pi**2 / length**2
            # eff_critical_force = self.EI_eff_II_second_order * np.pi**2 / length**2
            # self.elastic_buckling_load_to_applied_ratio = eff_critical_force / axial_force
            # self.relative_slenderness = np.sqrt(self.characteristic_force_plastic / critical_force)
            #
            # if self.eccentricity_depth <= 0.1:
            #     confined_factor_h_a_0 = min( 0.25 * (3 + 2 * self.relative_slenderness) , 1)
            #     confined_factor_h_c_0 = max( 4.9 - 18.5 * self.relative_slenderness + 17*self.relative_slenderness**2,0)
            #     self.confined_factor_h_a = confined_factor_h_a_0 + (1+ confined_factor_h_a_0) * 10 * self.eccentricity_depth
            #     self.confined_factor_h_c = confined_factor_h_c_0 * (1 - 10 * self.eccentricity_depth)
            # else:
            #     self.confined_factor_h_a = 1
            #     self.confined_factor_h_c = 0

            self.slices['concrete_force'] = np.nan_to_num(
                self.slices.concrete_stress * self.slices.concrete_area *
                (1 + self.confined_factor_h_c * outer_thickness / height *
                 self.slices.conf_steel_stress / self.slices.concrete_stress)
            )

            if self.reinf_steel is not None:
                self.slices['reinf_steel_force'] = np.nan_to_num(
                    self.slices.reinf_steel_stress * self.slices.reinf_steel_area
                )

            else:
                self.slices['reinf_steel_force'] = 0
            self.slices['conf_steel_force'] = np.nan_to_num(
                (self.slices.conf_steel_stress * self.slices.conf_steel_area * self.confined_factor_h_a)
            )
            self.max_characteristic_force = (self.slices.concrete_area.sum() * self.concrete.characteristic_strength *
                                                 (1 + self.confined_factor_h_c * outer_thickness / height *
                                                  self.conf_steel.characteristic_strength / self.concrete.characteristic_strength))
            if self.conf_steel is not None:
                self.max_characteristic_force += self.slices.conf_steel_area.sum() * self.conf_steel.characteristic_strength * self.confined_factor_h_a
            if self.reinf_steel is not None:
                self.max_characteristic_force += self.slices.reinf_steel_area.sum() * self.reinf_steel.characteristic_strength

        self.slices['total_force'] = self.slices.concrete_force + self.slices.reinf_steel_force + self.slices.conf_steel_force
        self.slices['total_moment'] = self.slices.total_force * (mid_height - self.slices.mid_height)
        # self.slices['steel_moment'] = self.slices.steel_force * (mid_height - self.slices.mid_height)

        total_force = self.slices.total_force.sum()
        sum_force = total_force - axial_force



        # return (total_force, sum_force, characteristic_force_plastic, relative_slenderness, eccentricity_depth,
        #        confined_factor_h_a, confined_factor_h_c)
        self.height = height
        self.outer_thickness = outer_thickness
        return total_force, sum_force

    def calculate_neutral_axis(self, axial_force: float, outer_thickness: float = 0.0):
        max_y = self.get_y_limits()

        # Returns total force for an assumed neutral axis
        def optim_fun(guess: float) -> float:
            self.neutral_axis = guess
            total_force, sum_force = self.calculate_forces(axial_force, outer_thickness)
            return sum_force

        optim_results = optimize.root_scalar(
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
    def plastic_load(self) -> float:
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
