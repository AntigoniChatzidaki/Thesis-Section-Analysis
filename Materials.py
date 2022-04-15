from dataclasses import dataclass

@dataclass
class Material:
    name: str
    youngs_modulus: float
    characteristic_strength: float
    color: str

    @property
    def ultimate_strain(self):
        return self.characteristic_strength / self.youngs_modulus

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Steel(Material):
      pass


class Concrete(Material):
    conf_steel: Steel
    reinf_steel: Steel
    concrete_width: float = None
    concrete_height: float = None
    total_height: float = None
    ultimate_strain: float = None
    critical_strain: float = None
    top_strain: float = None
    confined_youngs_modulus: float = None
    design_strength: float = None
    n: float = None

    def __init__(self, name, youngs_modulus, strength, color, confining_steel=None, reinf_steel=None):
        super().__init__(name, youngs_modulus, strength, color)
        self.conf_steel = confining_steel
        self.reinf_steel = reinf_steel
        # Values from eurocode
        a_cc = 0.85
        gamma_c = 1.5
        design_strength = a_cc * self.characteristic_strength / gamma_c
        if design_strength / 1e6 < 50:
            n = 2
            e_cu2 = 0.0035
            e_c2 = 0.002
        else:
            n = 1.4 + 23.4 * ((90 - design_strength / 1e6) / 100) ** 4
            e_cu2 = (2.6 + 35 * ((90 - design_strength / 1e6) / 100) ** 4) / 1000
            e_c2 = (2.0 + 0.085 * (design_strength / 1e6 - 50) ** 0.53) / 1000
        if self.conf_steel is None:
            total_height = self.concrete_height
            self.ultimate_strain = e_cu2
            self.critical_strain = e_c2
            self.total_height = total_height
            self.design_strength = design_strength
            self.n = n
        else:
            total_height = self.concrete_height
            self.ultimate_strain = e_cu2
            self.critical_strain = e_c2
            self.total_height = total_height
            self.design_strength = design_strength
            self.n = n
        self.total_height = total_height

    # def ultimate_strain(self):
    #     return self.ultimate_strain
