from dataclasses import dataclass

@dataclass
class Material:
    name: str
    youngs_modulus: float
    strength: float
    color: str

    @property
    def ultimate_strain(self):
        return self.strength / self.youngs_modulus

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Concrete(Material):
    confined: bool
    frp: Material
    frp_thickness: float = None
    concrete_width: float = None
    concrete_height: float = None
    total_height: float = None
    ultimate_strain: float = None
    critical_strain: float = None
    top_strain: float = None
    confined_youngs_modulus: float = None
    design_strength: float = None
    n: float = None

    def __init__(self, name, youngs_modulus, strength, color, confined, frp):
        super().__init__(name, youngs_modulus, strength, color)
        self.confined = confined
        self.frp = frp

        if not self.confined:
            # Values from eurocode
            a_cc = 1.0
            gamma_c = 1.5
            design_strength = a_cc * self.strength / gamma_c
            if design_strength/10e6 < 50:
                n = 2
                e_cu2 = 0.0020
                e_c2 = 0.0035
            else:
                n = 1.4 + 23.4 * ((90 - design_strength/10e6) / 100) ** 4
                e_cu2 = (2.6 + 35 * ((90 - design_strength/10e6) / 100) ** 4) / 1000
                e_c2 = (2.0 + 0.085 * (design_strength/10e6 - 50) ** 0.53) / 1000
            total_height = self.concrete_height
            self.ultimate_strain = e_cu2
            self.critical_strain = e_c2
            self.total_height = total_height
            self.design_strength = design_strength
            self.n = n
        else:
            # values from paper
            beta_j = max((self.frp.youngs_modulus * self.frp_thickness) / (self.strength * self.concrete_width / 2),
                         6.5)
            conf_concrete_strength = self.strength + 3.5 * self.frp.youngs_modulus / (self.concrete_width / 2) * (
                        1 - 6.5 / beta_j) * self.frp.ultimate_strain
            conf_concrete_unconfined_strain = 0.0030 + 0.6 * beta_j ** 0.8 * self.frp.ultimate_strain ** 1.45
            conf_concrete_youngs_modulus = (conf_concrete_strength - self.strength) / conf_concrete_unconfined_strain
            conf_concrete_top_strain = 2 * self.strength / (self.youngs_modulus - conf_concrete_youngs_modulus)
            total_height = self.concrete_height + 2 * self.frp_thickness
            # e_ccu = conf_concrete_unconfined_strain
            # e_t = conf_concrete_top_strain
            self.ultimate_strain = conf_concrete_unconfined_strain
            self.top_strain = conf_concrete_top_strain
            self.confined_youngs_modulus = conf_concrete_youngs_modulus
        self.total_height = total_height

    # def ultimate_strain(self):
    #     return self.ultimate_strain
