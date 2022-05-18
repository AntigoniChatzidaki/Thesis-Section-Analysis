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
    design_strength: float = None
    n: float = None

    def __init__(self, name, unconfined_youngs_modulus, characteristic_strength, color, reinf_steel=None, confining_steel=None,
                 moment_sls_uls: float = 0, phi_t: float = 3):
        self.unconfined_youngs_modulus = unconfined_youngs_modulus
        if confining_steel is None:
            youngs_modulus = unconfined_youngs_modulus
            a_cc = 0.85
        else:
            youngs_modulus = unconfined_youngs_modulus * 1 / (1 + 0.5 * moment_sls_uls * phi_t)
            a_cc = 1

        super().__init__(name, youngs_modulus, characteristic_strength, color)
        self.conf_steel = confining_steel
        self.reinf_steel = reinf_steel
        # Values from eurocode
        gamma_c = 1.5

        design_strength = a_cc * self.characteristic_strength / gamma_c
        if design_strength / 1e6 < 50:
            n = 2
            e_cu2 = 0.0035
            e_c2 = 0.002
            e_cu3 = 0.0035
            e_c3 = 0.00175
        else:
            n = 1.4 + 23.4 * ((90 - characteristic_strength / 1e6) / 100) ** 4
            e_cu2 = (2.6 + 35 * ((90 - characteristic_strength / 1e6) / 100) ** 4) / 1000
            e_c2 = (2.0 + 0.085 * (characteristic_strength / 1e6 - 50) ** 0.53) / 1000
            e_cu3 = e_cu2
            e_c3 = (1.75 + 0.55*((characteristic_strength/1e6 -50)/40))

        self.ultimate_strain = e_cu2
        self.ultimate_strain_3 = e_cu3
        self.critical_strain = e_c2
        self.design_strength = design_strength
        self.n = n
        self.total_height = self.concrete_height
        self.phi_t = phi_t
        self.moment_sls_uls = moment_sls_uls

    # def ultimate_strain(self):
    #     return self.ultimate_strain
