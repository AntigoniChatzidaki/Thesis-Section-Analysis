from Materials import Material, Steel, Concrete
from Sections import Rectangular

# 600x600 -Nmax
frp = Material('FRP', 600e6, 80e9, 'black')
conf_steel = Steel('Confining Steel', 190e9, 355e6, 'blue')
reinf_steel = Steel('Reinforcing Steel', 190e9, 500e6, 'black')
concrete = Concrete('Concrete', 37e9, 50e6, 'gray',reinf_steel,None, 0.707878495, 2.5) # TODO

test_section = Rectangular(concrete, 0.3, 0.3)
test_section.add_reinforcements(reinf_steel, 0.032, [(0.058,0.058) , (0.242,0.058) , (0.058,0.242) , (0.242,0.242)])

