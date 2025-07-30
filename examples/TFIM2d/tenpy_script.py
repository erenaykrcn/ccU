import tenpy

BD = 100

model_params = dict(
    bc_MPS='infinite', lattice='Square', Lx=4, Ly=4,
    Jz=1, hx=1,
    #conserve='best', # conserve parity
)
model = tenpy.SpinModel(model_params)

du_row = [['down'], ['up'], ['down'], ['up']]
ud_row = [['up'], ['down'], ['up'], ['down']]
psi_Neel = tenpy.MPS.from_lat_product_state(model.lat, [du_row, ud_row, du_row, ud_row])
dmrg_params = dict( mixer=True, trunc_params=dict(chi_max=100, svd_min=1e-10, ), )
engine = tenpy.TwoSiteDMRGEngine( psi_Neel , model , dmrg_params )
energy , psi = engine.run()


with open(f"tenpy_log.txt", "a") as file:
    file.write(f"\n Energy BD={BD} \n")


import numpy as np


du_row = [['down'], ['up'], ['down'], ['up']]
ud_row = [['up'], ['down'], ['up'], ['down']]
psi_Neel = tenpy.MPS.from_lat_product_state(model.lat, [du_row, ud_row, du_row, ud_row])
with open(f"tenpy_log.txt", "a") as file:
    file.write(f"\n Overlap with Neel 1: {np.abs(psi.overlap(psi_Neel))} \n")


du_row = [['down'], ['up'], ['down'], ['up']]
ud_row = [['up'], ['down'], ['up'], ['down']]
psi_Neel = tenpy.MPS.from_lat_product_state(model.lat, [ud_row, du_row, ud_row, du_row])
with open(f"tenpy_log.txt", "a") as file:
    file.write(f"\n Overlap with Neel 1: {np.abs(psi.overlap(psi_Neel))} \n")