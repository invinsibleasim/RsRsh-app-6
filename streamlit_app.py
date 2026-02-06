# make_sample_light_iv_sets.py
import numpy as np
import pandas as pd

q = 1.602176634e-19
kB = 1.380649e-23
T  = 298.15
Ns = 60

# baseline params (tweak to taste)
I0  = 1e-9      # A
n   = 1.6
Rs  = 0.15      # Ω
Rsh = 800.0     # Ω
Voc = 232.5     # V at 1000 W/m2
Isc = 3.02      # A at 1000 W/m2

def IL_from_Voc_Isc(Voc, Isc, I0, n, Rs, Rsh, T, Ns):
    Ns_eff = max(int(Ns), 1)
    Voc_cell = Voc / Ns_eff
    IL_voc = I0*(np.exp(q*Voc_cell/(n*kB*T)) - 1.0) + Voc/Rsh
    Vd_sc_cell = (-Isc*Rs)/Ns_eff
    IL_isc = Isc + I0*(np.exp(q*Vd_sc_cell/(n*kB*T)) - 1.0) + (-Isc*Rs)/Rsh
    return 0.5*(IL_voc + IL_isc)

def solve_light(V, IL, I0, n, Rs, Rsh, T, Ns, iters=80):
    V = np.asarray(V, float)
    I = np.zeros_like(V)
    Ns_eff = max(int(Ns), 1)
    for _ in range(iters):
        Vd = V - I*Rs
        Vc = Vd/Ns_eff
        expo = np.clip(q*Vc/(n*kB*T), -100, 100)
        Id = I0*(np.exp(expo)-1.0)
        Ish = Vd/Rsh
        Inew = IL - Id - Ish
        if np.max(np.abs(Inew - I)) < 1e-9:
            return Inew
        I = 0.5*I + 0.5*Inew
    return I

def write_csv(G, scale):
    # scale IL with irradiance linearly; Voc small shift
    IL = IL_from_Voc_Isc(Voc, Isc*scale, I0, n, Rs, Rsh, T, Ns)
    V = np.linspace(0, Voc*(0.98 + 0.02*scale), 350)
    I = solve_light(V, IL, I0, n, Rs, Rsh, T, Ns)
    pd.DataFrame({"V_light_V": V, "I_light_A": I}).to_csv(f"lightIV_{int(G)}Wm2.csv", index=False)

irr = [1000, 800, 600, 400, 200]
for G in irr:
    write_csv(G, G/1000.0)
print("Wrote:", [f"lightIV_{g}Wm2.csv" for g in irr])
