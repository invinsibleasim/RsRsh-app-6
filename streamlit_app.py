# streamlit_app.py
# EL image → Dark & Light I–V (module) with robust per-cell diode solve
# Adds multi-irradiance Light I–V fitting (global Rs, Rsh from ≥3 measured curves),
# and can synthesize a Light I–V from EL→dark params + (Voc, Isc).

import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="EL → Dark & Light I–V (Module)", layout="wide")
st.title("EL image → Dark & Light I–V (Module)")

st.markdown("""
Upload **EL images** at **multiple forward-bias setpoints** and a **metadata CSV**:

- CSV columns (header required): `filename, V_applied_V, I_meas_A, exposure_s, gain_iso, temp_C`
- Optional: dark frame image and flat-field image for camera corrections

**Light I–V:**
- Upload **measured Light I–V CSVs** (columns: `V_light_V, I_light_A`) for different **irradiances** (e.g., 1000/800/600/400/200 W/m²), or
- Let the app **synthesize** a Light I–V from the EL‑derived dark parameters plus your **Voc** and **Isc**.

**Series resistance requirement:**  
- **≥ 3 measured light I–V curves** (with irradiances) to compute a robust **multi‑curve Rs/Rsh**; otherwise the app shows **single‑curve** small‑signal estimates.
""")

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Preprocessing")
    border_crop_px = st.number_input("Crop border (pixels)", min_value=0, max_value=500, value=10, step=2)
    trim_low = st.slider("Trim lowest % pixels", 0, 40, 5, step=1)
    trim_high = st.slider("Trim highest % pixels", 0, 40, 5, step=1)
    eps = st.number_input("Min intensity clamp (epsilon)", 1e-12, 1e-3, 1e-6, format="%.1e")
    roi_box = st.text_input("ROI (x0,y0,w,h) optional", "")

    st.header("Fitting (EL→Dark)")
    # Physical constants
    q = 1.602176634e-19
    kB = 1.380649e-23
    # Ns: number of cells in series (IMPORTANT for per-cell voltage)
    Ns = st.number_input("Series cell count (Ns)", min_value=1, max_value=200, value=60, step=1)
    ref_T = st.number_input("Assume module T for EL slope (°C)", -20.0, 120.0, 25.0)
    n_reg = st.number_input("n regularization weight (0=off)", 0.0, 10.0, 1.0, step=0.5)
    max_iter = st.number_input("Max iterations (implicit solve)", 10, 500, 60, step=10)

    st.header("Dark I–V sweep (display)")
    v_min = st.number_input("Vmin for dark sweep (V)", 0.0, 2000.0, 0.0)
    v_max = st.number_input("Vmax for dark sweep (V)", 0.1, 2000.0, 40.0)
    v_pts = st.number_input("Points", 10, 5000, 400)

    st.header("Light I–V (measurement or synth)")
    # Measured IV (for synthesizing IL and comparison)
    Voc_meas = st.number_input("Measured Voc (V)", value=232.5431, step=0.001)
    Isc_meas = st.number_input("Measured Isc (A)", value=3.02288, step=0.001)

    # Multiple measured Light I–V curves (accept many files)
    light_iv_files = st.file_uploader(
        "Measured Light I–V CSVs (V_light_V, I_light_A) — upload 1..5 files",
        type=["csv"], accept_multiple_files=True
    )

# -----------------------
# File inputs for EL
# -----------------------
csv_file = st.file_uploader("Metadata CSV (for EL)", type=["csv"])
imgs = st.file_uploader("EL images (multi-select)", type=["png","tif","tiff","jpg","jpeg"], accept_multiple_files=True)
dark_frame_file = st.file_uploader("Optional: dark frame", type=["png","tif","tiff","jpg","jpeg"])
flat_field_file = st.file_uploader("Optional: flat-field", type=["png","tif","tiff","jpg","jpeg"])

if not csv_file or not imgs:
    st.info("Upload metadata CSV and at least 3 EL images to begin.")
    st.stop()

# -----------------------
# Load metadata
# -----------------------
meta = pd.read_csv(csv_file)
required_cols = ["filename", "V_applied_V", "I_meas_A", "exposure_s", "gain_iso", "temp_C"]
missing = [c for c in required_cols if c not in meta.columns]
if missing:
    st.error(f"CSV missing columns: {missing}")
    st.stop()

# Index uploaded images by name
img_by_name = {f.name: f for f in imgs}

# -----------------------
# Load helper images
# -----------------------
def load_pil(file):
    try:
        if file is None:
            return None
        return Image.open(file)
    except Exception:
        return None

dark_frame = load_pil(dark_frame_file)
flat_field = load_pil(flat_field_file)

# -----------------------
# Image helpers
# -----------------------
def pil_to_float(im: Image.Image) -> np.ndarray:
    """Convert PIL image to float64, normalized to ~[0, 1]."""
    if im is None:
        return None
    if im.mode in ("I;16", "I"):
        arr = np.array(im, dtype=np.float64)
        # If 16-bit like I;16: use 65535 for scaling; else just max() to be safe
        scale = 65535.0 if arr.max() > 255 else float(arr.max() or 1.0)
        arr = arr / (scale if scale > 0 else 1.0)
    else:
        arr = np.array(im.convert("L"), dtype=np.float64)
        scale = float(arr.max() or 1.0)
        arr = arr / (scale if scale > 0 else 1.0)
    return arr

def apply_corrections(arr: np.ndarray, dark: Image.Image = None, flat: Image.Image = None) -> np.ndarray:
    a = arr.copy()
    if dark is not None:
        d = pil_to_float(dark)
        if d is not None:
            if d.shape != a.shape:
                d = np.resize(d, a.shape)
            a = np.clip(a - d, 0, None)
    if flat is not None:
        f = pil_to_float(flat)
        if f is not None:
            if f.shape != a.shape:
                f = np.resize(f, a.shape)
            f = np.where(f <= 0, 1.0, f)
            a = a / f
    return a

def crop_to_roi(arr: np.ndarray, border_px=0, roi_spec=""):
    h, w = arr.shape
    x0, y0, ww, hh = 0, 0, w, h
    if roi_spec.strip():
        try:
            x0, y0, ww, hh = [int(v) for v in roi_spec.split(",")]
            x0 = np.clip(x0, 0, w-1); y0 = np.clip(y0, 0, h-1)
            ww = np.clip(ww, 1, w-x0); hh = np.clip(hh, 1, h-y0)
        except Exception:
            pass
    arr2 = arr[y0:y0+hh, x0:x0+ww]
    if border_px > 0 and arr2.shape[0] > 2*border_px and arr2.shape[1] > 2*border_px:
        arr2 = arr2[border_px:-border_px, border_px:-border_px]
    return arr2

# -----------------------
# Compute robust metric per image
# -----------------------
rows = []
for _, r in meta.iterrows():
    name = str(r["filename"])
    if name not in img_by_name:
        st.error(f"Image '{name}' not uploaded.")
        st.stop()
    im = Image.open(img_by_name[name])
    arr = pil_to_float(im)
    arr = apply_corrections(arr, dark_frame, flat_field)
    arr = crop_to_roi(arr, border_px=border_crop_px, roi_spec=roi_box)
    arr = np.clip(arr, eps, None)

    # robust trimmed log-mean
    logA = np.log(arr)
    flat = np.sort(logA.flatten())
    n = len(flat)
    lo = int(n * (trim_low/100.0))
    hi = int(n * (1.0 - trim_high/100.0))
    lo = np.clip(lo, 0, n-1); hi = np.clip(hi, lo+1, n)
    trimmed = flat[lo:hi]
    log_mean = float(np.mean(trimmed))
    rows.append({
        "filename": name,
        "log_mean": log_mean,
        "V": float(r["V_applied_V"]),
        "I": float(r["I_meas_A"]),
        "exposure_s": float(r["exposure_s"]),
        "gain_iso": float(r["gain_iso"]),
        "temp_C": float(r["temp_C"]),
    })

df = pd.DataFrame(rows).sort_values("V").reset_index(drop=True)

# -----------------------
# EL slope fit → ideality n (using per-cell slope)
# -----------------------
T_K = (df["temp_C"].mean() if df["temp_C"].notna().all() else ref_T) + 273.15
A = np.vstack([np.ones(len(df)), df["V"].values]).T
y = df["log_mean"].values
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
b = coef[1]  # slope vs module voltage
# n ≈ q * Ns / (b * kT)
n_EL = float(q * Ns / (b * kB * T_K)) if b > 0 else 2.0

col1, col2 = st.columns(2)
with col1:
    st.write("**EL fit (log-intensity vs Vmodule):**")
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(df["V"], df["log_mean"], color="tab:blue", label="data")
    ax.plot(df["V"], coef[0]+coef[1]*df["V"], color="tab:orange", label=f"fit: n ≈ {n_EL:.2f}")
    ax.set_xlabel("Applied V (module, V)"); ax.set_ylabel("log mean intensity")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)
with col2:
    st.metric("Estimated ideality n (EL-derived)", f"{n_EL:.2f}")

# ======================================================
# Single-diode (dark) model fit
# I = I0*(exp(q*Vd_cell/(n*kT)) - 1) + Vd/Rsh
# with Vd = Vmodule - I*Rs,  Vd_cell = Vd/Ns
# ======================================================
V_meas = df["V"].values.astype(float)
I_meas = df["I"].values.astype(float)

def solve_current_dark(V_module, I0, n, Rs, Rsh, T=T_K, Ns=Ns, iters=60):
    """Fixed-point solve for I at module voltage (dark), using per-cell junction voltage."""
    V_module = np.asarray(V_module, dtype=float)
    I = np.zeros_like(V_module)
    Ns_eff = max(int(Ns), 1)
    for _ in range(int(iters)):
        Vd = V_module - I*Rs
        Vd_cell = Vd / Ns_eff
        expo_arg = np.clip(q * Vd_cell / (n * kB * T), -100.0, 100.0)
        Id = I0 * (np.exp(expo_arg) - 1.0)
        Ish = Vd / Rsh if Rsh > 0 else 0.0
        I_new = Id + Ish
        if not np.all(np.isfinite(I_new)):
            I_new = np.nan_to_num(I_new, nan=0.0, posinf=1e30, neginf=-1e30)
        if np.max(np.abs(I_new - I)) < 1e-9:
            I = I_new
            break
        I = 0.5*I + 0.5*I_new
    return I

def loss_dark(params):
    I0, n, Rs, Rsh = params
    if I0 <= 0 or n <= 0 or Rs < 0 or Rsh <= 0:
        return 1e99
    I_pred = solve_current_dark(V_meas, I0, n, Rs, Rsh, T=T_K, Ns=Ns, iters=max_iter)
    if not np.all(np.isfinite(I_pred)):
        return 1e99
    reg = n_reg * (n - n_EL)**2
    return float(np.mean((I_pred - I_meas)**2) + reg)

# Coarse + refine (dependency-light)
rng = np.random.default_rng(42)
def safe_seed():
    return [1e-9, max(1.0, min(2.0, float(n_EL))), 0.1, 1e4]  # [I0, n, Rs, Rsh]

best = safe_seed()
best_loss = loss_dark(best)
if not np.isfinite(best_loss):
    best_loss = 1e99

I0_grid  = np.logspace(-12, -7, 6)                             # A
n_grid   = np.linspace(max(1.0, n_EL-0.4), min(2.5, n_EL+0.4), 6)
Rs_grid  = np.linspace(0.0, 1.0, 6)                             # Ω (module)
Rsh_grid = np.logspace(2, 5, 6)                                 # Ω

for I0 in I0_grid:
    for n_try in n_grid:
        for Rs in Rs_grid:
            for Rsh in Rsh_grid:
                L = loss_dark((I0, n_try, Rs, Rsh))
                if np.isfinite(L) and L < best_loss:
                    best_loss = L
                    best = [float(I0), float(n_try), float(Rs), float(Rsh)]

for _ in range(200):
    trial = [
        max(1e-14, best[0] * 10**rng.normal(0, 0.2)),
        float(np.clip(best[1] + rng.normal(0, 0.05), 0.9, 3.0)),
        float(np.clip(best[2] + rng.normal(0, 0.05), 0.0, 5.0)),
        float(np.clip(best[3] * 10**rng.normal(0, 0.2), 1.0, 1e7)),
    ]
    L = loss_dark(trial)
    if np.isfinite(L) and L < best_loss:
        best_loss = L
        best = trial

I0_fit, n_fit, Rs_fit, Rsh_fit = best
st.write(
    f"**Dark fit @ {T_K:.1f} K (Ns={Ns}):**  "
    f"I0 = {I0_fit:.3e} A,  n = {n_fit:.2f},  Rs = {Rs_fit:.3f} Ω,  Rsh = {Rsh_fit:.1f} Ω"
)

# Dark I–V display
V_sweep = np.linspace(v_min, v_max, int(v_pts))
I_sweep = solve_current_dark(V_sweep, I0_fit, n_fit, Rs_fit, Rsh_fit, T=T_K, Ns=Ns, iters=max_iter)
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(V_sweep, I_sweep, label="Dark I–V (fit)")
ax2.scatter(V_meas, I_meas, color="tab:red", zorder=5, label="Measured points")
ax2.set_xlabel("Voltage (V, module)"); ax2.set_ylabel("Current (A)")
ax2.grid(True, alpha=0.3); ax2.legend()
st.pyplot(fig2)

# Export dark IV CSV
out = pd.DataFrame({"V_V": V_sweep, "I_A": I_sweep})
st.download_button("Download dark IV (CSV)", out.to_csv(index=False).encode("utf-8"),
                   file_name="dark_IV_from_EL.csv", mime="text/csv")

# ======================================================
# Light I–V (measured or synth) + Rs, Rsh from light IV
# ======================================================
def solve_current_light(V_module, IL, I0, n, Rs, Rsh, T=T_K, Ns=Ns, iters=60):
    """Fixed-point solve for LIGHT current: I = IL - diode - shunt (implicit in I)."""
    V_module = np.asarray(V_module, dtype=float)
    I = np.zeros_like(V_module)
    Ns_eff = max(int(Ns), 1)
    for _ in range(int(iters)):
        Vd = V_module - I*Rs
        Vd_cell = Vd / Ns_eff
        expo_arg = np.clip(q * Vd_cell / (n * kB * T), -100.0, 100.0)
        Id = I0 * (np.exp(expo_arg) - 1.0)
        Ish = Vd / Rsh if Rsh > 0 else 0.0
        I_new = IL - Id - Ish
        if not np.all(np.isfinite(I_new)):
            I_new = np.nan_to_num(I_new, nan=0.0, posinf=1e30, neginf=-1e30)
        if np.max(np.abs(I_new - I)) < 1e-9:
            I = I_new
            break
        I = 0.5*I + 0.5*I_new
    return I

def estimate_IL_from_Voc_Isc(Voc, Isc, I0, n, Rs, Rsh, T=T_K, Ns=Ns):
    """Compute IL from both Voc and Isc constraints, then average."""
    Ns_eff = max(int(Ns), 1)
    Voc_cell = Voc / Ns_eff
    IL_voc = I0 * (np.exp(np.clip(q*Voc_cell/(n*kB*T), -100.0, 100.0)) - 1.0) + Voc / Rsh
    Vd_sc_cell = (-Isc * Rs) / Ns_eff
    IL_isc = Isc + I0 * (np.exp(np.clip(q*Vd_sc_cell/(n*kB*T), -100.0, 100.0)) - 1.0) + (-Isc * Rs) / Rsh
    return 0.5*(IL_voc + IL_isc)

def diff_slope(V, I):
    """Return dV/dI via central differences; invert dI/dV from np.gradient."""
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    dI = np.gradient(I, V, edge_order=2)  # dI/dV
    with np.errstate(divide='ignore', invalid='ignore'):
        dVdI = 1.0 / dI
    return dVdI

def rs_rsh_from_light_iv(V, I):
    """Estimate Rsh near Isc (V≈0) and Rs near Voc (I≈0) from light IV."""
    dVdI = diff_slope(V, I)
    idx_sc = np.argmin(np.abs(V - 0.0))
    idx_voc = np.argmin(np.abs(I - 0.0))
    w = 3
    sc_slice = slice(max(0, idx_sc - w), min(len(V), idx_sc + w + 1))
    oc_slice = slice(max(0, idx_voc - w), min(len(V), idx_voc + w + 1))
    Rsh_est = -np.nanmean(dVdI[sc_slice])
    Rs_est = np.nanmean(dVdI[oc_slice])
    return Rs_est, Rsh_est

# --- Light I–V ingestion: multiple measured curves + irradiances ---
meas_curves = []  # list of dict: {"V":..., "I":..., "G": irradiance, "name":...}
if light_iv_files:
    st.subheader("Measured Light I–V curves (enter irradiance for each)")
    defaults = [1000.0, 800.0, 600.0, 400.0, 200.0]
    for i, f in enumerate(light_iv_files):
        colA, colB = st.columns([2, 1])
        with colA:
            st.write(f"**{f.name}**")
        with colB:
            G = st.number_input(f"Irradiance for {f.name} (W/m²)",
                                min_value=50.0, max_value=1200.0,
                                value=defaults[i] if i < len(defaults) else 1000.0,
                                step=50.0, key=f"G_{i}")
        try:
            dfL = pd.read_csv(f)
            if not {"V_light_V", "I_light_A"}.issubset(dfL.columns):
                st.error(f"{f.name}: CSV must contain columns: V_light_V, I_light_A")
            else:
                V = dfL["V_light_V"].astype(float).to_numpy()
                I = dfL["I_light_A"].astype(float).to_numpy()
                idx = np.argsort(V)
                V, I = V[idx], I[idx]
                meas_curves.append({"V": V, "I": I, "G": float(G), "name": f.name})
        except Exception as e:
            st.error(f"{f.name}: parse error: {e}")

# --- Path 1: Measured curves → single-curve estimates + multi-curve fit if ≥3 ---
st.subheader("Light I–V (measured or synthesized)")
if len(meas_curves) > 0:
    # Plot
    figL, axL = plt.subplots(figsize=(6.5, 4.0))
    colors = ["tab:green", "tab:olive", "tab:purple", "tab:cyan", "tab:gray"]
    for k, c in enumerate(meas_curves):
        col = colors[k % len(colors)]
        axL.plot(c["V"], c["I"], color=col, lw=2, label=f"{c['name']}  ({int(c['G'])} W/m²)")
    axL.set_xlabel("Voltage (V)"); axL.set_ylabel("Current (A)")
    axL.grid(True, alpha=0.3); axL.legend()
    st.pyplot(figL)

    # Per-curve small-signal estimates
    st.write("**Per-curve small-signal (single-curve) estimates:**")
    for c in meas_curves:
        rs_local, rsh_local = rs_rsh_from_light_iv(c["V"], c["I"])
        st.write(f"- {c['name']} @ {int(c['G'])} W/m² → Rs ≈ {rs_local:.3f} Ω,  Rsh ≈ {rsh_local:.1f} Ω")

    # Multi-curve global fit for Rs, Rsh (I0,n fixed) + per-curve IL, if ≥3
    if len(meas_curves) >= 3:
        st.write("**Multi‑curve fit (global Rs,Rsh; per-curve IL; I0,n fixed from dark fit)**")
        curves = meas_curves
        m = len(curves)

        # Initial guesses
        Rs_g = max(0.0, Rs_fit)
        Rsh_g = max(10.0, Rsh_fit)
        IL_list = []
        for c in curves:
            idx0 = np.argmin(np.abs(c["V"] - 0.0))
            IL_list.append(max(0.0, c["I"][idx0]))  # start IL ≈ Isc
        IL_list = np.array(IL_list, dtype=float)

        lam_IL = 0.05   # tether IL to measured Isc
        lam_G  = 0.02   # encourage IL ∝ irradiance

        def loss_light_global(Rs, Rsh, IL_vec):
            if Rs < 0 or Rsh <= 0 or np.any(~np.isfinite(IL_vec)) or not np.isfinite(Rs) or not np.isfinite(Rsh):
                return 1e99
            total = 0.0
            for j, c in enumerate(curves):
                Vj, Ij = c["V"], c["I"]
                I_model = solve_current_light(Vj, IL_vec[j], I0_fit, n_fit, Rs, Rsh, T=T_K, Ns=Ns, iters=max_iter)
                if not np.all(np.isfinite(I_model)):
                    return 1e99
                total += np.mean((I_model - Ij)**2)
            # regularize to Isc and irradiance proportionality
            Isc_meas_arr = np.array([c["I"][np.argmin(np.abs(c["V"] - 0.0))] for c in curves], dtype=float)
            G_arr = np.array([c["G"] for c in curves], dtype=float)
            total += lam_IL * np.mean((IL_vec - Isc_meas_arr)**2)
            ratio = IL_vec / np.clip(G_arr, 1.0, None)
            total += lam_G * np.var(ratio)
            return float(total)

        best_Rs, best_Rsh = Rs_g, Rsh_g
        best_IL = IL_list.copy()
        bestL = loss_light_global(best_Rs, best_Rsh, best_IL)

        rng2 = np.random.default_rng(123)
        for _ in range(300):
            t_Rs  = max(0.0, best_Rs  + rng2.normal(0, 0.02))
            t_Rsh = max(5.0, best_Rsh * 10**rng2.normal(0, 0.05))
            t_IL  = best_IL * (1.0 + rng2.normal(0, 0.02, size=best_IL.shape))
            L = loss_light_global(t_Rs, t_Rsh, t_IL)
            if np.isfinite(L) and L < bestL:
                bestL, best_Rs, best_Rsh, best_IL = L, t_Rs, t_Rsh, t_IL

        st.write(f"**Multi‑curve result:**  Rs ≈ {best_Rs:.3f} Ω,  Rsh ≈ {best_Rsh:.1f} Ω  (global)")

        # Overlay fits
        figM, axM = plt.subplots(figsize=(6.5, 4.0))
        for k, c in enumerate(curves):
            col = colors[k % len(colors)]
            I_model = solve_current_light(c["V"], best_IL[k], I0_fit, n_fit, best_Rs, best_Rsh, T=T_K, Ns=Ns, iters=max_iter)
            axM.plot(c["V"], c["I"], color=col, lw=2, label=f"Meas {int(c['G'])} W/m²")
            axM.plot(c["V"], I_model, color=col, ls="--", lw=2, label=f"Fit  {int(c['G'])} W/m²")
        axM.set_xlabel("Voltage (V)"); axM.set_ylabel("Current (A)")
        axM.grid(True, alpha=0.3); axM.legend(ncol=2, fontsize=9)
        st.pyplot(figM)
    else:
        st.info("Upload **≥ 3** measured light I–V curves with irradiances to compute a robust multi‑curve Rs/Rsh.")

# --- Path 2: Synthesize Light I–V from dark fit + (Voc, Isc) when no measured curves
if len(meas_curves) == 0:
    IL_est = estimate_IL_from_Voc_Isc(Voc=Voc_meas, Isc=Isc_meas,
                                      I0=I0_fit, n=n_fit, Rs=Rs_fit, Rsh=Rsh_fit,
                                      T=T_K, Ns=Ns)
    V_light = np.linspace(0.0, Voc_meas, 400)
    I_light = solve_current_light(V_light, IL_est, I0_fit, n_fit, Rs_fit, Rsh_fit, T=T_K, Ns=Ns, iters=max_iter)

    Rs_light_est, Rsh_light_est = rs_rsh_from_light_iv(V_light, I_light)

    figL2, axL2 = plt.subplots(figsize=(6, 4))
    axL2.plot(V_light, I_light, 'g-', lw=2, label="Synth Light I–V (from EL→dark params)")
    axL2.scatter([0, Voc_meas], [Isc_meas, 0], color="k", zorder=5, label="(0, Isc), (Voc, 0)")
    axL2.set_xlabel("Voltage (V)"); axL2.set_ylabel("Current (A)")
    axL2.grid(True, alpha=0.3); axL2.legend()
    st.pyplot(figL2)

    st.write(f"**From synthesized Light I–V:**  Rs_light ≈ {Rs_light_est:.3f} Ω,  Rsh_light ≈ {Rsh_light_est:.1f} Ω")

    outL = pd.DataFrame({"V_light_V": V_light, "I_light_A": I_light})
    st.download_button("Download synth Light I–V (CSV)", outL.to_csv(index=False).encode("utf-8"),
                       file_name="synth_light_IV.csv", mime="text/csv")

# --- Quick comparison (indicative): use the highest irradiance measured curve, if present
st.subheader("Comparison: Light-derived vs Dark-fit (indicative)")
if len(meas_curves) >= 1:
    rep = sorted(meas_curves, key=lambda c: c["G"], reverse=True)[0]
    Rs_light_rep, Rsh_light_rep = rs_rsh_from_light_iv(rep["V"], rep["I"])
    st.write(
        f"- **Dark fit:** Rs = {Rs_fit:.3f} Ω,  Rsh = {Rsh_fit:.1f} Ω  \n"
        f"- **Light‑derived (rep: {rep['name']} @ {int(rep['G'])} W/m²):** "
        f"Rs ≈ {Rs_light_rep:.3f} Ω,  Rsh ≈ {Rsh_light_rep:.1f} Ω"
    )
elif len(meas_curves) == 0:
    st.write(
        f"- **Dark fit:** Rs = {Rs_fit:.3f} Ω,  Rsh = {Rsh_fit:.1f} Ω  \n"
        f"- **Light‑derived (synth):** see values above."
    )

# ======================================================
# IEC 60891 Rs (multi‑IV) — Procedure 1, multi-curve fit
# ======================================================
st.header("IEC 60891 Rs (multi‑IV)")

st.markdown(
    "Upload a CSV with **multiple measured light I–V curves** and their conditions. "
    "Required columns: `SweepID, V_V, I_A, G_Wm2, T_C`. "
    "The app translates each curve to **target (G2, T2)** with **IEC 60891 Procedure 1** "
    "and estimates a **global Rs** (and optionally κ) that minimizes cross‑curve mismatch."
)

iv60891_file = st.file_uploader("IEC 60891 multi‑IV CSV", type=["csv"], key="iv60891")

col60891a, col60891b, col60891c = st.columns(3)
with col60891a:
    G2 = st.number_input("Target irradiance G₂ (W/m²)", value=1000.0, step=10.0, min_value=100.0)
with col60891b:
    T2 = st.number_input("Target temperature T₂ (°C)", value=25.0, step=0.5)
with col60891c:
    I_grid_pts = st.number_input("Interpolation points (I-grid)", value=200, min_value=50, max_value=2000, step=50)

st.subheader("60891 parameters")
colP1, colP2, colP3 = st.columns(3)
with colP1:
    alpha_Isc = st.number_input("α (A/°C) — Isc temp. coeff.", value=0.0005, format="%.6f",
                                help="Module Isc temperature coefficient (absolute, A/°C)")
with colP2:
    beta_Voc = st.number_input("β (V/°C) — Voc temp. coeff.", value=-0.10, format="%.4f",
                               help="Module Voc temperature coefficient (absolute, V/°C)")
with colP3:
    fit_kappa = st.checkbox("Also fit κ (curve correction)", value=False)

kappa_init = st.number_input("Initial κ (V/A/°C)", value=0.0, step=0.001, format="%.4f",
                             help="Set 0 if unknown; enable the κ toggle above to co-fit.")

run_60891 = st.button("Estimate Rs (IEC 60891)")

def _interp_to_grid(I2, V2, I_grid):
    # monotonic guard: sort by I2
    idx = np.argsort(I2)
    I2s, V2s = I2[idx], V2[idx]
    # restrict to grid within range
    mask = (I_grid >= I2s.min()) & (I_grid <= I2s.max())
    Vg = np.full_like(I_grid, np.nan, dtype=float)
    if mask.any():
        Vg[mask] = np.interp(I_grid[mask], I2s, V2s)
    return Vg

def _translate_proc1(df_sweep, Rs, kappa, alpha, beta, G2, T2):
    """
    IEC 60891 Procedure 1 translation for one sweep.
    Inputs: df_sweep must have columns V_V, I_A, G_Wm2, T_C (float)
    Returns I2, V2 (translated to G2,T2).
    """
    V1 = df_sweep["V_V"].to_numpy(dtype=float)
    I1 = df_sweep["I_A"].to_numpy(dtype=float)
    G1 = df_sweep["G_Wm2"].to_numpy(dtype=float)
    T1 = df_sweep["T_C"].to_numpy(dtype=float)

    # Estimate Isc1 per sweep (use max current near V≈0; here: simply max current)
    Isc1 = np.nanmax(I1)

    # Current translation (independent of Rs)
    I2 = I1 + Isc1 * (np.clip(G2,1e-3,None)/np.clip(G1,1e-3,None) - 1.0) + alpha * (T2 - T1)

    # Voltage translation (depends on Rs, κ, β)
    V2 = V1 - Rs * (I2 - I1) - kappa * I2 * (T2 - T1) + beta * (T2 - T1)
    return I2, V2

if run_60891:
    if iv60891_file is None:
        st.error("Please upload the multi‑IV CSV first.")
        st.stop()

    try:
        dff = pd.read_csv(iv60891_file)
    except Exception as e:
        st.error(f"CSV read error: {e}")
        st.stop()

    required_cols = {"SweepID", "V_V", "I_A", "G_Wm2", "T_C"}
    if not required_cols.issubset(set(dff.columns)):
        st.error(f"CSV must contain columns: {sorted(required_cols)}")
        st.stop()

    # Prepare groups
    groups = dff.groupby("SweepID", dropna=False)
    sweeps = []
    for gid, g in groups:
        g = g.copy()
        # sort by current to get consistent interpolation later
        g.sort_values(by=["I_A","V_V"], inplace=True)
        sweeps.append((gid, g))

    if len(sweeps) < 2:
        st.error("Need at least 2 measured I–V curves (preferably ≥3) for a robust 60891 Rs.")
        st.stop()

    # I-grid for cross-curve variance metric (use common overlapping current domain later)
    # Build preliminary translated currents to get a safe grid range using initial guesses.
    Rs0 = float(Rs_fit) if "Rs_fit" in globals() else 0.15
    kap0 = float(kappa_init)
    I2_ranges = []
    for _, g in sweeps:
        I2p, _ = _translate_proc1(g, Rs0, kap0, alpha_Isc, beta_Voc, G2, T2)
        I2_ranges.append((np.nanmin(I2p), np.nanmax(I2p)))
    I_min = np.max([a for a, _ in I2_ranges])
    I_max = np.min([b for _, b in I2_ranges])
    if not np.isfinite(I_min) or not np.isfinite(I_max) or I_max <= I_min:
        st.error("Translated curves have no common current range — check inputs (G,T,coefficients).")
        st.stop()

    I_grid = np.linspace(I_min, I_max, int(I_grid_pts))

    # Loss function: translate each sweep to (G2,T2), interpolate V2 on I_grid,
    # then compute variance (or MSE vs the mean) of V2 across curves.
    def loss_Rs_kappa(Rs, kappa):
        if Rs < 0 or not np.isfinite(Rs) or not np.isfinite(kappa):
            return 1e99
        V_stack = []
        for _, g in sweeps:
            I2, V2 = _translate_proc1(g, Rs, kappa, alpha_Isc, beta_Voc, G2, T2)
            Vg = _interp_to_grid(I2, V2, I_grid)
            V_stack.append(Vg)
        V_stack = np.vstack(V_stack)  # shape: n_sweeps × n_grid
        # Use rows that are valid across all sweeps
        valid = np.all(np.isfinite(V_stack), axis=0)
        if not np.any(valid):
            return 1e99
        Vs = V_stack[:, valid]
        meanV = np.nanmean(Vs, axis=0, keepdims=True)
        # SSE across curves relative to the mean at each grid point
        sse = np.nansum((Vs - meanV) ** 2)
        # Keep Rs in a plausible band (soft regularization)
        reg = 0.0
        return float(sse + reg)

    # Optimize Rs (and optionally κ): lightweight randomized search around initial guesses
    best = (Rs0, kap0)
    bestL = loss_Rs_kappa(best[0], best[1])
    rng = np.random.default_rng(2026)
    trials = 600 if fit_kappa else 400

    for _ in range(trials):
        if fit_kappa:
            t_Rs = max(0.0, best[0] + rng.normal(0, 0.02))
            t_k  = best[1] + rng.normal(0, 0.002)
        else:
            t_Rs = max(0.0, best[0] + rng.normal(0, 0.02))
            t_k  = kap0
        L = loss_Rs_kappa(t_Rs, t_k)
        if np.isfinite(L) and L < bestL:
            best, bestL = (t_Rs, t_k), L

    Rs_iec, kappa_iec = best
    st.success(f"**IEC 60891 estimate:**  Rs ≈ {Rs_iec:.4f} Ω" + (f",  κ ≈ {kappa_iec:.4f} V/A/°C" if fit_kappa else ""))

    # Plot the collapsed, translated curves
    figC, axC = plt.subplots(figsize=(6.6, 4.2))
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown"]
    for i, (gid, g) in enumerate(sweeps):
        I2, V2 = _translate_proc1(g, Rs_iec, kappa_iec, alpha_Isc, beta_Voc, G2, T2)
        axC.plot(I2, V2, lw=2, color=colors[i % len(colors)], label=f"{gid}")
    axC.set_xlabel("Current I₂ (A) @ target"); axC.set_ylabel("Voltage V₂ (V) @ target")
    axC.set_title(f"IEC 60891 Proc‑1 translation to (G₂={G2:.0f} W/m², T₂={T2:.1f} °C)")
    axC.grid(alpha=0.3); axC.legend(ncol=2, fontsize=9)
    st.pyplot(figC)

    # Export translated curves
    out_rows = []
    for gid, g in sweeps:
        I2, V2 = _translate_proc1(g, Rs_iec, kappa_iec, alpha_Isc, beta_Voc, G2, T2)
        tmp = pd.DataFrame({"SweepID": gid, "I2_A": I2, "V2_V": V2})
        out_rows.append(tmp)
    df_out = pd.concat(out_rows, ignore_index=True)
    st.download_button("Download translated curves (CSV)",
                       df_out.to_csv(index=False).encode("utf-8"),
                       file_name="IEC60891_translated_IV.csv", mime="text/csv")

    st.caption(
        "Notes: This implements **IEC 60891 Procedure 1** translations to a common target and "
        "estimates **Rs** by minimizing the variance among the translated curves on a shared current grid. "
        "Set **α (Isc temp. coeff.)** and **β (Voc temp. coeff.)** per your module type. "
        "For full normative details (and alternative Rs methods), see the 2021 Ed.3 and its annex."
    )
