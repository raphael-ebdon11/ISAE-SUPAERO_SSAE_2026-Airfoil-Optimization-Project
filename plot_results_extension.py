"""
=============================================================================
EXTENSION TO plot_results.py
=============================================================================
Adds the analyses requested by the rubric:
  (1) Pressure coefficient (Cp) distributions at the three required Cl values
  (2) Boundary layer analysis (Cf, shape factor H, transition/separation)
      at the three required Cl values
  (3) Center of pressure (Xcp/c) trace vs Cl

The three Cl targets come directly from the rubric:
    - Cl = 0.2            (cruise / low-lift)
    - Cl at L/D_max       (from hires_polar.txt, ~0.73)
    - Cl = Cl_max - 0.1   (just below stall, ~0.76)

HOW TO INTEGRATE:
    1. Drop this file next to your existing plot_results.py (or paste the
       functions into it).
    2. At the bottom of plot_results.py, after your existing plotting,
       add the three calls shown in `run_rubric_extensions()` below.
    3. Make sure XFOIL is on PATH (same requirement as your main optimizer).
=============================================================================
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Configuration — edit these paths to match your repo layout
# -----------------------------------------------------------------------------
AIRFOIL_DAT   = "final_optimized_airfoil.dat"                        # the .dat you feed XFOIL
HIRES_POLAR   = "hires_polar.txt"                                    # existing high-res polar
XFOIL_EXE     = r"C:\Users\rapha\XFOIL6.99\xfoil.exe"                # full path on Windows
REYNOLDS      = 5.0e4
NCRIT         = 12.0
N_PANELS      = 200
XFOIL_ITER    = 400
OUTPUT_DIR    = "."                                                  # where plots are saved


# =============================================================================
# 1. PARSE THE EXISTING POLAR TO FIND ANGLES AT THE TARGET Cl VALUES
# =============================================================================

def read_xfoil_polar(polar_path):
    """
    Parse a standard XFOIL polar file (the one created by PACC).
    Returns a dict of numpy arrays: alpha, CL, CD, CDp, CM, Xtr_top, Xtr_bot.
    """
    rows = []
    with open(polar_path, "r") as f:
        data_started = False
        for line in f:
            s = line.strip()
            # XFOIL polar data starts after the "------" separator line
            if s.startswith("----"):
                data_started = True
                continue
            if not data_started or not s:
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            try:
                rows.append([float(x) for x in parts[:7]])
            except ValueError:
                continue

    arr = np.array(rows)
    return {
        "alpha":   arr[:, 0],
        "CL":      arr[:, 1],
        "CD":      arr[:, 2],
        "CDp":     arr[:, 3],
        "CM":      arr[:, 4],
        "Xtr_top": arr[:, 5],
        "Xtr_bot": arr[:, 6],
    }


def find_alpha_for_cl(polar, cl_target):
    """Linear interpolation to get alpha at a target Cl (monotonic region only)."""
    cl = polar["CL"]
    al = polar["alpha"]

    # Restrict to the monotonically increasing pre-stall region so we don't
    # pick up a post-stall solution with the same Cl.
    idx_peak = int(np.argmax(cl))
    cl_pre   = cl[:idx_peak + 1]
    al_pre   = al[:idx_peak + 1]

    if cl_target > cl_pre.max() or cl_target < cl_pre.min():
        raise ValueError(
            f"Cl={cl_target:.3f} outside converged polar range "
            f"[{cl_pre.min():.3f}, {cl_pre.max():.3f}]"
        )
    return float(np.interp(cl_target, cl_pre, al_pre))


def identify_target_alphas(polar):
    """
    Return the three (label, cl_target, alpha, cd_actual) tuples
    required by the rubric.
    """
    cl = polar["CL"]
    cd = polar["CD"]
    al = polar["alpha"]

    # Cl at L/D_max (restricted to the pre-peak region for consistency)
    idx_peak = int(np.argmax(cl))
    ld = cl[:idx_peak + 1] / np.maximum(cd[:idx_peak + 1], 1e-8)
    idx_ldmax = int(np.argmax(ld))
    cl_ldmax  = float(cl[idx_ldmax])

    cl_max  = float(cl.max())
    cl_near = cl_max - 0.1  # "a bit below Cl_max" per the rubric

    targets = [
        ("Cl = 0.2 (cruise)",                  0.20,     None),
        (f"Cl @ L/D_max = {cl_ldmax:.2f}",     cl_ldmax, None),
        (f"Cl = Cl_max - 0.1 = {cl_near:.2f}", cl_near,  None),
    ]

    results = []
    for label, cl_t, _ in targets:
        try:
            alpha_t = find_alpha_for_cl(polar, cl_t)
            cd_t    = float(np.interp(alpha_t, al, cd))
            results.append((label, cl_t, alpha_t, cd_t))
        except ValueError as e:
            print(f"[warn] {label}: {e}")
    return results


# =============================================================================
# 2. DRIVE XFOIL TO EXTRACT Cp AND BOUNDARY LAYER DATA AT A GIVEN ALPHA
# =============================================================================

def run_xfoil_point(airfoil_dat, alpha, workdir):
    """
    Run XFOIL at a single alpha (viscous, free-transition N=12) and write:
        cp_alpha.txt   — surface Cp from CPWR
        bl_alpha.txt   — BL integral params from DUMP

    Returns (cp_path, bl_path). Both are strings written inside `workdir`.
    """
    airfoil_abs = os.path.basename(airfoil_dat)  # use bare name; XFOIL runs in workdir
    cp_path     = os.path.join(workdir, f"cp_a{alpha:+.2f}.txt")
    bl_path     = os.path.join(workdir, f"bl_a{alpha:+.2f}.txt")

    # Remove stale output files so XFOIL doesn't prompt to overwrite
    for p in (cp_path, bl_path):
        if os.path.exists(p):
            os.remove(p)

    cmds = "\n".join([
        "PLOP",
        "G F",           # disable graphics
        "",
        f"LOAD {airfoil_abs}",
        "",              # accept default airfoil name
        "PPAR",
        f"N {N_PANELS}",
        "",
        "",
        "OPER",
        f"VISC {REYNOLDS:.1f}",
        "VPAR",
        f"N {NCRIT}",
        "",
        f"ITER {XFOIL_ITER}",
        f"ALFA {alpha:.4f}",
        f"CPWR {os.path.basename(cp_path)}",
        f"DUMP {os.path.basename(bl_path)}",
        "",
        "QUIT",
        "",
    ])

    proc = subprocess.run(
        [XFOIL_EXE],
        input=cmds,
        capture_output=True,
        text=True,
        cwd=workdir,
        timeout=90,
    )

    if not (os.path.exists(cp_path) and os.path.exists(bl_path)):
        raise RuntimeError(
            f"XFOIL failed at alpha={alpha:.2f} deg.\n"
            f"stdout tail:\n{proc.stdout[-800:]}"
        )
    return cp_path, bl_path


def parse_cpwr(path):
    """
    CPWR format: two header lines (Alfa/CL line + column-name line) then x y Cp data.
    Returns numpy arrays x, y, cp.
    """
    data = np.loadtxt(path, skiprows=2)
    return data[:, 0], data[:, 1], data[:, 2]


def parse_dump(path):
    """
    DUMP format: header + rows of
        s   x   y   Ue/Vinf   Dstar   Theta   Cf   H
    Only the airfoil surface portion is kept (discard the wake rows, which
    sit after the trailing-edge point; XFOIL marks the wake by s > s_TE).
    Returns a dict of arrays restricted to the airfoil surface.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 8:
                continue
            try:
                rows.append([float(v) for v in parts[:8]])
            except ValueError:
                continue
    arr = np.array(rows)

    x_all = arr[:, 1]

    # The wake portion sits at the tail of the file and has x monotonically
    # increasing past x ~ 1.0. The surface portion goes TE -> LE -> TE with
    # x in [0, 1]. Keep rows where x <= 1.0 + small tol.
    mask = x_all <= 1.0 + 1e-6
    arr  = arr[mask]

    return {
        "s":      arr[:, 0],
        "x":      arr[:, 1],
        "y":      arr[:, 2],
        "Ue":     arr[:, 3],
        "Dstar":  arr[:, 4],
        "Theta":  arr[:, 5],
        "Cf":     arr[:, 6],
        "H":      arr[:, 7],
    }


def split_surfaces(x, y, values):
    """
    XFOIL writes surface data starting at the TE, going over the upper
    surface to the LE, then over the lower surface back to the TE.
    Split by the LE index (argmin of x).
    Returns (upper_dict, lower_dict) each with keys x, y, val — ordered LE->TE.
    """
    i_le = int(np.argmin(x))
    # Upper: from start (TE) to LE  -> reverse so it goes LE -> TE
    upper = {
        "x":   x[:i_le + 1][::-1],
        "y":   y[:i_le + 1][::-1],
        "val": values[:i_le + 1][::-1],
    }
    lower = {
        "x":   x[i_le:],
        "y":   y[i_le:],
        "val": values[i_le:],
    }
    return upper, lower


# =============================================================================
# 3. PLOTS — Cp DISTRIBUTIONS
#    [FIX 1]: Annotation placement no longer collides with the title
# =============================================================================

def plot_cp_distributions(points_data, save_path):
    """
    points_data: list of dicts with keys
        label, cl_target, alpha, cp_path
    Produces a single figure with subplots, one per operating point.
    """
    n = len(points_data)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, pt in zip(axes, points_data):
        x, y, cp = parse_cpwr(pt["cp_path"])
        upper, lower = split_surfaces(x, y, cp)

        ax.plot(upper["x"], upper["val"], "-",  color="#1f77b4", lw=1.6, label="Upper")
        ax.plot(lower["x"], lower["val"], "--", color="#d62728", lw=1.6, label="Lower")

        # Annotate the suction peak (most negative Cp on upper surface)
        i_peak = int(np.argmin(upper["val"]))
        cp_min = upper["val"][i_peak]
        x_peak = upper["x"][i_peak]
        ax.plot(x_peak, cp_min, "o", color="#1f77b4", ms=6)

        # Place the annotation to the right of the peak and slightly BELOW it
        # (less-negative Cp direction). This keeps the label inside the axes
        # and away from the title, regardless of how deep the suction peak is.
        y_offset = 0.35 * abs(cp_min) if abs(cp_min) > 1 else 0.25
        ax.annotate(
            f"$C_{{p,\\mathrm{{min}}}}$ = {cp_min:.2f}\n@ x/c = {x_peak:.2f}",
            xy=(x_peak, cp_min),
            xytext=(max(x_peak + 0.15, 0.25), cp_min + y_offset),
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", lw=0.6, color="gray"),
        )

        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.invert_yaxis()  # convention: -Cp upward
        ax.set_xlabel("x/c")
        ax.set_title(f"{pt['label']}\n$\\alpha$ = {pt['alpha']:.2f}°", pad=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")

    axes[0].set_ylabel("$C_p$")
    fig.suptitle("Pressure Coefficient Distributions — Optimized Airfoil",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[ok] Cp plot saved -> {save_path}")
    plt.close(fig)


# =============================================================================
# 4. PLOTS — BOUNDARY LAYER (Cf and H)
#    [FIX 2]: Cf y-axis clipped so stagnation singularity doesn't dominate
# =============================================================================

def _find_transition_index(cf_upper_x, cf_upper_val):
    """
    Heuristic: laminar Cf is small and smooth; turbulent reattachment causes
    an abrupt rise in Cf. Return the x/c of the steepest positive Cf gradient
    on the upper surface as the transition marker. This is a visual aid only —
    XFOIL's Xtr value from the polar file is the authoritative one.
    """
    if len(cf_upper_x) < 5:
        return None
    dcf_dx = np.gradient(cf_upper_val, cf_upper_x)
    # Look only past the leading edge region
    mask = cf_upper_x > 0.05
    if not np.any(mask):
        return None
    idx = np.argmax(dcf_dx[mask])
    return float(cf_upper_x[mask][idx])


def plot_boundary_layer(points_data, xtr_top_values, xtr_bot_values, save_path):
    """
    One row per operating point. Columns:
        (a) Cf vs x/c  — upper and lower, with Xtr markers
        (b) H  vs x/c  — separation indicated by H > ~4
    """
    n = len(points_data)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.6 * n), squeeze=False)

    for row, (pt, xtr_top, xtr_bot) in enumerate(
        zip(points_data, xtr_top_values, xtr_bot_values)
    ):
        bl = parse_dump(pt["bl_path"])
        up_cf, lo_cf = split_surfaces(bl["x"], bl["y"], bl["Cf"])
        up_H,  lo_H  = split_surfaces(bl["x"], bl["y"], bl["H"])

        # ----- Cf panel -----
        ax_cf = axes[row][0]
        ax_cf.plot(up_cf["x"], up_cf["val"], "-",  color="#1f77b4", lw=1.5, label="Upper")
        ax_cf.plot(lo_cf["x"], lo_cf["val"], "--", color="#d62728", lw=1.5, label="Lower")
        ax_cf.axhline(0.0, color="k", lw=0.7, alpha=0.6)

        # Mark XFOIL's transition locations from the polar
        if 0.0 < xtr_top < 1.0:
            ax_cf.axvline(xtr_top, color="#1f77b4", ls=":", lw=1.2, label=f"Xtr_top = {xtr_top:.2f}")
        if 0.0 < xtr_bot < 1.0:
            ax_cf.axvline(xtr_bot, color="#d62728", ls=":", lw=1.2, label=f"Xtr_bot = {xtr_bot:.2f}")

        # Clip the y-axis so the LE stagnation singularity doesn't flatten
        # the interesting behavior against the zero line. The key physics
        # (laminar Cf ~ 1e-3, turbulent Cf ~ 5e-3 to 1e-2, separation Cf < 0)
        # all sits inside this range.
        ax_cf.set_ylim(-0.004, 0.014)
        ax_cf.set_xlim(-0.02, 1.02)

        ax_cf.set_xlabel("x/c")
        ax_cf.set_ylabel("$C_f$")
        ax_cf.set_title(f"Skin friction — {pt['label']}, "
                        f"$\\alpha$ = {pt['alpha']:.2f}°")
        ax_cf.grid(alpha=0.3)
        ax_cf.legend(fontsize=8, loc="upper right")

        # ----- H panel -----
        ax_H = axes[row][1]
        ax_H.plot(up_H["x"], up_H["val"], "-",  color="#1f77b4", lw=1.5, label="Upper")
        ax_H.plot(lo_H["x"], lo_H["val"], "--", color="#d62728", lw=1.5, label="Lower")
        # H ~ 2.5 = laminar attached; H > ~4 = laminar separation;
        # H drops to ~1.5-1.8 when turbulent.
        ax_H.axhline(4.0, color="gray", ls=":", lw=1.0, label="H = 4 (separation)")
        ax_H.axhline(2.6, color="gray", ls="-.", lw=0.8, alpha=0.6, label="H = 2.6 (laminar BL)")

        ax_H.set_xlabel("x/c")
        ax_H.set_ylabel("Shape factor $H$")
        ax_H.set_title(f"Shape factor — {pt['label']}")
        ax_H.set_ylim(1.0, min(10.0, np.max(up_H["val"]) * 1.1))
        ax_H.grid(alpha=0.3)
        ax_H.legend(fontsize=8, loc="best")

    fig.suptitle("Boundary Layer State — Optimized Airfoil", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[ok] BL plot saved -> {save_path}")
    plt.close(fig)


# =============================================================================
# 5. CENTER OF PRESSURE TRACE
#    [FIX 3]: Overlay points are now read FROM the polar curve itself so the
#    dots sit on the line (no method-mismatch inconsistency).
# =============================================================================

def compute_xcp_from_cp(cp_path):
    """
    Compute Xcp/c from a CPWR file by integrating (Cp_lower - Cp_upper)*x
    over the chord and dividing by the net normal force integral.

        Xcp/c = integral( (Cp_l - Cp_u) * x  dx ) / integral( (Cp_l - Cp_u) dx )

    NOTE: This uses a chord-projected thin-airfoil approximation and ignores
    the chordwise (axial) force contribution to the moment. For a reflex
    cambered airfoil at low Cl, it can disagree with the polar-based
    Xcp = 0.25 - Cm/Cl by 0.1-0.2 chord. The polar-based method is the
    standard reported value and is what the plot in plot_xcp_trace() uses.
    This function is retained for the summary printout as a sanity check.
    """
    x, y, cp = parse_cpwr(cp_path)
    upper, lower = split_surfaces(x, y, cp)

    # Put both surfaces on a common chordwise grid (0..1) for the integral
    xg = np.linspace(0.0, 1.0, 400)
    cp_u = np.interp(xg, upper["x"], upper["val"])
    cp_l = np.interp(xg, lower["x"], lower["val"])

    dcp = cp_l - cp_u       # net "suction"
    num = np.trapezoid(dcp * xg, xg)
    den = np.trapezoid(dcp, xg)
    if abs(den) < 1e-8:
        return np.nan
    return float(num / den)


def plot_xcp_trace(polar, extra_points, save_path):
    """
    Plot Xcp/c vs Cl using the polar-based thin-airfoil relation:

        Xcp/c = 0.25 - Cm_c4 / Cl

    The three rubric points are overlaid using values interpolated from
    the same curve, so the dots sit on the line (no methodology mismatch).
    """
    cl = polar["CL"]
    cm = polar["CM"]

    valid = np.abs(cl) > 0.05
    xcp_curve = np.full_like(cl, np.nan, dtype=float)
    xcp_curve[valid] = 0.25 - cm[valid] / cl[valid]

    # Sort by Cl (restricted to pre-peak region) so interpolation is well-defined
    idx_peak = int(np.argmax(cl))
    cl_mono  = cl[:idx_peak + 1]
    xcp_mono = xcp_curve[:idx_peak + 1]
    order    = np.argsort(cl_mono)
    cl_sorted  = cl_mono[order]
    xcp_sorted = xcp_mono[order]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(cl[valid], xcp_curve[valid], "-", color="#2ca02c", lw=1.8,
            label="$X_{cp}/c = 0.25 - C_m/C_l$")

    # Overlay the three rubric points using values from the same curve
    colors = ["#1f77b4", "#ff7f0e", "#9467bd"]
    for pt, color in zip(extra_points, colors):
        cl_pt  = pt["cl_actual"]
        xcp_pt = float(np.interp(cl_pt, cl_sorted, xcp_sorted))
        # Stash on the point so the summary printout can use the consistent value
        pt["xcp_polar"] = xcp_pt
        ax.plot(cl_pt, xcp_pt, "o", ms=9, color=color,
                markeredgecolor="black", markeredgewidth=0.6, zorder=5,
                label=f"{pt['label']}: $X_{{cp}}/c$ = {xcp_pt:.3f}")

    ax.axhline(0.25, color="gray", ls=":", lw=1.0, alpha=0.7,
               label="Quarter-chord")
    ax.set_xlabel("$C_l$")
    ax.set_ylabel("$X_{cp}/c$")
    ax.set_title("Center of Pressure vs Lift Coefficient — Optimized Airfoil")
    # Tighter y-range so the variation is visible
    ax.set_ylim(0.15, 0.55)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[ok] Xcp plot saved -> {save_path}")
    plt.close(fig)


# =============================================================================
# 6. ORCHESTRATOR — call this from plot_results.py
# =============================================================================

def run_rubric_extensions(
    airfoil_dat=AIRFOIL_DAT,
    polar_path=HIRES_POLAR,
    output_dir=OUTPUT_DIR,
):
    """
    End-to-end runner. Produces three figures:
        - cp_distributions.png
        - boundary_layer.png
        - xcp_trace.png
    and prints a summary table of BL/Cp values at the three rubric points.
    """
    os.makedirs(output_dir, exist_ok=True)
    polar = read_xfoil_polar(polar_path)
    targets = identify_target_alphas(polar)

    print("\n=== Rubric operating points ===")
    print(f"{'Label':<35} {'Cl_target':>10} {'alpha (deg)':>12} {'Cd':>10}")
    for label, cl_t, alpha, cd_t in targets:
        print(f"{label:<35} {cl_t:>10.3f} {alpha:>12.3f} {cd_t:>10.5f}")

    # Run XFOIL at each target alpha
    workdir = tempfile.mkdtemp(prefix="rubric_ext_")
    shutil.copy(airfoil_dat, os.path.join(workdir, os.path.basename(airfoil_dat)))

    points_data = []
    xtr_top_vals, xtr_bot_vals = [], []

    for label, cl_t, alpha, cd_t in targets:
        print(f"\n[xfoil] Running alpha = {alpha:.2f} deg for '{label}' ...")
        try:
            cp_path, bl_path = run_xfoil_point(
                airfoil_dat=os.path.basename(airfoil_dat),
                alpha=alpha,
                workdir=workdir,
            )
            # Interpolate Xtr values from the already-computed polar
            xtr_top = float(np.interp(alpha, polar["alpha"], polar["Xtr_top"]))
            xtr_bot = float(np.interp(alpha, polar["alpha"], polar["Xtr_bot"]))
            cl_act  = float(np.interp(alpha, polar["alpha"], polar["CL"]))
            xcp_int = compute_xcp_from_cp(cp_path)   # kept as sanity check

            points_data.append({
                "label":        label,
                "cl_target":    cl_t,
                "cl_actual":    cl_act,
                "alpha":        alpha,
                "cp_path":      cp_path,
                "bl_path":      bl_path,
                "xtr_top":      xtr_top,
                "xtr_bot":      xtr_bot,
                "xcp_integral": xcp_int,
                # xcp_polar is added by plot_xcp_trace() below
            })
            xtr_top_vals.append(xtr_top)
            xtr_bot_vals.append(xtr_bot)

            print(f"    Xtr_top = {xtr_top:.3f}, Xtr_bot = {xtr_bot:.3f}, "
                  f"Xcp/c (Cp-int) = {xcp_int:.3f}")
        except Exception as e:
            print(f"  [fail] {e}")

    if not points_data:
        print("No valid points — aborting.")
        return

    # Generate plots
    plot_cp_distributions(
        points_data,
        save_path=os.path.join(output_dir, "cp_distributions.png"),
    )
    plot_boundary_layer(
        points_data, xtr_top_vals, xtr_bot_vals,
        save_path=os.path.join(output_dir, "boundary_layer.png"),
    )
    plot_xcp_trace(
        polar, points_data,
        save_path=os.path.join(output_dir, "xcp_trace.png"),
    )

    # Final summary table for the report — uses the polar-based Xcp for
    # consistency with the plot.
    print("\n=== Summary (copy into report) ===")
    header = (f"{'Operating point':<35} {'alpha':>8} {'Cl':>8} "
              f"{'Xtr_top':>9} {'Xtr_bot':>9} {'Xcp/c':>8}")
    print(header)
    print("-" * len(header))
    for p in points_data:
        xcp_reported = p.get("xcp_polar", p["xcp_integral"])
        print(f"{p['label']:<35} {p['alpha']:>8.2f} {p['cl_actual']:>8.3f} "
              f"{p['xtr_top']:>9.3f} {p['xtr_bot']:>9.3f} "
              f"{xcp_reported:>8.3f}")

    # Clean up XFOIL scratch files (optional — comment out if you want to keep
    # the raw cp_*.txt / bl_*.txt for the appendix)
    # shutil.rmtree(workdir)
    print(f"\nRaw XFOIL outputs kept in: {workdir}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_rubric_extensions()