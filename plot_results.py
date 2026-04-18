"""
plot_results.py  —  Visualize airfoil optimizer output.

Reads final_optimized_airfoil.dat and original_airfoil.dat (seed),
runs XFOIL on the seed if its polar is missing, then produces a
6-panel figure showing geometry and aerodynamic polars with the
project constraints highlighted.

Usage:
    python plot_results.py
"""

import os
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
XFOIL_PATH   = r"C:\Users\rapha\XFOIL6.99\xfoil.exe"
OPT_DAT      = "final_optimized_airfoil.dat"
SEED_DAT     = "original_airfoil.dat"          # written by the optimizer
OPT_POLAR    = "hires_polar.txt"               # written by the optimizer
SEED_POLAR   = "seed_polar_plot.txt"           # generated here if missing

# XFOIL run settings (match the optimizer's high-res sweep exactly)
N_PANELS     = 200
ITER_LIMIT   = 300
ALPHA_END    = 14.0
ALPHA_STEP   = 0.2
TIMEOUT      = 20.0

# Constraint limits (from project brief)
CM_LIMIT     = 0.0     # Cm @ Cl=0.8  ≥ 0  (tailless UAV, reflexed)
CD02_LIMIT   = 0.012   # Cd @ Cl=0.2  ≤ 0.012
CD08_LIMIT   = 0.04    # Cd @ Cl=0.8  ≤ 0.040

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_xfoil(dat_filename, polar_filename):
    """Run XFOIL free-transition sweep (TYPE 2, Re=50 k, Ncrit=12)."""
    if os.path.exists(polar_filename):
        os.remove(polar_filename)

    inp = f"_input_{polar_filename}.txt"
    with open(inp, "w") as f:
        f.write("PLOP\nG F\n\n")
        f.write(f"LOAD {dat_filename}\n")
        f.write(f"PPAR\nN {N_PANELS}\n\n\n")
        f.write("PANE\n\n\n")
        f.write("OPER\nTYPE 2\nVISC 50000\n")
        f.write(f"ITER {ITER_LIMIT}\n")
        f.write("VPAR\nN 12.0\n\n")
        f.write("ALFA 0\n")
        f.write("PACC\n")
        f.write(f"{polar_filename}\n\n")
        f.write(f"ASEQ 0 {ALPHA_END} {ALPHA_STEP}\n\nQUIT\n")

    try:
        subprocess.run(
            [XFOIL_PATH],
            stdin=open(inp, "r"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=TIMEOUT,
        )
    except Exception as e:
        print(f"  XFOIL warning: {e}")
    finally:
        if os.path.exists(inp):
            os.remove(inp)


def load_polar(polar_filename):
    """Return (alpha, cl, cd, cm) arrays, clipped at Cl_max."""
    if not os.path.exists(polar_filename):
        return None
    try:
        data = np.loadtxt(polar_filename, skiprows=12)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if len(data) < 4:
            return None
        alpha = data[:, 0]
        cl    = data[:, 1]
        cd    = data[:, 2]
        cm    = data[:, 4]
        peak  = np.argmax(cl)
        return alpha[:peak+1], cl[:peak+1], cd[:peak+1], cm[:peak+1]
    except Exception as e:
        print(f"  Could not load {polar_filename}: {e}")
        return None


def load_geometry(dat_filename):
    """Return (x, y) arrays from a Selig-format .dat file."""
    try:
        data = np.loadtxt(dat_filename, skiprows=1)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"  Could not load {dat_filename}: {e}")
        return None, None


def compute_metrics(cl, cd, cm):
    """Return dict of key constraint and objective values."""
    cl_cd  = cl / cd
    cl_max = np.max(cl)
    cl_cd_max = np.max(cl_cd)

    f_cd = interp1d(cl, cd, kind="linear", fill_value="extrapolate")
    f_cm = interp1d(cl, cm, kind="linear", fill_value="extrapolate")

    cd02 = float(f_cd(0.2))  if cl[0] <= 0.2  <= cl[-1] else float(f_cd(cl[0]))
    cd08 = float(f_cd(0.8))  if cl[0] <= 0.8  <= cl[-1] else np.nan
    cm08 = float(f_cm(0.8))  if cl[0] <= 0.8  <= cl[-1] else np.nan

    chi   = cl_cd_max * (cl_max / cd02)
    bonus = cm08 * 10000 if not np.isnan(cm08) else 0.0
    score = chi + bonus

    return dict(
        cl_max=cl_max, cl_cd_max=cl_cd_max,
        cd02=cd02, cd08=cd08, cm08=cm08,
        chi=chi, bonus=bonus, score=score,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Ensure seed polar exists -------------------------------------------
    if not os.path.exists(SEED_POLAR):
        if os.path.exists(SEED_DAT):
            print(f"Running XFOIL on seed airfoil ({SEED_DAT}) ...")
            run_xfoil(SEED_DAT, SEED_POLAR)
        else:
            print(f"WARNING: {SEED_DAT} not found — seed polar will be skipped.")

    # --- Load polars --------------------------------------------------------
    opt_result  = load_polar(OPT_POLAR)
    seed_result = load_polar(SEED_POLAR)

    if opt_result is None:
        print(f"ERROR: Could not load optimized polar from '{OPT_POLAR}'.""Run the optimizer first.")
        return

    a_opt, cl_opt, cd_opt, cm_opt = opt_result
    m_opt = compute_metrics(cl_opt, cd_opt, cm_opt)

    if seed_result is not None:
        a_seed, cl_seed, cd_seed, cm_seed = seed_result
        m_seed = compute_metrics(cl_seed, cd_seed, cm_seed)
        have_seed_polar = True
    else:
        have_seed_polar = False

    # --- Load geometries ----------------------------------------------------
    x_opt,  y_opt  = load_geometry(OPT_DAT)
    x_seed, y_seed = load_geometry(SEED_DAT)

    # --- Print summary ------------------------------------------------------
    print("\n" + "=" * 52)
    print(f"{'Metric':<30} {'Optimized':>10}  {'Seed':>10}")
    print("=" * 52)
    rows = [
        ("χ  (objective)",        "chi",      "{:.1f}"),
        ("Reflex bonus",          "bonus",    "{:.1f}"),
        ("Total score",           "score",    "{:.1f}"),
        ("Max Cl/Cd",             "cl_cd_max","{:.1f}"),
        ("Max Cl",                "cl_max",   "{:.3f}"),
        ("Cd @ Cl=0.2 (≤0.012)", "cd02",     "{:.4f}"),
        ("Cm @ Cl=0.8 (≥0.000)", "cm08",     "{:.4f}"),
        ("Cd @ Cl=0.8 (≤0.040)", "cd08",     "{:.4f}"),
    ]
    for label, key, fmt in rows:
        v_opt  = fmt.format(m_opt[key])
        v_seed = fmt.format(m_seed[key]) if have_seed_polar else "  —"
        print(f"  {label:<28} {v_opt:>10}  {v_seed:>10}")
    print("=" * 52 + "\n")

    # =========================================================================
    # Figure
    # =========================================================================
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Airfoil Optimization Results  —  Re = 50 000, Free Transition (N=12), Tailless UAV",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.42, wspace=0.38,
        left=0.07, right=0.97, top=0.93, bottom=0.07,
    )

    # Color palette
    C_OPT  = "#1f77b4"   # blue  — optimized
    C_SEED = "#ff7f0e"   # orange — seed / original
    C_LIM  = "#d62728"   # red   — constraint limit

    # -------------------------------------------------------------------------
    # Panel 0 (top, spanning all 3 columns): Airfoil geometry
    # -------------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0, :])

    if x_opt is not None:
        ax0.plot(x_opt, y_opt, color=C_OPT,  lw=1.8, label="Optimized")
    if x_seed is not None:
        ax0.plot(x_seed, y_seed, color=C_SEED, lw=1.4, ls="--", label="Seed (original)")

    ax0.axhline(0, color="k", lw=0.5, ls=":")
    ax0.set_aspect("equal")
    ax0.set_xlim(-0.02, 1.02)
    ax0.set_xlabel("x/c")
    ax0.set_ylabel("y/c")
    ax0.set_title("Airfoil Geometry Comparison")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, ls=":", alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 1 (row 1, col 0): Cl vs alpha
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1, 0])

    if have_seed_polar:
        ax1.plot(a_seed, cl_seed, color=C_SEED, lw=1.4, ls="--", label="Seed")
    ax1.plot(a_opt, cl_opt, color=C_OPT, lw=1.8, label="Optimized")
    ax1.axhline(0.8, color=C_LIM, lw=0.9, ls=":", label="Cl=0.8 (ref)")
    ax1.set_xlabel("α (°)")
    ax1.set_ylabel("Cl")
    ax1.set_title("Lift Curve")
    ax1.legend(fontsize=8)
    ax1.grid(True, ls=":", alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 2 (row 1, col 1): Drag polar  Cd vs Cl
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 1])

    if have_seed_polar:
        ax2.plot(cd_seed, cl_seed, color=C_SEED, lw=1.4, ls="--", label="Seed")
    ax2.plot(cd_opt, cl_opt, color=C_OPT, lw=1.8, label="Optimized")

    # Constraint markers
    ax2.axvline(CD02_LIMIT, color=C_LIM, lw=0.9, ls="--", label=f"Cd≤{CD02_LIMIT} @ Cl=0.2")
    ax2.axhline(0.2, color="grey", lw=0.7, ls=":")
    ax2.axhline(0.8, color="grey", lw=0.7, ls=":")

    # Mark operating point Cl=0.2
    ax2.plot(m_opt["cd02"], 0.2, "o", color=C_OPT, ms=7, label=f"Cd={m_opt['cd02']:.4f} @ Cl=0.2")

    ax2.set_xlabel("Cd")
    ax2.set_ylabel("Cl")
    ax2.set_title("Drag Polar")
    ax2.legend(fontsize=8)
    ax2.grid(True, ls=":", alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 3 (row 1, col 2): Cl/Cd vs Cl
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 2])

    cl_cd_opt = cl_opt / cd_opt
    if have_seed_polar:
        cl_cd_seed = cl_seed / cd_seed
        ax3.plot(cl_seed, cl_cd_seed, color=C_SEED, lw=1.4, ls="--", label="Seed")
    ax3.plot(cl_opt, cl_cd_opt, color=C_OPT, lw=1.8, label="Optimized")

    peak_i = np.argmax(cl_cd_opt)
    ax3.plot(cl_opt[peak_i], cl_cd_opt[peak_i], "*", color=C_OPT, ms=12, label=f"Max Cl/Cd={m_opt['cl_cd_max']:.1f}")
    ax3.axhline(0, color="k", lw=0.5)
    ax3.set_xlabel("Cl")
    ax3.set_ylabel("Cl / Cd")
    ax3.set_title("Lift-to-Drag Ratio")
    ax3.legend(fontsize=8)
    ax3.grid(True, ls=":", alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 4 (row 2, col 0): Cm vs Cl
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])

    if have_seed_polar:
        ax4.plot(cl_seed, cm_seed, color=C_SEED, lw=1.4, ls="--", label="Seed")
    ax4.plot(cl_opt, cm_opt, color=C_OPT, lw=1.8, label="Optimized")

    ax4.axhline(CM_LIMIT, color=C_LIM, lw=1.2, ls="--", label=f"Cm≥{CM_LIMIT:.1f} (constraint)")
    ax4.axvline(0.8, color="grey", lw=0.7, ls=":", label="Cl=0.8")

    if not np.isnan(m_opt["cm08"]):
        ax4.plot(0.8, m_opt["cm08"], "o", color=C_OPT, ms=7, label=f"Cm={m_opt['cm08']:.4f} @ Cl=0.8")

    # Shade the feasible region (Cm ≥ 0)
    ax4.axhspan(CM_LIMIT, ax4.get_ylim()[1] if ax4.get_ylim()[1] > CM_LIMIT else 0.05, alpha=0.08, color="green", label="Feasible region")

    ax4.set_xlabel("Cl")
    ax4.set_ylabel("Cm")
    ax4.set_title("Pitching Moment  (must be ≥ 0 @ Cl=0.8)")
    ax4.legend(fontsize=8)
    ax4.grid(True, ls=":", alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 5 (row 2, col 1): Cd vs alpha
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 1])

    if have_seed_polar:
        ax5.plot(a_seed, cd_seed, color=C_SEED, lw=1.4, ls="--", label="Seed")
    ax5.plot(a_opt, cd_opt, color=C_OPT, lw=1.8, label="Optimized")
    ax5.set_xlabel("α (°)")
    ax5.set_ylabel("Cd")
    ax5.set_title("Drag vs Angle of Attack")
    ax5.legend(fontsize=8)
    ax5.grid(True, ls=":", alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 6 (row 2, col 2): Score card text box
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")

    feasible = (
        m_opt["cm08"]  >= CM_LIMIT   and
        m_opt["cd02"]  <= CD02_LIMIT and
        (np.isnan(m_opt["cd08"]) or m_opt["cd08"] <= CD08_LIMIT) and
        m_opt["cl_max"] >= 0.8
    )
    status_str = "FEASIBLE  ✓" if feasible else "INFEASIBLE  ✗"
    status_col = "green" if feasible else "red"

    lines = [
        ("Status",             status_str,                                   status_col),
        ("",                   "",                                           "black"),
        ("Objective  χ = (Cl/Cd)max · Clmax/Cd02", f"{m_opt['chi']:.1f}", "black"),
        ("",                   "",                                           "black"),
        ("Max Cl/Cd",          f"{m_opt['cl_cd_max']:.2f}",                "black"),
        ("Max Cl",             f"{m_opt['cl_max']:.3f}",                    "black"),
        ("",                   "",                                           "black"),
        ("— Constraints —",    "",                                           "grey"),
        ("Cd @ Cl=0.2",        f"{m_opt['cd02']:.4f}  (≤ {CD02_LIMIT})",
            "green" if m_opt["cd02"] <= CD02_LIMIT else "red"),
        ("Cm @ Cl=0.8",        f"{m_opt['cm08']:.4f}  (≥ {CM_LIMIT:.1f})",
            "green" if m_opt["cm08"] >= CM_LIMIT else "red"),
        ("Cd @ Cl=0.8",        f"{m_opt['cd08']:.4f}  (≤ {CD08_LIMIT})" if not np.isnan(m_opt["cd08"]) else "n/a",
            "green" if (not np.isnan(m_opt["cd08"]) and m_opt["cd08"] <= CD08_LIMIT) else "red"),
    ]

    y_pos = 0.97
    dy    = 0.082
    ax6.text(0.5, y_pos + 0.015, "Optimized Airfoil — Summary", ha="center", va="top", fontsize=9.5, fontweight="bold", transform=ax6.transAxes)
    for label, value, col in lines:
        y_pos -= dy
        if label:
            ax6.text(0.02, y_pos, label + ":", ha="left", va="top", fontsize=8, transform=ax6.transAxes, color="black")
            ax6.text(0.98, y_pos, value, ha="right", va="top", fontsize=8, fontweight="bold", transform=ax6.transAxes, color=col)

    ax6.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="grey", lw=0.8, transform=ax6.transAxes, clip_on=False))

    # -------------------------------------------------------------------------
    # Save and show
    # -------------------------------------------------------------------------
    out_png = "airfoil_results.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Figure saved to  '{out_png}'")
    plt.show()


if __name__ == "__main__":
    main()
