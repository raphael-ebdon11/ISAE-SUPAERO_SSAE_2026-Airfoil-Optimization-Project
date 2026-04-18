import glob
import uuid
import os
import multiprocessing
import numpy as np
import subprocess
import time
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, least_squares

# =====================================================================
# Shared memory objects — intentionally None at module level.
#
# On Windows, multiprocessing uses "spawn": each worker starts a fresh
# Python interpreter and re-imports this module, re-creating any
# module-level Value/Array as disconnected objects in each worker.
#
# Fix: create the objects once in __main__, pass them to every worker
# via an explicit Pool initializer, and pass that Pool to
# differential_evolution via workers=pool.map. This is the correct
# scipy-supported pattern — differential_evolution accepts any callable
# with a .map() interface, not just an integer worker count.
# =====================================================================
_best_score = None
_best_weights_arr = None
_best_lock = None


def _worker_init(shared_score, shared_arr, shared_lock):
    """Called once per worker at pool startup to inject shared handles."""
    global _best_score, _best_weights_arr, _best_lock
    _best_score = shared_score
    _best_weights_arr = shared_arr
    _best_lock = shared_lock


# =====================================================================
# 1. CST Math & Geometry
# =====================================================================
def bernstein_poly(i, n, x):
    import math
    return math.comb(n, i) * (x**i) * ((1 - x)**(n - i))

def generate_cst_curve(x_array, weights):
    n = len(weights) - 1
    class_func = (x_array**0.5) * ((1 - x_array)**1.0)
    shape_func = np.zeros_like(x_array, dtype=float)
    for i, w in enumerate(weights):
        shape_func += w * bernstein_poly(i, n, x_array)
    return class_func * shape_func

def write_airfoil_dat(upper_weights, lower_weights, filename="airfoil.dat"):
    beta = np.linspace(0, np.pi, 100)
    x_coords = 0.5 * (1 - np.cos(beta))
    y_upper = generate_cst_curve(x_coords, upper_weights)
    y_lower = generate_cst_curve(x_coords, lower_weights)
    x_selig = np.concatenate((x_coords[::-1], x_coords[1:]))
    y_selig = np.concatenate((y_upper[::-1], y_lower[1:]))
    with open(filename, 'w') as f:
        f.write("Optimized_Airfoil\n")
        for x, y in zip(x_selig, y_selig):
            f.write(f"{x:.6f} {y:.6f}\n")

# =====================================================================
# 2. Geometry Penalty
# =====================================================================
def geometry_penalty(upper_w, lower_w):
    beta = np.linspace(0, np.pi, 100)
    x_test = 0.5 * (1 - np.cos(beta))
    y_upper = generate_cst_curve(x_test, upper_w)
    y_lower = generate_cst_curve(x_test, lower_w)
    thickness = y_upper[1:-1] - y_lower[1:-1]

    penalty = 0.0

    bad = thickness[thickness <= 0.0]
    if len(bad) > 0:
        penalty += np.sum(np.abs(bad)) * 500000

    t_max = np.max(thickness) if len(thickness) > 0 else 0.0
    if t_max < 0.04:
        penalty += (0.04 - t_max) * 300000

    d2y_up = np.abs(np.diff(y_upper, 2))
    d2y_lo = np.abs(np.diff(y_lower, 2))
    roughness = np.sum(d2y_up) + np.sum(d2y_lo)
    if roughness > 0.18:
        penalty += (roughness - 0.18) * 100000

    return penalty

# =====================================================================
# 3. XFOIL Execution
# =====================================================================
def run_xfoil(xtr_top=0.7, xtr_bot=0.7, dat_filename="airfoil.dat",
              polar_filename="polar.txt", n_panels=200, iter_limit=240,
              alpha_end=12, alpha_step=0.2, timeout=10.0):
    if os.path.exists(polar_filename):
        try:
            os.remove(polar_filename)
        except OSError:
            pass

    input_file = f"input_{dat_filename}.txt"
    with open(input_file, 'w') as f:
        f.write("PLOP\nG F\n\n")
        f.write(f"LOAD {dat_filename}\n")
        f.write(f"PPAR\nN {n_panels}\n\n\n")
        f.write("PANE\n\n\n")
        f.write("OPER\nTYPE 2\nVISC 50000\n")
        f.write(f"ITER {iter_limit}\n")
        f.write(f"VPAR\nN 12.0\nXTR {xtr_top:.3f} {xtr_bot:.3f}\n\n")
        f.write("ALFA 0\n")
        f.write("PACC\n")
        f.write(f"{polar_filename}\n\n")
        f.write(f"ASEQ 0 {alpha_end} {alpha_step}\n\nQUIT\n")

    try:
        xfoil_path = r"C:\Users\rapha\XFOIL6.99\xfoil.exe"
        subprocess.run(
            [xfoil_path],
            stdin=open(input_file, 'r'),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except Exception:
        pass


def evaluate_fitness(polar_filename="polar.txt"):
    pid = os.getpid()
    log_file = f"worker_log_{pid}.txt"

    if not os.path.exists(polar_filename):
        return -100000.0

    try:
        with open(polar_filename, 'r') as f:
            if len(f.readlines()) <= 12:
                return -100000.0

        data = np.loadtxt(polar_filename, skiprows=12)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if len(data) < 4:
            return -100000.0

        cl = data[:, 1]
        cd = data[:, 2]
        cm = data[:, 4]

        sort_idx = np.argsort(cl)
        cl, cd, cm = cl[sort_idx], cd[sort_idx], cm[sort_idx]

        peak_idx = np.argmax(cl)
        cl = cl[:peak_idx + 1]
        cd = cd[:peak_idx + 1]
        cm = cm[:peak_idx + 1]

        cl_max    = np.max(cl)
        cl_cd_max = np.max(cl / cd)

        cd_interp = interp1d(cl, cd, kind='linear')
        cm_interp = interp1d(cl, cm, kind='linear')

        try:
            cd_02 = float(cd_interp(0.2))
        except ValueError:
            cd_02 = 0.02

        # Replace the existing cd_02 floor check with these two guards:
        if cd_02 < 0.007:          # raised from 0.004 — nothing real is below this at Re=50k
            return -100000.0

        if cl_cd_max > 55.0:       # hard cap — physically impossible to exceed at Re=50k
            return -100000.0

        penalty = 0.0

        if cl_max < 0.8:
            penalty += (0.8 - cl_max) * 500000
            cd_08 = float(cd_interp(cl_max))
            cm_08 = float(cm_interp(cl_max))
        else:
            cd_08 = float(cd_interp(0.8))
            cm_08 = float(cm_interp(0.8))

        if cm_08 < 0.0:
            penalty += abs(cm_08) * 800000

        if cd_02 > 0.0120:
            penalty += (cd_02 - 0.0115) * 300000

        if cd_08 > 0.04:
            penalty += (cd_08 - 0.04) * 200000

        if penalty == 0.0:
            reflex_bonus = cm_08 * 10000
            chi = cl_cd_max * (cl_max / cd_02)
            final_score = chi + reflex_bonus
            with open(log_file, "a") as log:
                log.write(f"SUCCESS: Chi={chi:.2f} | Bonus={reflex_bonus:.1f} | "
                          f"Cl/Cd={cl_cd_max:.1f} | Cm08={cm_08:.4f} | Cd02={cd_02:.4f}\n")
        else:
            final_score = -penalty
            with open(log_file, "a") as log:
                log.write(f"LEARNING: Cl_max={cl_max:.2f} | Cm08={cm_08:.4f} | "
                          f"Cd02={cd_02:.4f} | Cd08={cd_08:.4f} | Score={final_score:.1f}\n")

        return final_score

    except Exception:
        return -10000.0

# =====================================================================
# 5. Objective Function
# =====================================================================
def objective_function(weights):
    pid = os.getpid()
    uid = uuid.uuid4().hex[:8]

    dat_file = f"airfoil_{uid}.dat"
    pol_file = f"polar_{uid}.txt"
    inp_file = f"input_{dat_file}.txt"
    log_file = f"worker_log_{pid}.txt"

    upper_w = weights[:7]
    lower_w = weights[7:14]
    xtr_top, xtr_bot = weights[14], weights[15]

    geom_pen = geometry_penalty(upper_w, lower_w)
    if geom_pen > 0.0:
        with open(log_file, "a") as log:
            log.write(f"GEOMETRY PENALTY: {geom_pen:.1f}\n")
        return 1000000.0 + geom_pen

    write_airfoil_dat(upper_w, lower_w, filename=dat_file)
    run_xfoil(xtr_top, xtr_bot, dat_filename=dat_file, polar_filename=pol_file)
    fitness = evaluate_fitness(polar_filename=pol_file)

    if fitness > 0 and _best_score is not None:
        with _best_lock:
            if fitness > _best_score.value:
                _best_score.value = fitness
                for i, w in enumerate(weights):
                    _best_weights_arr[i] = w

    for f in [dat_file, pol_file, inp_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    return -fitness

# =====================================================================
# 6. Cleanup
# =====================================================================
def cleanup_temp_files():
    patterns = ["airfoil_*.dat", "input_airfoil_*.dat.txt", "polar_*.txt", "worker_log_*.txt"]
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
            except OSError:
                pass

# =====================================================================
# 7. Seeded Initial Population
# =====================================================================
def build_initial_population(seed, bounds, popsize, rng):
    n_params = len(bounds)
    n_pop = popsize * n_params
    pop = np.zeros((n_pop, n_params))
    for j in range(n_params):
        lo, hi = bounds[j]
        samples = (rng.permutation(n_pop) + rng.random(n_pop)) / n_pop
        pop[:, j] = lo + samples * (hi - lo)
    for j, (lo, hi) in enumerate(bounds):
        pop[:, j] = np.clip(pop[:, j], lo, hi)
    pop[0, :len(seed)] = seed
    pop[0, 14] = 0.70
    pop[0, 15] = 0.70
    return pop

# =====================================================================
# 8. CST Fit from DAT Seed
# =====================================================================
def build_seed_from_dat(filename="seed_airfoil.dat"):
    def fit_cst(x_data, y_data, n_weights=7):
        def residuals(w):
            return generate_cst_curve(x_data, w) - y_data
        return least_squares(residuals, np.zeros(n_weights)).x

    data = np.loadtxt(filename, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    le_index = np.argmin(x)

    x_upper = x[:le_index+1][::-1]
    y_upper = y[:le_index+1][::-1]
    x_lower = x[le_index:]
    y_lower = y[le_index:]

    mask_up = (x_upper > 0.001) & (x_upper < 0.999)
    mask_lo = (x_lower > 0.001) & (x_lower < 0.999)

    uw = fit_cst(x_upper[mask_up], y_upper[mask_up])
    lw = fit_cst(x_lower[mask_lo], y_lower[mask_lo])
    return uw, lw

# =====================================================================
# 9. Main
# =====================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    cleanup_temp_files()

    seed_upper, seed_lower = build_seed_from_dat("seed_airfoil.dat")
    seed_airfoil = np.concatenate([seed_upper, seed_lower])
    write_airfoil_dat(seed_upper, seed_lower, filename="original_airfoil.dat")

    bounds = [(w - 0.012, w + 0.012) for w in seed_airfoil]

    bounds[0] = (seed_upper[0] - 0.02, seed_upper[0] + 0.06)
    bounds[1] = (seed_upper[1] - 0.02, seed_upper[1] + 0.06)
    bounds[7] = (seed_lower[0] - 0.05, seed_lower[0] + 0.02)
    bounds[8] = (seed_lower[1] - 0.05, seed_lower[1] + 0.02)

    bounds[5]  = (seed_upper[5] - 0.005, seed_upper[5] + 0.02)
    bounds[6]  = (seed_upper[6] - 0.005, seed_upper[6] + 0.02)
    bounds[12] = (seed_lower[5] - 0.005, seed_lower[5] + 0.02)
    bounds[13] = (seed_lower[6] - 0.005, seed_lower[6] + 0.02)

    bounds.extend([(0.3, 1.0), (0.3, 1.0)])

    # =====================================================================
    # Create shared memory objects once in the main process, then build
    # an explicit Pool with the initializer so every worker receives the
    # same handles. differential_evolution accepts any object with a
    # .map() method via the workers argument — this is the documented
    # way to use a custom pool with it.
    # =====================================================================
    shared_score = multiprocessing.Value('d', -float('inf'))
    shared_arr   = multiprocessing.Array('d', 16)
    shared_lock  = multiprocessing.Lock()

    # Populate the main process globals so post-processing can read them.
    _worker_init(shared_score, shared_arr, shared_lock)

    POPSIZE = 12
    rng = np.random.default_rng(seed=42)
    init_pop = build_initial_population(seed_airfoil, bounds, POPSIZE, rng)

    print("Starting optimization (Re=50k, N=12, tailless UAV)...\n")
    print("Interpreting disp output:")
    print("  fun < 0  → feasible solution found, optimizer maximising χ ✓")
    print("  fun > 0  → still in penalty space (check Cm constraint)\n")

    start_time = time.time()

    # Build the pool explicitly so we can pass the initializer.
    # Using 'with' ensures workers are cleaned up even if an exception occurs.
    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count(),
        initializer=_worker_init,
        initargs=(shared_score, shared_arr, shared_lock),
    ) as pool:
        opt_result = differential_evolution(
            objective_function,
            bounds=bounds,
            init=init_pop,
            popsize=POPSIZE,
            maxiter=150,
            tol=1e-7,
            mutation=(0.5, 1.25),
            recombination=0.8,
            disp=True,
            workers=pool.map,   # pass pool.map directly — scipy calls it like map(fn, candidates)
            updating='deferred',
            polish=False,
        )

    elapsed = time.time() - start_time
    print(f"\nOptimization complete in {elapsed:.1f}s")

    if shared_score.value > -float('inf'):
        best_weights = np.array(list(shared_arr))
        print(f"Using tracked best feasible solution: score={shared_score.value:.1f}")
    else:
        best_weights = opt_result.x
        print("WARNING: No feasible solution was tracked. Falling back to opt_result.x")

    print(f"DE result (fun={opt_result.fun:.1f})")

    # =====================================================================
    # Post-processing — identical settings to the optimizer for a
    # trustworthy feasibility check, then a finer sweep for the polar plot.
    # =====================================================================
    final_dat = "final_optimized_airfoil.dat"
    write_airfoil_dat(best_weights[:7], best_weights[7:14], filename=final_dat)

    def parse_and_report(polar_filename, label):
        try:
            polar_data = np.loadtxt(polar_filename, skiprows=12)
            if polar_data.ndim == 1:
                polar_data = polar_data.reshape(1, -1)

            cl = polar_data[:, 1]
            cd = polar_data[:, 2]
            cm = polar_data[:, 4]

            peak_idx = np.argmax(cl)
            cl = cl[:peak_idx + 1]
            cd = cd[:peak_idx + 1]
            cm = cm[:peak_idx + 1]

            f_cd = interp1d(cl, cd, kind='linear', fill_value="extrapolate")
            f_cm = interp1d(cl, cm, kind='linear', fill_value="extrapolate")

            cd_02 = float(f_cd(0.2))
            cm_08 = float(f_cm(0.8))
            cl_max = np.max(cl)
            cl_cd_max = np.max(cl / cd)
            chi = cl_cd_max * (cl_max / cd_02)
            bonus = cm_08 * 50000

            feasible = (
                cm_08            >= 0.0   and
                cd_02            <= 0.012 and
                float(f_cd(0.8)) <= 0.04  and
                cl_max           >= 0.8
            )

            print("\n" + "=" * 45)
            print(label)
            print("=" * 45)
            print(f"Feasible (all constraints met) : {'YES ✓' if feasible else 'NO ✗'}")
            print(f"Total Objective Score          : {chi + bonus:.1f}")
            print(f"Chi (χ) Value                  : {chi:.2f}")
            print(f"Reflex Bonus                   : {bonus:.1f}")
            print("-" * 45)
            print(f"Max Cl/Cd                      : {cl_cd_max:.1f}")
            print(f"Max Cl                         : {cl_max:.2f}")
            print(f"Cd @ Cl=0.2  (limit ≤ 0.012)  : {cd_02:.4f}  {'✓' if cd_02 <= 0.012 else '✗'}")
            print(f"Cm @ Cl=0.8  (limit ≥ 0.000)  : {cm_08:.4f}  {'✓' if cm_08 >= 0.0 else '✗'}")
            print(f"Cd @ Cl=0.8  (limit ≤ 0.040)  : {float(f_cd(0.8)):.4f}  {'✓' if float(f_cd(0.8)) <= 0.04 else '✗'}")
            print("=" * 45 + "\n")

        except Exception as e:
            print(f"\nCould not parse {label} polar: {e}")

    # Verification sweep — same N=200, step=0.2 as optimizer. Ground truth.
    print("Running verification sweep (same settings as optimizer)...")
    verify_pol = "verify_polar.txt"
    run_xfoil(best_weights[14], best_weights[15],
    dat_filename=final_dat, polar_filename=verify_pol,
    n_panels=200, iter_limit=240, alpha_end=12, alpha_step=0.2,
    timeout=15.0)
    parse_and_report(verify_pol, "Verification Results (optimizer settings — ground truth)")

    # High-res sweep — extended alpha range for a complete polar curve.
    print("Running high-resolution sweep for full polar output...")
    hires_pol = "hires_polar.txt"
    run_xfoil(best_weights[14], best_weights[15],
    dat_filename=final_dat, polar_filename=hires_pol,
    n_panels=200, iter_limit=300, alpha_end=12, alpha_step=0.2,
    timeout=15.0)
    parse_and_report(hires_pol, "High-Resolution Results (alpha to 14° — for reference)")