#----------------------------------------------------------------------
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
#----------------------------------------------------------------------

# ------------------------------------------------------------------
# Laminar separation bubble checks.
#
# At Re=50k, a well-behaved airfoil should reach Cl_max smoothly
# over many alpha steps. Two failure modes are caught here:
#
# 1. Early stall (peak_idx < 20): fewer than 20 converged points
#    before Cl_max suggests the bubble burst early and XFOIL lost
#    convergence rather than finding a clean stall. With alpha_step=0.2
#    and alpha starting at 0, this means stall before ~4 degrees.
#
# 2. Non-monotonic Cl (max_drop < -0.05): a drop of more than 0.05
#    in Cl between consecutive alpha steps is the signature of a
#    separation bubble bursting and reattaching — the "wobble" visible
#    in the Cl-alpha curve. This makes XFOIL's drag predictions in
#    that region unreliable and the design fragile in practice.
# ------------------------------------------------------------------