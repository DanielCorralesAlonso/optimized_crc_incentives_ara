"""Old-vs-new check for the public scheme's replicate refactor.

Rebuilds the PREVIOUS _simulate_theta (which redrew the population at every
state of nature) and compares it with the new block-based one.  At theta index 0
the two must agree EXACTLY -- the old seed was sim_seed + 0, which is the seed
the blocks are drawn with -- and across states they must agree in mean to within
Monte-Carlo error.
"""
import sys, types, time, numpy as np, pandas as pd
sys.path.insert(0, "v2")
class _Any(types.ModuleType):
    __path__ = []
    def __getattr__(self, n): return type(n, (), {})
class _Finder:
    def find_module(self, name, path=None): return self if _stub(name) else None
    def load_module(self, name):
        m = _Any(name); sys.modules[name] = m; return m
def _stub(n):
    root = n.split(".")[0]
    return root in ("pgmpy", "pysmile", "pysmile_license", "xgboost", "sklearn")
import importlib.abc, importlib.machinery
class F(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if _stub(name):
            return importlib.machinery.ModuleSpec(name, _L())
        return None
class _L(importlib.abc.Loader):
    def create_module(self, spec): return _Any(spec.name)
    def exec_module(self, module): pass
sys.meta_path.insert(0, F())

import costs_and_utilities as cu
import public_incentive_scheme as pis
from screening_assignment import to_profiles, OUT_FILE

profiles = to_profiles(pd.read_csv(OUT_FILE))
k_axis   = np.linspace(0.0, 150.0, 25)
static   = pis._precompute_static(profiles, k_axis)
P        = pis._ara_sweep(profiles, k_axis, 400, 12345)
NREP, SIM = 200, 777
u_pm = cu.u_pm_risk_neutral

def legacy(P, static, u_pm, n_rep, seed, theta_bar):
    M = P.shape[1]; n_total = int(static["n"].sum()); reps = np.empty((M, n_rep))
    for m in range(M):
        rec = cu.draw_replicates(
            static["n"], static["q"], static["sen"], static["spe"], P[:, m],
            static["comp"][:, m], static["hpar"][:, m], static["tslope"][:, m],
            n_total, float(static["k_axis"][m]), n_rep,
            np.random.default_rng(seed), theta_bar)
        reps[m] = np.asarray(u_pm(rec))
    d = reps - reps[0]
    return reps.mean(axis=1), d.std(axis=1, ddof=1)/np.sqrt(n_rep)

blocks = pis._draw_blocks(P, static, NREP, SIM)
rng = np.random.default_rng(0); th0 = cu.draw_theta_bar(rng)
u_old, se_old = legacy(P, static, u_pm, NREP, SIM + 0, th0)
u_new, se_new = pis._simulate_theta(blocks, u_pm, th0)
print(f"theta 0, exact match expected:")
print(f"  max |u_old - u_new|  = {np.abs(u_old-u_new).max():.3e} "
      f"(rel {np.abs(u_old-u_new).max()/np.abs(u_old).max():.2e})")
print(f"  max |se_old - se_new| = {np.abs(se_old-se_new).max():.3e}")

M_T = 100
t0=time.time()
old = np.array([legacy(P, static, u_pm, NREP, SIM+m, th)[0]
                for m, th in enumerate([cu.draw_theta_bar(np.random.default_rng(0))
                                        for _ in range(1)]*0 +
                                       [cu.draw_theta_bar(r) for r in [np.random.default_rng(0)]
                                        for _ in range(M_T)])])
t_old=time.time()-t0
r = np.random.default_rng(0)
t0=time.time()
new = np.array([pis._simulate_theta(blocks, u_pm, cu.draw_theta_bar(r))[0] for _ in range(M_T)])
t_new=time.time()-t0
co, cn = old-old[:,[0]], new-new[:,[0]]
print(f"\n{M_T} states: legacy {t_old:.1f}s | blocks {t_new:.2f}s | "
      f"speedup {t_old/t_new:.0f}x")
print(f"  curve mean  old {co.mean(0)[12]:8.3f}  new {cn.mean(0)[12]:8.3f}  "
      f"(mid-grid, MC se ~{co.std(0)[12]/np.sqrt(M_T):.3f})")
print(f"  max |mean diff| across grid: {np.abs(co.mean(0)-cn.mean(0)).max():.3f}")
print(f"  band sd  old {co.std(0)[12]:.2f}  new {cn.std(0)[12]:.2f}  "
      f"(new is lower: replicate noise removed)")
