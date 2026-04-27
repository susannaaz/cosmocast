from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


THETA_FULL: dict[str, dict[str, dict[str, float]]] = {
    "ucor": {
        "theta0": {
            "omega_b": 0.02237,
            "omega_cdm": 0.11933,
            "h": 0.6766,
            "tau_reio": 0.0561,
            "P_RR_1": 2.3e-9,
            "P_RR_2": 2.3e-9,
            "P_II_1": 1.5e-9,
            "P_II_2": 1.5e-9,
        },
        "corr": {"P_RI_1": 1.0e-15, "P_RI_2": 1.0e-15},
    },
    "acor": {
        "theta0": {
            "omega_b": 0.02237,
            "omega_cdm": 0.11933,
            "h": 0.6766,
            "tau_reio": 0.0561,
            "P_RR_1": 2.3e-9,
            "P_RR_2": 2.3e-9,
            "P_II_1": 1.5e-9,
            "P_II_2": 1.5e-9,
        },
        "corr": {"P_RI_1": -1.85e-9, "P_RI_2": 1.85e-9},
    },
    "pcor": {
        "theta0": {
            "omega_b": 0.02237,
            "omega_cdm": 0.11933,
            "h": 0.6766,
            "tau_reio": 0.0561,
            "P_RR_1": 2.3e-9,
            "P_RR_2": 2.3e-9,
            "P_II_1": 1.5e-9,
            "P_II_2": 1.5e-9,
        },
        "corr": {"P_RI_1": 1.85e-9, "P_RI_2": 1.85e-9},
    },
    "ad": {
        "theta0": {
            "omega_b": 0.02237,
            "omega_cdm": 0.11933,
            "h": 0.6766,
            "tau_reio": 0.0561,
            "P_RR_1": 2.34e-9,
            "P_RR_2": 2.04e-9,
        },
        "corr": {},
    },
}

LATEX_LABELS = {
    "omega_b": r"\omega_b",
    "omega_cdm": r"\omega_{\rm cdm}",
    "h": r"h",
    "tau_reio": r"\tau",
    "P_RR_1": r"10^{10}P_{RR,1}",
    "P_RR_2": r"10^{10}P_{RR,2}",
    "P_II_1": r"10^{10}P_{II,1}",
    "P_II_2": r"10^{10}P_{II,2}",
}

LB_FREQ = [40, 50, 60, 68, 78, 89, 100, 119, 140, 166, 195, 235, 280, 337, 402]

# Per nominal frequency: list of (sigma_P [uK*arcmin], beam FWHM [arcmin]).
# Bands with two entries are dual-telescope; they get combined by inverse-variance.
LB_SUBCOMPONENTS: dict[int, list[tuple[float, float]]] = {
    40: [(37.42, 70.5)],
    50: [(33.46, 58.5)],
    60: [(21.31, 51.1)],
    68: [(19.91, 41.6), (31.77, 47.1)],
    78: [(15.55, 36.9), (19.13, 43.8)],
    89: [(12.28, 33.0), (28.77, 41.5)],
    100: [(10.34, 30.2), (8.48, 37.8)],
    119: [(7.69, 26.3), (5.70, 33.6)],
    140: [(7.25, 23.7), (6.38, 30.8)],
    166: [(5.57, 28.9)],
    195: [(7.05, 28.0), (10.50, 28.6)],
    235: [(10.79, 24.7)],
    280: [(13.80, 22.5)],
    337: [(21.95, 20.9)],
    402: [(47.45, 17.9)],
}


def litebird_noise_curves(
    *,
    lmin: int = 2,
    lmax: int = 200,
    dell: int = 10,
) -> dict[str, Any]:
    """
    Return LiteBIRD inverse-variance combined noise curves on the ell grid
    used in the older notebook: ell = arange(lmin, lmax, dell).

    Important: this is intentionally NOT scaled by time; it matches the
    hardcoded LiteBIRD table from the original script.
    """
    import numpy as np

    ell = np.arange(lmin, lmax, dell, dtype=float)
    n_ell_ee_freq = np.zeros((len(LB_FREQ), len(ell)), dtype=float)
    n_ell_tt_freq = np.zeros((len(LB_FREQ), len(ell)), dtype=float)

    for i, freq in enumerate(LB_FREQ):
        inv_ee = np.zeros_like(ell, dtype=float)
        inv_tt = np.zeros_like(ell, dtype=float)
        for (sigma_p_arcmin, fwhm_arcmin) in LB_SUBCOMPONENTS[int(freq)]:
            sigma_p_rad = np.deg2rad(sigma_p_arcmin / 60.0)
            sigma_t_rad = sigma_p_rad / np.sqrt(2.0)
            beam_rad = fwhm_arcmin / np.sqrt(8.0 * np.log(2.0)) / 60.0 * np.pi / 180.0
            n_ee_comp = sigma_p_rad**2 * 4.0 * np.pi * np.exp(ell * (ell + 1.0) * beam_rad**2)
            n_tt_comp = sigma_t_rad**2 * 4.0 * np.pi * np.exp(ell * (ell + 1.0) * beam_rad**2)
            inv_ee += 1.0 / n_ee_comp
            inv_tt += 1.0 / n_tt_comp
        n_ell_ee_freq[i] = 1.0 / inv_ee
        n_ell_tt_freq[i] = 1.0 / inv_tt

    # Combine across frequencies by inverse-variance (map-level combination approximation).
    n_ell_ee = 1.0 / np.sum(1.0 / n_ell_ee_freq, axis=0)
    n_ell_tt = 1.0 / np.sum(1.0 / n_ell_tt_freq, axis=0)
    return {"ell": ell.astype(int), "n_ell_ee": n_ell_ee, "n_ell_tt": n_ell_tt}


def _repo_root() -> Path:
    # nbs/utils.py -> repo root
    return Path(__file__).resolve().parent.parent


def _ensure_fisher_multi_importable() -> None:
    # Pickles reference fisher_multi.FisherResult.
    path = _repo_root() / "cosmocast_makelik" / "multi_freq_liq"
    sys.path.append(str(path))


def load_pickle(path: str | Path) -> Any:
    p = Path(path)
    with p.open("rb") as f:
        return pickle.load(f)


def load_so_fisher_all(path: str | Path | None = None) -> dict:
    _ensure_fisher_multi_importable()
    if path is None:
        path = _repo_root() / "fisher_data" / "so_fisher_all.pkl"
    return load_pickle(path)


def load_litebird_grids(path: str | Path | None = None) -> dict:
    _ensure_fisher_multi_importable()
    if path is None:
        path = _repo_root() / "fisher_data" / "lb_fisher_10.pkl"
    return load_pickle(path)


def _iso_tag(iso: str | None) -> str:
    return "adiabatic" if iso is None else str(iso)


def _maybe_skip(path: Path, skip_existing: bool) -> bool:
    return skip_existing and path.exists()


def _restrict_fisher_params(fisher, *, keep_params: list[str]):
    """
    Return a FisherResult restricted to a subset of parameters.

    Used for the adiabatic-only forecast, where older pickles may still include
    isocurvature parameters in the `iso=None` case.
    """
    import numpy as np

    missing = [p for p in keep_params if p not in fisher.param_list]
    if missing:
        raise KeyError(f"Cannot restrict FisherResult; missing params: {missing}")
    idx = [fisher.param_list.index(p) for p in keep_params]
    F_sub = fisher.F[np.ix_(idx, idx)]
    Cov_sub = np.linalg.pinv(F_sub)
    sigma_sub = np.sqrt(np.diag(Cov_sub))
    dC_sub = fisher.dC
    if isinstance(getattr(fisher, "dC", None), list) and len(fisher.dC) == len(fisher.param_list):
        dC_sub = [fisher.dC[i] for i in idx]
    return fisher.__class__(
        F=F_sub,
        Cov_params=Cov_sub,
        sigma=sigma_sub,
        dC=dC_sub,
        bands=fisher.bands,
        param_list=keep_params,
        metadata=getattr(fisher, "metadata", {}) or {},
    )


@dataclass(frozen=True)
class JointDataEntry:
    """
    One element of the joint data vector:
        (experiment, spectrum, ell_bin)
    with per-bin metadata needed for covariance construction.
    """

    experiment: str
    spectrum: str  # "TT", "TE", "EE"
    ell: int
    dell: int
    fsky: float
    var: float
    n_channels: int = 1


def save_fisher_plots(
    fisher,
    *,
    theta0: dict,
    scaled10_params: set[str],
    out_dir: str | Path,
    stem: str,
    label_map: dict[str, str] | None = None,
    skip_existing: bool = True,
    write_table: bool = True,
    write_corr: bool = True,
    write_triangle: bool = True,
) -> None:
    """
    Save the core Fisher plot types used by the original script:
    - summary table image
    - correlation matrix image
    - getdist triangle plot
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_map = label_map or LATEX_LABELS

    if write_table:
        table_path = out_dir / f"{stem}_summary.png"
        if not _maybe_skip(table_path, skip_existing):
            fisher.save_summary_table(
                theta0=theta0,
                scaled_params=scaled10_params,
                exp_name=stem,
                save_path=str(table_path),
            )

    if write_corr:
        corr_path = out_dir / f"{stem}_corr.png"
        if not _maybe_skip(corr_path, skip_existing):
            fisher.plot_correlation(exp_name=stem, save_path=str(corr_path))

    if write_triangle:
        tri_path = out_dir / f"{stem}_triangle.pdf"
        if not _maybe_skip(tri_path, skip_existing):
            fisher.plot_triangle(
                theta0=theta0,
                label_map=label_map,
                scaled_params=scaled10_params,
                exp_name=stem,
                save_path=str(tri_path),
            )
            try:
                import matplotlib.pyplot as plt

                plt.close("all")
            except Exception:
                pass


def _default_steps_for_param_list(param_list: Sequence[str]) -> dict[str, float]:
    """
    Match the step sizes used in the pickle-generation utilities in this repo.

    This is only used for COMBINATION_MODE=="joint_cov", where we need to
    recompute derivatives from theory even when the per-experiment Fishers
    are loaded from pickle.
    """
    steps_abs = {"omega_b": 2.5e-5, "omega_cdm": 1.5e-4, "h": 6.0e-4, "tau_reio": 6.0e-5}
    steps = dict(steps_abs)
    if "P_RR_1" in param_list:
        steps["P_RR_1"] = 0.01
    if "P_RR_2" in param_list:
        steps["P_RR_2"] = 0.01
    if "P_II_1" in param_list:
        steps["P_II_1"] = 0.05
    if "P_II_2" in param_list:
        steps["P_II_2"] = 0.05
    if "ln10A_s" in param_list:
        # ln(1e10 A_s); typical Fisher step in this space
        steps["ln10A_s"] = 0.01
    if "A_s" in param_list:
        steps["A_s"] = 0.01 * 2.1e-9
    if "n_s" in param_list:
        steps["n_s"] = 0.004
    missing = [p for p in param_list if p not in steps]
    if missing:
        raise KeyError(f"No default step size known for params: {missing}")
    return steps


def _scaled10_params_for_param_list(param_list: Sequence[str]) -> set[str]:
    out = set()
    for p in ["P_RR_1", "P_RR_2", "P_II_1", "P_II_2"]:
        if p in param_list:
            out.add(p)
    return out


def _tag_float(x: float, *, ndp: int = 3) -> str:
    """
    Filesystem-friendly float tag, e.g. 0.007 -> '0p007'.
    """
    s = f"{float(x):.{int(ndp)}f}"
    return s.replace(".", "p").replace("-", "m")


def make_run_tag(
    *,
    year_tag: str,
    combination_mode: str,
    use_iso_signal_in_cov: bool,
    planck_likelihood_mode: str = "pliklite_lowT",
    planck_tau_prior_sigma: float = 0.007,
    ell_by_ell_policy: str = "min_noise",
    use_standard_lcdm_amplitude: bool = False,
    use_ns: bool = False,
    param_list: Sequence[str] | None = None,
) -> str:
    """
    Construct a run tag that makes outputs easy to distinguish.

    Backwards compatibility:
      - Returns legacy tags for the common defaults:
        * fisher_sum + pliklite_lowT + iso-signal-in-cov -> '{year_tag}'
        * joint_cov -> '{year_tag}_jointcov' (plus optional suffixes)
        * ell_by_ell -> '{year_tag}_ellbyell' (plus optional suffixes)
    """
    mode = str(combination_mode)
    base = str(year_tag)

    if mode == "fisher_sum":
        base = base
    elif mode == "joint_cov":
        base = f"{base}_jointcov"
    elif mode == "ell_by_ell":
        base = f"{base}_ellbyell"
    else:
        base = f"{base}_{mode}"

    suffix: list[str] = []
    if not bool(use_iso_signal_in_cov):
        suffix.append("noisosignal")
    if mode == "ell_by_ell" and ell_by_ell_policy and ell_by_ell_policy != "min_noise":
        suffix.append(f"policy-{ell_by_ell_policy}")
    if planck_likelihood_mode == "TTTEEE_lowE":
        suffix.append(f"plancklowE-tauprior{_tag_float(planck_tau_prior_sigma, ndp=3)}")
    elif planck_likelihood_mode != "pliklite_lowT":
        suffix.append(f"planck-{planck_likelihood_mode}")

    if bool(use_standard_lcdm_amplitude):
        suffix.append("stdAs")
    if bool(use_ns):
        suffix.append("ns")

    if param_list is not None:
        pset = set(param_list)
        if "ln10A_s" in pset or "A_s" in pset:
            suffix.append("stdAs")
        if "n_s" in pset:
            suffix.append("ns")
        if any(p.startswith("P_RR_") for p in pset):
            suffix.append("binnedAs")

    # Preserve the historical default directory name for the original workflow.
    if (
        mode == "fisher_sum"
        and bool(use_iso_signal_in_cov)
        and planck_likelihood_mode == "pliklite_lowT"
        and not suffix
    ):
        return str(year_tag)

    if suffix:
        return base + "_" + "_".join(suffix)
    return base


def _theta0_standard_lcdm(*, theta0_base: dict[str, float], n_s: float = 0.965, A_s: float = 2.1e-9) -> dict[str, float]:
    """
    Construct a theta0 dict for standard adiabatic ΛCDM runs that use ln10A_s/n_s.
    Keeps the background parameters from the repo's THETA_FULL but swaps the
    primordial amplitude parameterization.
    """
    import numpy as np

    th = {
        "omega_b": float(theta0_base["omega_b"]),
        "omega_cdm": float(theta0_base["omega_cdm"]),
        "h": float(theta0_base["h"]),
        "tau_reio": float(theta0_base["tau_reio"]),
        "ln10A_s": float(np.log(1e10 * float(A_s))),
        "n_s": float(n_s),
    }
    return th


def _ensure_theta0_has_standard_lcdm_params(
    *,
    theta0: dict[str, float],
    param_list: Sequence[str],
    default_A_s: float = 2.1e-9,
    default_n_s: float = 0.965,
) -> dict[str, float]:
    """
    Defensive helper: ensure theta0 contains keys requested in `param_list` for
    the standard adiabatic ΛCDM parameterization (ln10A_s and/or n_s).

    Some workflows drive the forecast from legacy THETA_FULL fiducials that do
    not contain `ln10A_s`/`n_s`; fisher_multi will raise KeyError when it tries
    to perturb those parameters unless we seed them here.
    """
    import numpy as np

    th = dict(theta0)
    pset = set(param_list)

    if "ln10A_s" in pset and "ln10A_s" not in th:
        if "A_s" in th:
            th["ln10A_s"] = float(np.log(1e10 * float(th["A_s"])))
        else:
            th["ln10A_s"] = float(np.log(1e10 * float(default_A_s)))

    if "A_s" in pset and "A_s" not in th:
        if "ln10A_s" in th:
            th["A_s"] = float(np.exp(float(th["ln10A_s"])) * 1e-10)
        else:
            th["A_s"] = float(default_A_s)

    if "n_s" in pset and "n_s" not in th:
        th["n_s"] = float(default_n_s)

    return th


def _cls_provider_with_standard_lcdm_amplitude(compute_cls: Callable, *, iso_mode: str | None):
    """
    Wrap iso_theory.compute_cls so fisher_multi can differentiate w.r.t. ln10A_s.
    Also ensures CLASS receives A_s/n_s in standard mode (power_mode='standard').
    """
    import numpy as np

    def _provider(*, lmax: int, **th):
        th_now = dict(th)
        power_mode = "legacy_binned"
        if iso_mode is None and ("ln10A_s" in th_now or "A_s" in th_now or "n_s" in th_now):
            power_mode = "standard"
        if "ln10A_s" in th_now:
            th_now["A_s"] = float(np.exp(float(th_now.pop("ln10A_s"))) * 1e-10)
        # In standard mode, ignore legacy binned primordial params if present.
        if power_mode == "standard":
            for k in ["P_RR_1", "P_RR_2", "P_II_1", "P_II_2", "P_RI_1", "P_RI_2", "k1", "k2"]:
                th_now.pop(k, None)
        return _compute_cls_indexed(compute_cls, lmax=int(lmax), iso_mode=iso_mode, power_mode=power_mode, **th_now)

    return _provider


def _jointcov_letters(spec: str) -> tuple[str, str]:
    if spec == "TT":
        return ("T", "T")
    if spec == "EE":
        return ("E", "E")
    if spec == "TE":
        return ("T", "E")
    raise KeyError(f"Unsupported spectrum '{spec}' (expected TT/TE/EE)")


def _fid_cl(fiducial_cls: dict[str, Any], a: str, b: str, ell: int) -> float:
    """
    Access fiducial C_ell^{ab} with (T,E) conventions.
    Assumes fiducial_cls values are arrays indexed by integer ell.
    """
    import numpy as np

    if (a, b) == ("T", "T"):
        return float(np.asarray(fiducial_cls["TT"])[ell])
    if (a, b) == ("E", "E"):
        return float(np.asarray(fiducial_cls["EE"])[ell])
    if (a, b) in {("T", "E"), ("E", "T")}:
        return float(np.asarray(fiducial_cls["TE"])[ell])
    raise KeyError(f"Unsupported field pair ({a},{b})")


def _gaussian_cov_cl(
    *,
    fiducial_cls: dict[str, Any],
    spec1: str,
    spec2: str,
    ell: int,
    fsky: float,
    dell: int,
) -> float:
    """
    Standard Gaussian covariance for binned CMB spectra, evaluated on the *signal* only:

        Cov(C_ell^{XY}, C_ell^{WZ}) = [ C^{XW} C^{YZ} + C^{XZ} C^{YW} ] / ((2 ell + 1) fsky dell)

    Noise terms are intentionally excluded here; they are assumed independent across
    experiments (and across channels when compressing within an experiment).
    """
    X, Y = _jointcov_letters(spec1)
    W, Z = _jointcov_letters(spec2)
    denom = (2.0 * float(ell) + 1.0) * float(fsky) * float(dell)
    if denom <= 0:
        return 0.0
    c_xw = _fid_cl(fiducial_cls, X, W, ell)
    c_yz = _fid_cl(fiducial_cls, Y, Z, ell)
    c_xz = _fid_cl(fiducial_cls, X, Z, ell)
    c_yw = _fid_cl(fiducial_cls, Y, W, ell)
    return (c_xw * c_yz + c_xz * c_yw) / denom


def _resolve_fsky_overlap(
    f_sky_overlap: Any,
    *,
    exp_a: str,
    exp_b: str,
    fsky_a: float,
    fsky_b: float,
    ell: int,
) -> float:
    """
    f_sky_overlap can be:
      - None: use min(fsky_a, fsky_b)
      - dict: keys like ('Planck','SO') or ('SO','Planck')
      - callable: (exp_a, exp_b, ell, fsky_a, fsky_b) -> float
    """
    if f_sky_overlap is None:
        return float(min(fsky_a, fsky_b))
    if callable(f_sky_overlap):
        return float(f_sky_overlap(exp_a, exp_b, ell, fsky_a, fsky_b))
    if isinstance(f_sky_overlap, dict):
        if (exp_a, exp_b) in f_sky_overlap:
            return float(f_sky_overlap[(exp_a, exp_b)])
        if (exp_b, exp_a) in f_sky_overlap:
            return float(f_sky_overlap[(exp_b, exp_a)])
    return float(min(fsky_a, fsky_b))


def _entries_from_bands(
    *,
    experiment: str,
    bands: Sequence[Any],
    spectra: Sequence[str],
    fiducial_cls: dict[str, Any],
    f_sky_overlap: Any = None,
    compress_channels: bool = True,
) -> list[JointDataEntry]:
    """
    Convert a list of fisher_multi.SpectrumBand-like objects to a per-experiment
    joint vector with entries (experiment, spectrum, ell_bin).

    If compress_channels=True, multiple channels/frequency-pairs at the same
    (spectrum, ell) are optimally combined while accounting for shared cosmic
    variance (signal-only cross-cov, independent noise).
    """
    import numpy as np

    wanted = set(spectra)
    by_key: dict[tuple[str, int], list[tuple[float, float, int]]] = {}
    # (spec, ell) -> list of (var, fsky, dell)
    for b in bands:
        if getattr(b, "cell_type", None) not in wanted:
            continue
        spec = str(b.cell_type)
        ell_arr = np.asarray(getattr(b, "ell"), dtype=float)
        var_arr = np.asarray(getattr(b, "cov"), dtype=float)
        fsky = float(getattr(b, "fsky", 1.0))
        dell = int(getattr(b, "dell", 1))
        for e, v in zip(ell_arr, var_arr):
            ell = int(np.round(float(e)))
            by_key.setdefault((spec, ell), []).append((float(v), fsky, dell))

    entries: list[JointDataEntry] = []
    for (spec, ell), triples in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        if not compress_channels or len(triples) == 1:
            v, fsky, dell = triples[0]
            entries.append(
                JointDataEntry(
                    experiment=experiment,
                    spectrum=spec,
                    ell=int(ell),
                    dell=int(dell),
                    fsky=float(fsky),
                    var=float(v),
                    n_channels=len(triples),
                )
            )
            continue

        # Optimal combination of multiple bandpower estimators measuring the same sky:
        # Var(combined) = 1 / (1^T Cov^{-1} 1), with off-diagonal signal-only covariance.
        m = len(triples)
        cov = np.zeros((m, m), dtype=float)
        for i, (vi, fskyi, delli) in enumerate(triples):
            cov[i, i] = float(vi)
            for j in range(i + 1, m):
                vj, fskyj, dellj = triples[j]
                fsky_ij = _resolve_fsky_overlap(
                    f_sky_overlap,
                    exp_a=experiment,
                    exp_b=experiment,
                    fsky_a=float(fskyi),
                    fsky_b=float(fskyj),
                    ell=int(ell),
                )
                dell_ij = int(min(int(delli), int(dellj)))
                cov_ij = _gaussian_cov_cl(
                    fiducial_cls=fiducial_cls,
                    spec1=spec,
                    spec2=spec,
                    ell=int(ell),
                    fsky=float(fsky_ij),
                    dell=int(dell_ij),
                )
                cov[i, j] = cov_ij
                cov[j, i] = cov_ij

        one = np.ones((m, 1), dtype=float)
        try:
            cov_inv = np.linalg.inv(cov)
        except Exception:
            cov_inv = np.linalg.pinv(cov)
        denom = float((one.T @ cov_inv @ one).ravel()[0])
        if denom <= 0 or not np.isfinite(denom):
            # Fallback: conservative "best single channel" variance.
            v_eff = float(min(t[0] for t in triples))
        else:
            v_eff = 1.0 / denom

        # fsky/dell metadata: keep the most conservative (small overlap, small bin).
        fsky_eff = float(min(t[1] for t in triples))
        dell_eff = int(min(t[2] for t in triples))
        entries.append(
            JointDataEntry(
                experiment=experiment,
                spectrum=spec,
                ell=int(ell),
                dell=int(dell_eff),
                fsky=float(fsky_eff),
                var=float(v_eff),
                n_channels=len(triples),
            )
        )
    return entries


def build_joint_experiment_covariance(
    experiments: dict[str, Sequence[Any]] | Sequence[tuple[str, Sequence[Any]]],
    *,
    spectra: Sequence[str] = ("TT", "TE", "EE"),
    ell: Sequence[int] | None = None,
    f_sky_overlap: Any = None,
    fiducial_cls: dict[str, Any] | None = None,
    include_cross_experiment_cov: bool = True,
) -> dict[str, Any]:
    """
    Build a joint CMB spectra data vector and its joint covariance, including
    cross-experiment *cosmic-variance* covariance blocks on overlapping sky.

    Data-vector elements are:
        (experiment, spectrum, ell_bin)

    Notes / approximation:
    - Cross-experiment covariance is only filled when entries share the same
      integer ell bin. If experiments use different binning, this is a
      conservative approximation (it may under-couple nearby bins).
    - Cross-experiment noise is assumed zero (independent noise realizations).
    """
    import numpy as np

    if fiducial_cls is None:
        raise ValueError("fiducial_cls is required to build cross-experiment cosmic-variance blocks.")

    items: Iterable[tuple[str, Sequence[Any]]]
    if isinstance(experiments, dict):
        items = experiments.items()
    else:
        items = experiments

    all_entries: list[JointDataEntry] = []
    for exp_name, bands in items:
        all_entries.extend(
            _entries_from_bands(
                experiment=str(exp_name),
                bands=list(bands),
                spectra=list(spectra),
                fiducial_cls=fiducial_cls,
                f_sky_overlap=f_sky_overlap,
                compress_channels=True,
            )
        )

    if ell is not None:
        wanted_ell = {int(e) for e in ell}
        all_entries = [e for e in all_entries if int(e.ell) in wanted_ell]

    # Group indices by ell for efficient block building/inversion.
    index_by_ell: dict[int, list[int]] = {}
    for i, ent in enumerate(all_entries):
        index_by_ell.setdefault(int(ent.ell), []).append(i)

    cov_by_ell: dict[int, np.ndarray] = {}
    for ell_i, idxs in sorted(index_by_ell.items()):
        n = len(idxs)
        cov = np.zeros((n, n), dtype=float)
        for a, ia in enumerate(idxs):
            ea = all_entries[ia]
            cov[a, a] = float(ea.var)
            if not include_cross_experiment_cov:
                continue
            for b in range(a + 1, n):
                ib = idxs[b]
                eb = all_entries[ib]
                if ea.experiment == eb.experiment:
                    continue
                fsky_ab = _resolve_fsky_overlap(
                    f_sky_overlap,
                    exp_a=ea.experiment,
                    exp_b=eb.experiment,
                    fsky_a=float(ea.fsky),
                    fsky_b=float(eb.fsky),
                    ell=int(ell_i),
                )
                dell_ab = int(min(int(ea.dell), int(eb.dell)))
                cov_ab = _gaussian_cov_cl(
                    fiducial_cls=fiducial_cls,
                    spec1=str(ea.spectrum),
                    spec2=str(eb.spectrum),
                    ell=int(ell_i),
                    fsky=float(fsky_ab),
                    dell=int(dell_ab),
                )
                cov[a, b] = cov_ab
                cov[b, a] = cov_ab
        cov_by_ell[int(ell_i)] = cov

    return {
        "entries": all_entries,
        "index_by_ell": index_by_ell,
        "cov_by_ell": cov_by_ell,
    }


def fisher_from_joint_covariance(
    *,
    theta0: dict[str, float],
    corr_dict: dict[str, float],
    iso_mode: str | None,
    compute_cls: Callable,
    experiments: dict[str, Sequence[Any]] | Sequence[tuple[str, Sequence[Any]]],
    param_list: Sequence[str],
    steps: dict[str, float],
    scaled10_params: set[str],
    spectra: Sequence[str] = ("TT", "TE", "EE"),
    use_iso_signal_in_cov: bool = True,
    f_sky_overlap: Any = None,
    include_cross_experiment_cov: bool = True,
    cond_warn: float = 1e12,
    rcond_pinv: float = 1e-12,
    verbose: bool = False,
):
    """
    Build one Fisher matrix from a single joint spectra data vector and its
    joint covariance (with cross-experiment cosmic variance).

    Returns fisher_multi.FisherResult, so downstream plotting works unchanged.
    """
    import numpy as np

    from cosmocast_makelik.multi_freq_liq import fisher_multi

    theta0 = _ensure_theta0_has_standard_lcdm_params(theta0=theta0, param_list=param_list)
    theta_fid = {**theta0, **corr_dict}

    use_standard = iso_mode is None and (
        "ln10A_s" in theta_fid or "A_s" in theta_fid or "n_s" in theta_fid
    )

    # lmax: cover all ell bins present in the joint vector.
    # (We build the joint covariance after we have fiducial C_ell arrays.)
    if use_standard:
        _provider = _cls_provider_with_standard_lcdm_amplitude(compute_cls, iso_mode=None)

        def _cls_indexed(*, lmax: int, iso: str | None, th: dict[str, float]) -> dict[str, Any]:
            if iso is not None:
                raise ValueError("Standard ΛCDM wrapper only supports iso=None.")
            return _provider(lmax=int(lmax), **th)

    else:

        def _cls_indexed(*, lmax: int, iso: str | None, th: dict[str, float]) -> dict[str, Any]:
            return _compute_cls_indexed(compute_cls, lmax=int(lmax), iso_mode=iso, **th)

    # Choose fiducial for covariance normalization: iso signal optional.
    cov_iso_mode = iso_mode if (use_iso_signal_in_cov or iso_mode is None) else None
    if use_standard:
        cov_iso_mode = None
    # Conservative: compute theory up to at least 4000 if present in SO, else max ell in bands.
    max_ell_guess = 0
    for _, bands in (experiments.items() if isinstance(experiments, dict) else experiments):
        for b in bands:
            try:
                max_ell_guess = max(max_ell_guess, int(np.max(np.asarray(getattr(b, "ell"), dtype=int))))
            except Exception:
                pass
    ell_max_theory = max(50, int(max_ell_guess))

    cls0 = _cls_indexed(lmax=ell_max_theory, iso=cov_iso_mode, th=theta_fid)

    joint = build_joint_experiment_covariance(
        experiments,
        spectra=spectra,
        f_sky_overlap=f_sky_overlap,
        fiducial_cls=cls0,
        include_cross_experiment_cov=include_cross_experiment_cov,
    )
    entries: list[JointDataEntry] = joint["entries"]
    cov_by_ell: dict[int, np.ndarray] = joint["cov_by_ell"]
    index_by_ell: dict[int, list[int]] = joint["index_by_ell"]

    if not entries:
        raise ValueError("Joint data vector is empty; check experiment ell ranges and spectra selection.")

    nvec = len(entries)
    npar = len(param_list)

    # Derivative matrix: D[i, k] = d C(spec_k, ell_k) / d theta_i
    D = np.zeros((npar, nvec), dtype=float)

    def _perturb(th0: dict[str, float], param: str, delta: float) -> dict[str, float]:
        th = dict(th0)
        if param in scaled10_params:
            th[param] = (1e10 * th0[param] + delta) * 1e-10
        else:
            th[param] = th0[param] + delta
        return th

    for ip, p in enumerate(param_list):
        if p not in steps:
            raise KeyError(f"No step size provided for parameter '{p}'")
        step = float(steps[p])
        th_hi = _perturb(theta_fid, p, +step)
        th_lo = _perturb(theta_fid, p, -step)
        cls_hi = _cls_indexed(lmax=ell_max_theory, iso=iso_mode, th=th_hi)
        cls_lo = _cls_indexed(lmax=ell_max_theory, iso=iso_mode, th=th_lo)
        for k, ent in enumerate(entries):
            spec = str(ent.spectrum)
            ell_k = int(ent.ell)
            D[ip, k] = (float(np.asarray(cls_hi[spec])[ell_k]) - float(np.asarray(cls_lo[spec])[ell_k])) / (2.0 * step)

    # Accumulate Fisher per-ell block for stability and diagnostics.
    F = np.zeros((npar, npar), dtype=float)
    conds: list[float] = []
    n_pinv = 0
    for ell_i, idxs in sorted(index_by_ell.items()):
        cov = cov_by_ell[int(ell_i)]
        # Condition number (best-effort; can be inf/nan for singular matrices)
        try:
            cond = float(np.linalg.cond(cov))
        except Exception:
            cond = float("inf")
        conds.append(cond)

        use_pinv = (not np.isfinite(cond)) or (cond > float(cond_warn))
        if not use_pinv:
            try:
                cov_inv = np.linalg.inv(cov)
            except Exception:
                use_pinv = True
        if use_pinv:
            n_pinv += 1
            cov_inv = np.linalg.pinv(cov, rcond=float(rcond_pinv))

        d = D[:, idxs]  # (npar, nblock)
        F += d @ cov_inv @ d.T

        if verbose and use_pinv:
            print(f"[joint_cov] ell={ell_i}: used pinv (cond={cond:.3g}, block_n={cov.shape[0]}).")

    F = 0.5 * (F + F.T)
    Cov_params = np.linalg.pinv(F)
    sigma = np.sqrt(np.diag(Cov_params))

    result = fisher_multi.FisherResult(
        F=F,
        Cov_params=Cov_params,
        sigma=sigma,
        dC=[D[i].copy() for i in range(npar)],
        bands=[],
        param_list=list(param_list),
        metadata={
            "combination_mode": "joint_cov",
            "use_iso_signal_in_cov": bool(use_iso_signal_in_cov),
            "iso_mode": iso_mode,
            "n_joint_entries": int(nvec),
            "n_ell_blocks": int(len(index_by_ell)),
            "cond_max": float(np.nanmax(conds)) if conds else float("nan"),
            "cond_median": float(np.nanmedian(conds)) if conds else float("nan"),
            "n_blocks_pinv": int(n_pinv),
            "rcond_pinv": float(rcond_pinv),
        },
    )
    return result


def _infer_noise_auto_from_var(
    *,
    var: float,
    c_ell: float,
    ell: int,
    dell: int,
    fsky: float,
) -> float:
    """
    Infer an effective noise N_ell from an auto-spectrum variance Var(Ĉ_ell)
    assuming the Knox-diagonal form:

        Var = 2/((2ell+1) fsky dell) * (C_ell + N_ell)^2

    This is used for ell-by-ell selection. It is an approximation for Planck
    PlikLite variances but gives a consistent "effective" noise for comparing
    experiments on the same ell bins.
    """
    import numpy as np

    pref = (2.0 * float(ell) + 1.0) * float(fsky) * float(dell) / 2.0
    if pref <= 0 or var <= 0:
        return 0.0
    n = np.sqrt(float(var) * pref) - float(c_ell)
    if not np.isfinite(n):
        return 0.0
    return float(max(0.0, n))


def _noise_dict_from_bands_auto(
    *,
    bands: Sequence[Any],
    spectrum: str,
    fid_c_ell: dict[str, Any],
    fsky_override: float | None = None,
) -> tuple[dict[int, float], dict[int, int], dict[int, float]]:
    """
    Build (N_ell, dell_ell, fsky_ell) dictionaries for an auto spectrum from bands,
    taking the minimum inferred noise across channels at each ell.
    """
    import numpy as np

    n_by_ell: dict[int, float] = {}
    dell_by_ell: dict[int, int] = {}
    fsky_by_ell: dict[int, float] = {}
    for b in bands:
        if str(getattr(b, "cell_type", "")) != spectrum:
            continue
        ell_arr = np.asarray(getattr(b, "ell"), dtype=float)
        var_arr = np.asarray(getattr(b, "cov"), dtype=float)
        dell = int(getattr(b, "dell", 1))
        fsky = float(getattr(b, "fsky", 1.0)) if fsky_override is None else float(fsky_override)
        for e, v in zip(ell_arr, var_arr):
            ell = int(np.round(float(e)))
            c = float(np.asarray(fid_c_ell[spectrum])[ell])
            n = _infer_noise_auto_from_var(var=float(v), c_ell=c, ell=ell, dell=dell, fsky=fsky)
            if (ell not in n_by_ell) or (n < n_by_ell[ell]):
                n_by_ell[ell] = float(n)
                dell_by_ell[ell] = int(dell)
                fsky_by_ell[ell] = float(fsky)
    return n_by_ell, dell_by_ell, fsky_by_ell


def fisher_from_ell_by_ell_effective_experiment(
    *,
    theta0: dict[str, float],
    corr_dict: dict[str, float],
    iso_mode: str | None,
    compute_cls: Callable,
    planck_bands: Sequence[Any],
    so_bands: Sequence[Any],
    litebird_bands: Sequence[Any],
    param_list: Sequence[str],
    steps: dict[str, float],
    scaled10_params: set[str],
    use_iso_signal_in_cov: bool = True,
    ell_by_ell_policy: str = "min_noise",
    verbose: bool = False,
):
    """
    Teo-style effective-experiment combination:
      - infer per-experiment auto-spectrum noise curves from band variances
      - at each ell and spectrum, select the experiment with minimum noise
      - build a single effective covariance from (C_ell + N_eff) and per-ell f_sky
      - compute one Fisher matrix from the effective bands.

    Experiment/f_sky conventions (per requirement):
      - Planck:   0.7
      - LiteBIRD: 0.7
      - SO SAT:   0.1
      - SO LAT:   0.4   (TT from LAT, EE from LAT_pol)
    """
    import numpy as np

    if ell_by_ell_policy != "min_noise":
        raise ValueError(f"Unsupported ELL_BY_ELL_POLICY={ell_by_ell_policy!r} (expected 'min_noise').")

    from cosmocast_makelik.multi_freq_liq import fisher_multi
    from cosmocast_makelik.multi_freq_liq.fisher_multi import SpectrumBand

    theta0 = _ensure_theta0_has_standard_lcdm_params(theta0=theta0, param_list=param_list)
    theta_fid = {**theta0, **corr_dict}

    # Fiducial spectra for covariance normalization
    cov_iso_mode = iso_mode if (use_iso_signal_in_cov or iso_mode is None) else None
    use_standard = iso_mode is None and (
        "ln10A_s" in theta_fid or "A_s" in theta_fid or "n_s" in theta_fid
    )
    if use_standard:
        cov_iso_mode = None

    # Determine lmax needed from available band ell bins
    def _max_ell(bands: Sequence[Any]) -> int:
        m = 0
        for b in bands:
            try:
                m = max(m, int(np.max(np.asarray(getattr(b, "ell"), dtype=int))))
            except Exception:
                pass
        return m

    ell_max_theory = max(50, _max_ell(planck_bands), _max_ell(so_bands), _max_ell(litebird_bands))
    if use_standard:
        cls_cov = _cls_provider_with_standard_lcdm_amplitude(compute_cls, iso_mode=None)(
            lmax=int(ell_max_theory),
            **theta_fid,
        )
    else:
        cls_cov = _compute_cls_indexed(compute_cls, lmax=int(ell_max_theory), iso_mode=cov_iso_mode, **theta_fid)

    # Split SO bands into SAT vs LAT/LAT_pol
    so_sat = [b for b in so_bands if str(getattr(b, "exp_key", "")) == "SAT"]
    so_lat_tt = [b for b in so_bands if str(getattr(b, "exp_key", "")) == "LAT"]
    so_lat_ee = [b for b in so_bands if str(getattr(b, "exp_key", "")) == "LAT_pol"]

    # Build noise dictionaries for TT/EE per "effective experiment"
    n_tt_planck, dell_tt_planck, fsky_tt_planck = _noise_dict_from_bands_auto(
        bands=planck_bands, spectrum="TT", fid_c_ell=cls_cov, fsky_override=0.7
    )
    n_ee_planck, dell_ee_planck, fsky_ee_planck = _noise_dict_from_bands_auto(
        bands=planck_bands, spectrum="EE", fid_c_ell=cls_cov, fsky_override=0.7
    )
    n_tt_lb, dell_tt_lb, fsky_tt_lb = _noise_dict_from_bands_auto(
        bands=litebird_bands, spectrum="TT", fid_c_ell=cls_cov, fsky_override=0.7
    )
    n_ee_lb, dell_ee_lb, fsky_ee_lb = _noise_dict_from_bands_auto(
        bands=litebird_bands, spectrum="EE", fid_c_ell=cls_cov, fsky_override=0.7
    )
    n_tt_sat, dell_tt_sat, fsky_tt_sat = _noise_dict_from_bands_auto(
        bands=so_sat, spectrum="TT", fid_c_ell=cls_cov, fsky_override=0.1
    )
    n_ee_sat, dell_ee_sat, fsky_ee_sat = _noise_dict_from_bands_auto(
        bands=so_sat, spectrum="EE", fid_c_ell=cls_cov, fsky_override=0.1
    )
    n_tt_lat, dell_tt_lat, fsky_tt_lat = _noise_dict_from_bands_auto(
        bands=so_lat_tt, spectrum="TT", fid_c_ell=cls_cov, fsky_override=0.4
    )
    n_ee_lat, dell_ee_lat, fsky_ee_lat = _noise_dict_from_bands_auto(
        bands=so_lat_ee, spectrum="EE", fid_c_ell=cls_cov, fsky_override=0.4
    )

    sources = {
        "Planck": {"TT": (n_tt_planck, dell_tt_planck, fsky_tt_planck), "EE": (n_ee_planck, dell_ee_planck, fsky_ee_planck)},
        "LiteBIRD": {"TT": (n_tt_lb, dell_tt_lb, fsky_tt_lb), "EE": (n_ee_lb, dell_ee_lb, fsky_ee_lb)},
        "SO SAT": {"TT": (n_tt_sat, dell_tt_sat, fsky_tt_sat), "EE": (n_ee_sat, dell_ee_sat, fsky_ee_sat)},
        "SO LAT": {"TT": (n_tt_lat, dell_tt_lat, fsky_tt_lat), "EE": (n_ee_lat, dell_ee_lat, fsky_ee_lat)},
    }

    # TE availability: only include sources that actually have TE bands.
    has_te = {
        "Planck": any(str(getattr(b, "cell_type", "")) == "TE" for b in planck_bands),
        "LiteBIRD": any(str(getattr(b, "cell_type", "")) == "TE" for b in litebird_bands),
        "SO SAT": any(str(getattr(b, "cell_type", "")) == "TE" for b in so_sat),
        "SO LAT": any(str(getattr(b, "cell_type", "")) == "TE" for b in so_bands),
    }

    def _select_auto(spec: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        # Returns (ell, var, counts) for the effective band of this auto-spectrum.
        # Also records counts per source.
        all_ell = set()
        for src in sources.values():
            all_ell.update(src.get(spec, ({}, {}, {}))[0].keys())
        ell_sorted = np.array(sorted(all_ell), dtype=int)

        counts = {k: 0 for k in sources.keys()}
        var = np.zeros_like(ell_sorted, dtype=float)
        for i, l in enumerate(ell_sorted):
            best_src = None
            best_n = None
            best_dell = None
            best_fsky = None
            for name, src in sources.items():
                n_map, dell_map, fsky_map = src.get(spec, ({}, {}, {}))
                if l not in n_map:
                    continue
                n = float(n_map[l])
                if (best_n is None) or (n < best_n):
                    best_n = n
                    best_src = name
                    best_dell = int(dell_map.get(l, 1))
                    best_fsky = float(fsky_map.get(l, 1.0))
            if best_src is None:
                var[i] = np.nan
                continue
            counts[best_src] += 1
            c = float(np.asarray(cls_cov[spec])[int(l)])
            var[i] = 2.0 / ((2.0 * float(l) + 1.0) * float(best_fsky) * float(best_dell)) * (c + float(best_n)) ** 2
        mask = np.isfinite(var)
        return ell_sorted[mask], var[mask], counts

    def _select_te() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        # Choose experiment by min sqrt(N_TT*N_EE), but only among sources with TE coverage.
        all_ell = set()
        for name, src in sources.items():
            if not has_te.get(name, False):
                continue
            all_ell.update(src["TT"][0].keys())
            all_ell.update(src["EE"][0].keys())
        ell_sorted = np.array(sorted(all_ell), dtype=int)

        counts = {k: 0 for k in sources.keys()}
        var = np.zeros_like(ell_sorted, dtype=float)
        for i, l in enumerate(ell_sorted):
            best_src = None
            best_proxy = None
            best_dell = None
            best_fsky = None
            best_ntt = None
            best_nee = None
            for name, src in sources.items():
                if not has_te.get(name, False):
                    continue
                ntt_map, dell_tt, fsky_tt = src["TT"]
                nee_map, dell_ee, fsky_ee = src["EE"]
                if (l not in ntt_map) or (l not in nee_map):
                    continue
                ntt = float(ntt_map[l])
                nee = float(nee_map[l])
                proxy = float(np.sqrt(max(0.0, ntt) * max(0.0, nee)))
                if (best_proxy is None) or (proxy < best_proxy):
                    best_proxy = proxy
                    best_src = name
                    best_ntt = ntt
                    best_nee = nee
                    best_dell = int(min(int(dell_tt.get(l, 1)), int(dell_ee.get(l, 1))))
                    best_fsky = float(min(float(fsky_tt.get(l, 1.0)), float(fsky_ee.get(l, 1.0))))
            if best_src is None:
                var[i] = np.nan
                continue
            counts[best_src] += 1
            c_te = float(np.asarray(cls_cov["TE"])[int(l)])
            c_tt = float(np.asarray(cls_cov["TT"])[int(l)])
            c_ee = float(np.asarray(cls_cov["EE"])[int(l)])
            var[i] = (
                1.0
                / ((2.0 * float(l) + 1.0) * float(best_fsky) * float(best_dell))
                * (c_te**2 + (c_tt + float(best_ntt)) * (c_ee + float(best_nee)))
            )
        mask = np.isfinite(var)
        return ell_sorted[mask], var[mask], counts

    ell_tt, cov_tt, counts_tt = _select_auto("TT")
    ell_ee, cov_ee, counts_ee = _select_auto("EE")
    ell_te, cov_te, counts_te = _select_te()

    bands_eff = [
        SpectrumBand(
            exp_key="ELL_BY_ELL",
            channel="ELL_BY_ELL_TT",
            cell_type="TT",
            ell=np.asarray(ell_tt, dtype=int),
            dell=1,
            fsky=1.0,
            cov=np.asarray(cov_tt, dtype=float),
        ),
        SpectrumBand(
            exp_key="ELL_BY_ELL",
            channel="ELL_BY_ELL_TE",
            cell_type="TE",
            ell=np.asarray(ell_te, dtype=int),
            dell=1,
            fsky=1.0,
            cov=np.asarray(cov_te, dtype=float),
        ),
        SpectrumBand(
            exp_key="ELL_BY_ELL",
            channel="ELL_BY_ELL_EE",
            cell_type="EE",
            ell=np.asarray(ell_ee, dtype=int),
            dell=1,
            fsky=1.0,
            cov=np.asarray(cov_ee, dtype=float),
        ),
    ]

    # Derivatives: always evaluated at selected iso_mode (even if covariance uses adiabatic fiducial)
    if use_standard:
        _cls_provider = _cls_provider_with_standard_lcdm_amplitude(compute_cls, iso_mode=None)
    else:

        def _cls_provider(*, lmax: int = ell_max_theory, **th):
            return _compute_cls_indexed(compute_cls, lmax=int(lmax), iso_mode=iso_mode, **th)

    fisher = fisher_multi.fisher_forecast(
        theta0=theta_fid,
        param_list=list(param_list),
        bands=bands_eff,
        compute_cls=_cls_provider,
        steps=steps,
        scaled_params=scaled10_params,
        ell_max=int(ell_max_theory),
        use_pinv=True,
    )
    fisher.metadata = {
        "combination_mode": "ell_by_ell",
        "ell_by_ell_policy": ell_by_ell_policy,
        "use_iso_signal_in_cov": bool(use_iso_signal_in_cov),
        "iso_mode": iso_mode,
        "counts": {"TT": counts_tt, "TE": counts_te, "EE": counts_ee},
        "n_ell": {"TT": int(len(ell_tt)), "TE": int(len(ell_te)), "EE": int(len(ell_ee))},
    }

    if verbose:
        print("[ell_by_ell] counts TT:", counts_tt)
        print("[ell_by_ell] counts TE:", counts_te)
        print("[ell_by_ell] counts EE:", counts_ee)
    return fisher


def triangle_overlay(
    fishers_named: list[tuple[Any, str]],
    *,
    theta0: dict,
    subset: list[str],
    scaled10_params: set[str],
    save_path: str | Path,
    label_map: dict[str, str] | None = None,
    skip_existing: bool = True,
) -> None:
    """
    Overlay multiple Fisher constraints as filled contours (like in the original script).
    """
    save_path = Path(save_path)
    if _maybe_skip(save_path, skip_existing):
        return

    try:
        from getdist.gaussian_mixtures import GaussianND
        from getdist import plots
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("getdist is required for triangle overlays") from e

    label_map = label_map or LATEX_LABELS
    labels = [label_map.get(p, p) for p in subset]

    def _centers(subset_now):
        return np.array(
            [1e10 * theta0[p] if p in scaled10_params else theta0[p] for p in subset_now],
            dtype=float,
        )

    gaussians = []
    legend_labels = []
    for fisher, name in fishers_named:
        idx = [fisher.param_list.index(p) for p in subset]
        cov_sub = fisher.Cov_params[np.ix_(idx, idx)]
        gaussians.append(GaussianND(_centers(subset), cov_sub, names=subset, labels=labels, label=name))
        legend_labels.append(name)

    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 14
    g.settings.axes_labelsize = 16
    g.settings.legend_fontsize = 14
    g.settings.lab_fontsize = 16
    g.triangle_plot(gaussians, subset, filled=True, legend_labels=legend_labels)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close("all")


def compare_and_save_constraints(
    so_fisher_all: dict,
    muci: dict,
    theta_full: dict,
    corr_plot: list[str],
    iso_plot: list[str | None],
    year_tag: str = "10yr",
    out_year_tag: str | None = None,
    out_base: str | Path = "images/unified_constraints",
    include_corr_subdir: bool = True,
    combination_mode: str = "fisher_sum",
    use_iso_signal_in_cov: bool = True,
    compute_cls: Callable | None = None,
    f_sky_overlap: Any = None,
    ell_by_ell_policy: str = "min_noise",
    planck_likelihood_mode: str = "pliklite_lowT",
    planck_tau_prior_sigma: float = 0.007,
    scaled10_params: set[str] | None = None,
    label_map: dict[str, str] | None = None,
    skip_existing: bool = True,
    write_tables: bool = True,
    write_corr: bool = True,
    write_triangles: bool = True,
    write_overlays: bool = True,
) -> None:
    """
    Save Planck-only, SO-only, LiteBIRD-only and unified constraints.

    The "unified" constraint uses the maximum-likelihood combination for
    independent Gaussian datasets, which is equivalent to adding Fisher
    matrices: F_total = F1 + F2 + ...

    If combination_mode=="joint_cov", build one joint spectra data vector and one
    joint covariance matrix (including cross-experiment cosmic-variance blocks),
    then compute one Fisher matrix from that joint covariance.
    """
    scaled10_params = scaled10_params or {"P_RR_1", "P_RR_2", "P_II_1", "P_II_2"}
    out_base = Path(out_base)
    os.makedirs(out_base, exist_ok=True)
    label_map = label_map or LATEX_LABELS
    if out_year_tag is None:
        # Default tagging keeps the legacy '10yr' for the original workflow, but
        # adds suffixes when options change.
        out_year_tag = make_run_tag(
            year_tag=year_tag,
            combination_mode=combination_mode,
            use_iso_signal_in_cov=use_iso_signal_in_cov,
            planck_likelihood_mode=planck_likelihood_mode,
            planck_tau_prior_sigma=planck_tau_prior_sigma,
            ell_by_ell_policy=ell_by_ell_policy,
            # This helper is used mainly for directory naming; if callers use the
            # standard ΛCDM option, they should pass an explicit out_year_tag or
            # compute a RUN_TAG that encodes it.
        )

    iso_plot = list(iso_plot)
    run_adiabatic_once = any(iso is None for iso in iso_plot)
    iso_plot_nonadiabatic = [iso for iso in iso_plot if iso is not None]

    def _run_case(*, corr: str, iso: str | None, include_corr: bool) -> None:
        theta0 = theta_full[corr]["theta0"]
        corr_dict = theta_full[corr].get("corr", {}) or {}
        iso_key = _iso_tag(iso)
        pk = so_fisher_all[corr]["PK_Lite"][iso][year_tag]
        so = so_fisher_all[corr]["SO"][iso][year_tag]
        lb = muci[corr]["LB"][iso][year_tag]

        if iso is None:
            # "No isocurvature" forecast: restrict to adiabatic parameters even if the
            # stored Fisher includes iso params (older pickles).
            keep = ["omega_b", "omega_cdm", "h", "tau_reio", "P_RR_1", "P_RR_2"]
            if any(p.startswith("P_II_") for p in pk.param_list):
                pk = _restrict_fisher_params(pk, keep_params=keep)
            if any(p.startswith("P_II_") for p in so.param_list):
                so = _restrict_fisher_params(so, keep_params=keep)
            if any(p.startswith("P_II_") for p in lb.param_list):
                lb = _restrict_fisher_params(lb, keep_params=keep)

        # Optional Planck lowE approximation (tau prior).
        if planck_likelihood_mode == "TTTEEE_lowE":
            print(
                f"[compare_and_save_constraints] WARNING: PLANCK_LIKELIHOOD_MODE='TTTEEE_lowE' "
                f"implemented via tau prior sigma={planck_tau_prior_sigma} (approximation; no true lowE bandpowers)."
            )
        pk = apply_planck_lowE_tau_prior(
            pk,
            planck_likelihood_mode=planck_likelihood_mode,
            tau_prior_sigma=planck_tau_prior_sigma,
        )

        # If the loaded/recomputed pickles use standard ΛCDM params (ln10A_s/n_s),
        # build a compatible fiducial dict for theory calls in joint_cov/ell_by_ell.
        theta0_unified = dict(theta0)
        corr_dict_unified = dict(corr_dict)
        if iso is None and ("ln10A_s" in pk.param_list or "n_s" in pk.param_list):
            theta0_unified = _theta0_standard_lcdm(theta0_base=theta0)
            corr_dict_unified = {}
            print(
                "[compare_and_save_constraints] Using standard ΛCDM fiducial (theta0 has ln10A_s/n_s) "
                f"for iso={iso_key}."
            )

        if combination_mode == "fisher_sum":
            unified = pk.combine(so).combine(lb)
        elif combination_mode == "joint_cov":
            if compute_cls is None:
                raise ValueError("compute_cls is required when combination_mode=='joint_cov'.")
            param_list = list(pk.param_list)
            steps = _default_steps_for_param_list(param_list)
            scaled10 = _scaled10_params_for_param_list(param_list)
            unified = fisher_from_joint_covariance(
                theta0=dict(theta0_unified),
                corr_dict=dict(corr_dict_unified),
                iso_mode=iso,
                compute_cls=compute_cls,
                experiments={"Planck": pk.bands, "SO": so.bands, "LiteBIRD": lb.bands},
                param_list=param_list,
                steps=steps,
                scaled10_params=scaled10,
                spectra=("TT", "TE", "EE"),
                use_iso_signal_in_cov=bool(use_iso_signal_in_cov),
                f_sky_overlap=f_sky_overlap,
                include_cross_experiment_cov=True,
                verbose=False,
            )
            print(
                f"[compare_and_save_constraints] joint_cov: corr={corr} iso={iso_key} "
                f"n_entries={unified.metadata.get('n_joint_entries')} "
                f"cond_max={unified.metadata.get('cond_max'):.3g} "
                f"cond_median={unified.metadata.get('cond_median'):.3g} "
                f"blocks_pinv={unified.metadata.get('n_blocks_pinv')}"
            )
        elif combination_mode == "ell_by_ell":
            if compute_cls is None:
                raise ValueError("compute_cls is required when combination_mode=='ell_by_ell'.")
            param_list = list(pk.param_list)
            steps = _default_steps_for_param_list(param_list)
            scaled10 = _scaled10_params_for_param_list(param_list)
            unified = fisher_from_ell_by_ell_effective_experiment(
                theta0=dict(theta0_unified),
                corr_dict=dict(corr_dict_unified),
                iso_mode=iso,
                compute_cls=compute_cls,
                planck_bands=pk.bands,
                so_bands=so.bands,
                litebird_bands=lb.bands,
                param_list=param_list,
                steps=steps,
                scaled10_params=scaled10,
                use_iso_signal_in_cov=bool(use_iso_signal_in_cov),
                ell_by_ell_policy=str(ell_by_ell_policy),
                verbose=False,
            )
            counts = (getattr(unified, "metadata", {}) or {}).get("counts", {})
            if counts:
                print(f"[compare_and_save_constraints] combination_mode=ell_by_ell policy={ell_by_ell_policy}")
                print("[compare_and_save_constraints] ell_by_ell f_sky mapping active (per-ell).")
                print(f"[compare_and_save_constraints] corr={corr} iso={iso_key}")
                print(f"[compare_and_save_constraints] TT assignments: {counts.get('TT', {})}")
                print(f"[compare_and_save_constraints] TE assignments: {counts.get('TE', {})}")
                print(f"[compare_and_save_constraints] EE assignments: {counts.get('EE', {})}")
        else:
            raise ValueError(
                f"Unknown combination_mode={combination_mode!r} (expected 'fisher_sum', 'joint_cov', or 'ell_by_ell')."
            )

        out_dir = out_base / out_year_tag
        if include_corr_subdir and include_corr:
            out_dir = out_dir / corr
        out_dir = out_dir / iso_key
        out_dir.mkdir(parents=True, exist_ok=True)

        save_fisher_plots(
            pk,
            theta0=theta0,
            scaled10_params=scaled10_params,
            out_dir=out_dir,
            stem=f"planck_{year_tag}",
            label_map=label_map,
            skip_existing=skip_existing,
            write_table=write_tables,
            write_corr=write_corr,
            write_triangle=write_triangles,
        )
        save_fisher_plots(
            so,
            theta0=theta0,
            scaled10_params=scaled10_params,
            out_dir=out_dir,
            stem=f"so_{year_tag}",
            label_map=label_map,
            skip_existing=skip_existing,
            write_table=write_tables,
            write_corr=write_corr,
            write_triangle=write_triangles,
        )
        save_fisher_plots(
            lb,
            theta0=theta0,
            scaled10_params=scaled10_params,
            out_dir=out_dir,
            stem=f"litebird_{year_tag}",
            label_map=label_map,
            skip_existing=skip_existing,
            write_table=write_tables,
            write_corr=write_corr,
            write_triangle=write_triangles,
        )
        save_fisher_plots(
            unified,
            theta0=theta0,
            scaled10_params=scaled10_params,
            out_dir=out_dir,
            stem=f"unified_{year_tag}",
            label_map=label_map,
            skip_existing=skip_existing,
            write_table=write_tables,
            write_corr=write_corr,
            write_triangle=write_triangles,
        )

        if write_overlays:
            regular_params = [p for p in ["omega_b", "omega_cdm", "h", "tau_reio"] if p in unified.param_list]
            scaled_params = [p for p in ["P_RR_1", "P_RR_2", "P_II_1", "P_II_2"] if p in unified.param_list]
            # Only SO scales with years; PlanckLite and LiteBIRD are not time-scaled in this workflow.
            overlay_fishers = [(pk, "Planck"), (so, f"SO {year_tag}"), (lb, "LiteBIRD")]
            triangle_overlay(
                overlay_fishers,
                theta0=theta0,
                subset=regular_params,
                scaled10_params=set(),
                save_path=out_dir / f"overlay_regular_{year_tag}.pdf",
                label_map=label_map,
                skip_existing=skip_existing,
            )
            triangle_overlay(
                overlay_fishers,
                theta0=theta0,
                subset=scaled_params,
                scaled10_params=scaled10_params,
                save_path=out_dir / f"overlay_scaled_{year_tag}.pdf",
                label_map=label_map,
                skip_existing=skip_existing,
            )

            # Second overlay set ("_summed"):
            #   Planck-only vs (Planck+SO) vs (Planck+SO+LiteBIRD).
            # Saved with suffix "_summed" as requested.
            if combination_mode == "fisher_sum":
                pk_so = pk.combine(so)
                pk_so_lb = unified
            elif combination_mode == "joint_cov":
                # Recompute the unified Fisher for subset experiment sets using the
                # same joint-covariance logic (shared cosmic variance, independent noise).
                if compute_cls is None:
                    raise ValueError("compute_cls is required for joint_cov overlays.")
                param_list = list(pk.param_list)
                steps = _default_steps_for_param_list(param_list)
                scaled10 = _scaled10_params_for_param_list(param_list)
                pk_so = fisher_from_joint_covariance(
                    theta0=dict(theta0_unified),
                    corr_dict=dict(corr_dict_unified),
                    iso_mode=iso,
                    compute_cls=compute_cls,
                    experiments={"Planck": pk.bands, "SO": so.bands},
                    param_list=param_list,
                    steps=steps,
                    scaled10_params=scaled10,
                    spectra=("TT", "TE", "EE"),
                    use_iso_signal_in_cov=bool(use_iso_signal_in_cov),
                    f_sky_overlap=f_sky_overlap,
                    include_cross_experiment_cov=True,
                    verbose=False,
                )
                pk_so_lb = unified
            elif combination_mode == "ell_by_ell":
                if compute_cls is None:
                    raise ValueError("compute_cls is required for ell_by_ell overlays.")
                param_list = list(pk.param_list)
                steps = _default_steps_for_param_list(param_list)
                scaled10 = _scaled10_params_for_param_list(param_list)
                pk_so = fisher_from_ell_by_ell_effective_experiment(
                    theta0=dict(theta0_unified),
                    corr_dict=dict(corr_dict_unified),
                    iso_mode=iso,
                    compute_cls=compute_cls,
                    planck_bands=pk.bands,
                    so_bands=so.bands,
                    litebird_bands=[],
                    param_list=param_list,
                    steps=steps,
                    scaled10_params=scaled10,
                    use_iso_signal_in_cov=bool(use_iso_signal_in_cov),
                    ell_by_ell_policy=str(ell_by_ell_policy),
                    verbose=False,
                )
                pk_so_lb = unified
            else:
                raise ValueError(f"Unknown combination_mode={combination_mode!r}")

            overlay_fishers_summed = [
                (pk, "Planck"),
                (pk_so, f"Planck + SO {year_tag}"),
                (pk_so_lb, f"Planck + SO {year_tag} + LiteBIRD"),
            ]
            triangle_overlay(
                overlay_fishers_summed,
                theta0=theta0,
                subset=regular_params,
                scaled10_params=set(),
                save_path=out_dir / f"overlay_regular_{year_tag}_summed.pdf",
                label_map=label_map,
                skip_existing=skip_existing,
            )
            triangle_overlay(
                overlay_fishers_summed,
                theta0=theta0,
                subset=scaled_params,
                scaled10_params=scaled10_params,
                save_path=out_dir / f"overlay_scaled_{year_tag}_summed.pdf",
                label_map=label_map,
                skip_existing=skip_existing,
            )

        sig_path = out_dir / "sigmas.txt"
        if not _maybe_skip(sig_path, skip_existing):
            with sig_path.open("w") as f:
                f.write(f"corr={corr} iso={iso_key} year={year_tag}\n")
                f.write("param  planck  so  litebird  unified\n")
                for param in unified.param_list:
                    i = unified.param_list.index(param)
                    f.write(
                        f"{param} {pk.sigma[i]:.6g} {so.sigma[i]:.6g} "
                        f"{lb.sigma[i]:.6g} {unified.sigma[i]:.6g}\n"
                    )

    # Run non-adiabatic isocurvature cases per-correlation (existing behavior).
    for corr in corr_plot:
        for iso in iso_plot_nonadiabatic:
            _run_case(corr=corr, iso=iso, include_corr=True)

    # Run adiabatic (iso=None) once, independent of corr type, in a directory
    # alongside the corr subdirs (i.e., without the corr subdirectory).
    if run_adiabatic_once:
        corr0 = corr_plot[0]
        print(f"[compare_and_save_constraints] Adiabatic (iso=None) run once using corr={corr0} (corr is ignored).")
        _run_case(corr=corr0, iso=None, include_corr=False)


def load_planck_pliklite_data(
    planck_root: str | Path = "/home/sa5705/scratch/scratch/cobaya_packages/data/planck_2018_pliklite_native/",
    *,
    include_low_ell_tt: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Load PlikLite (Planck 2018) data via cobaya, returning a lite_data dict
    compatible with the older notebooks in this repo.
    """
    import glob
    from cobaya.likelihoods.base_classes.planck_pliklite import PlanckPlikLite

    planck_root = str(planck_root)
    dataset_files = glob.glob(os.path.join(planck_root, "**", "*plik*lite*.dataset"), recursive=True)
    dataset = [f for f in dataset_files if "lite" in os.path.basename(f)][0]
    lite = PlanckPlikLite({"dataset_file": dataset})

    cov_lite = lite.cov
    data_lite = lite.X_data

    ell_mid = (lite.blmin + lite.blmax) / 2
    dell = lite.blmax[lite.blmax <= 2508] - lite.blmin[lite.blmin <= 2476] + 1
    ell_pk = {
        "TT": ell_mid[ell_mid <= 2508],
        "TE": ell_mid[ell_mid <= 1996],
        "EE": ell_mid[ell_mid <= 1996],
    }

    lite_data = {
        "TT": {"c_ell": data_lite[:215], "cov": cov_lite.diagonal()[:215], "ell": ell_pk["TT"], "dell": dell},
        "TE": {
            "c_ell": data_lite[215:414],
            "cov": cov_lite.diagonal()[215:414],
            "ell": ell_pk["TE"],
            "dell": dell[:199],
        },
        "EE": {"c_ell": data_lite[414:], "cov": cov_lite.diagonal()[414:], "ell": ell_pk["EE"], "dell": dell[:199]},
    }

    if include_low_ell_tt:
        # Prepend ell<30 TT bins (Planck low-ell TT) to match the original workflow.
        # This is hard-coded in `nbs/prepend_low_ell.py`.
        from prepend_low_ell import prepend_planck_low_ell

        lite_data = prepend_planck_low_ell(lite_data)
        print("[load_planck_pliklite_data] Prepended Planck low-ell TT bins (ell<30).")

    return lite_data


def apply_planck_lowE_tau_prior(
    fisher,
    *,
    planck_likelihood_mode: str,
    tau_prior_sigma: float = 0.007,
) -> Any:
    """
    Apply an approximate Planck lowE constraint via a Gaussian tau prior.

    This is used when PLANCK_LIKELIHOOD_MODE='TTTEEE_lowE' is requested but no
    true low-ell EE bandpowers/likelihood are available in this repo.
    """
    if planck_likelihood_mode == "pliklite_lowT":
        return fisher
    if planck_likelihood_mode != "TTTEEE_lowE":
        raise ValueError(
            f"Unknown planck_likelihood_mode={planck_likelihood_mode!r} (expected 'pliklite_lowT' or 'TTTEEE_lowE')."
        )
    if "tau_reio" not in fisher.param_list:
        return fisher
    out = fisher.with_prior({"tau_reio": float(tau_prior_sigma)})
    out.metadata = (getattr(out, "metadata", {}) or {}) | {
        "planck_likelihood_mode": planck_likelihood_mode,
        "planck_lowE": "approximated_by_tau_prior",
        "planck_tau_prior_sigma": float(tau_prior_sigma),
    }
    return out


def _compute_cls_indexed(compute_cls, *, lmax: int, **theta) -> dict[str, Any]:
    """
    Wrap `iso_theory.compute_cls` output into arrays indexed by integer ell:
    out['TT'][ell] = C_ell, etc.  (Also keeps `ell` array.)

    This makes it compatible with the indexing conventions used by
    `cosmocast_makelik/multi_freq_liq/fisher_multi.py`.
    """
    raw = compute_cls(lmax=lmax, **theta)
    ell = raw["ell"].astype(int)
    out: dict[str, Any] = {"ell": ell}
    for ct in ["TT", "EE", "TE", "BB"]:
        if ct not in raw:
            continue
        arr = raw[ct]
        indexed = [0.0] * (lmax + 1)
        for e, v in zip(ell, arr):
            if 0 <= int(e) <= lmax:
                indexed[int(e)] = float(v)
        out[ct] = indexed
    return out


def _planck_lite_bands_from_lite_data(lite_data) -> list:
    """
    Create fisher_multi.SpectrumBand objects from `lite_data` returned by
    `load_planck_pliklite_data`.
    """
    from cosmocast_makelik.multi_freq_liq.fisher_multi import SpectrumBand
    import numpy as np

    bands = []
    for cell_type in ["TT", "TE", "EE"]:
        d = lite_data[cell_type]
        ell = np.asarray(d["ell"], dtype=float).astype(int)
        cov = np.asarray(d["cov"], dtype=float)
        dell = d.get("dell", 1)
        if hasattr(dell, "__len__"):
            # keep it scalar-ish for SpectrumBand; doesn't affect fisher_forecast
            dell = int(np.asarray(dell).ravel()[0])
        bands.append(
            SpectrumBand(
                exp_key="PK_Lite",
                channel=f"PK_Lite_{cell_type}",
                cell_type=cell_type,
                ell=ell,
                dell=int(dell),
                fsky=1.0,
                cov=cov,
            )
        )
    return bands


def compute_so_fisher_all(
    *,
    output_path: str | Path,
    lmax: int = 4000,
    years: list[int] = [1, 5, 10],
    iso_types: list[str | None] = [None, "cdi", "nid", "niv"],
    corr_types: list[str] = ["pcor", "ucor", "acor"],
    use_iso_signal_in_cov: bool = True,
    planck_likelihood_mode: str = "pliklite_lowT",
    planck_tau_prior_sigma: float = 0.007,
    use_standard_lcdm_amplitude: bool = False,
    use_ns: bool = False,
    standard_A_s: float = 2.1e-9,
    standard_n_s: float = 0.965,
    fsky_sat: float = 0.1,
    fsky_lat: float = 0.4,
    dell: int = 10,
    lat_lmin: int = 400,
    lat_lmax: int = 3000,
    sat_lmin: int = 30,
    sat_lmax: int = 400,
) -> Path:
    """
    Recompute and save a `so_fisher_all.pkl`-compatible dictionary.

    Notes
    -----
    - Requires `classy` (CLASS) to be installed, since we call `compute_cls`.
    - Uses `noise_calc_modified.py` for SO noise curves and Knox diagonal
      covariances for SO bands.
    - Planck is handled via cobaya's PlikLite, using the diagonal variances
      from the likelihood's covariance.
    - PLANCK_LIKELIHOOD_MODE can optionally approximate Planck lowE via a tau prior.
    - use_standard_lcdm_amplitude/use_ns enable a more standard ΛCDM parameterization
      for the adiabatic (iso_mode=None) case only (does not affect isocurvature runs).
    - If `use_iso_signal_in_cov=False`, the Fisher derivatives are still taken
      w.r.t. the selected isocurvature mode, but the SO covariance/mocks are
      built from a common adiabatic fiducial spectrum (`iso_mode=None`).
    """
    import numpy as np
    from copy import deepcopy

    repo = _repo_root()
    output_path = Path(output_path)

    # Imports that rely on repo layout
    sys.path.append(str(repo / "cosmocast_makelik"))
    sys.path.append(str(repo / "cosmocast_makelik" / "multi_freq_liq"))
    sys.path.append(str(repo / "nbs"))

    from iso_theory import compute_cls
    from cosmocast_makelik.multi_freq_liq import likelihood_multi
    from cosmocast_makelik.multi_freq_liq import fisher_multi
    import noise_calc_modified as noise_calc

    # PlanckLite data -> bands
    lite_data = load_planck_pliklite_data()
    planck_bands = _planck_lite_bands_from_lite_data(lite_data)

    # SO band selection
    req_spec = {
        "SAT": likelihood_multi.SAT_pairs_all,
        "LAT": likelihood_multi.LAT_pairs_all,
        "LAT_pol": likelihood_multi.LAT_pairs_all,
        "LAT_cross": likelihood_multi.LAT_pairs_all,
    }

    # shared params/steps (match previous notebooks)
    steps_abs = {
        "omega_b": 2.5e-5,
        "omega_cdm": 1.5e-4,
        "h": 6.0e-4,
        "tau_reio": 6.0e-5,
    }
    steps_rr = {"P_RR_1": 0.01, "P_RR_2": 0.01}
    steps_iso = {**steps_rr, "P_II_1": 0.05, "P_II_2": 0.05}
    scaled10_params_iso = {"P_RR_1", "P_RR_2", "P_II_1", "P_II_2"}
    scaled10_params_ad = {"P_RR_1", "P_RR_2"}
    param_list_ad = ["omega_b", "omega_cdm", "h", "tau_reio", "P_RR_1", "P_RR_2"]
    param_list_iso = [*param_list_ad, "P_II_1", "P_II_2"]
    steps_ad = {**steps_abs, **steps_rr}
    steps_iso = {**steps_abs, **steps_iso}

    # cases container: per year (we rebuild noise curves per year)
    # NOTE: `likelihood_multi.build_full_lik_data` historically uses different
    # case_key conventions:
    # - SAT includes `fm{f_mode}`  (e.g. SAT_y10_sm1fm0)
    # - LAT/LAT_pol/LAT_cross omit it (e.g. LAT_y10_sm1)
    def _case_key(exp: str, yrs: int, sens_mode: int, f_mode: int) -> str:
        if exp == "SAT":
            return f"{exp}_y{yrs:.0f}_sm{sens_mode:.0f}fm{f_mode:.0f}"
        return f"{exp}_y{yrs:.0f}_sm{sens_mode:.0f}"

    so_fisher_all: dict = {}

    for corr in corr_types:
        so_fisher_all[corr] = {"SO": {}, "PK_Lite": {}}

        theta0 = THETA_FULL[corr]["theta0"]
        corr_dict = THETA_FULL[corr]["corr"]

        for iso_mode in iso_types:
            so_fisher_all[corr]["SO"][iso_mode] = {}
            so_fisher_all[corr]["PK_Lite"][iso_mode] = {}

            if use_standard_lcdm_amplitude and iso_mode is None:
                theta0_run = _theta0_standard_lcdm(theta0_base=theta0, n_s=standard_n_s, A_s=standard_A_s)
                corr_dict_run: dict[str, float] = {}
                _cls_provider = _cls_provider_with_standard_lcdm_amplitude(compute_cls, iso_mode=None)
            else:
                theta0_run = dict(theta0)
                corr_dict_run = dict(corr_dict)

                def _cls_provider(*, lmax: int = lmax, **th):
                    # fisher_multi will pass `lmax=...`; accept it and forward.
                    return _compute_cls_indexed(compute_cls, lmax=int(lmax), iso_mode=iso_mode, **th)

            # Planck-only Fisher (year-independent)
            # Raw theory for building SO mock "data" (expects arrays aligned with `ell`).
            # Derivatives should always follow the selected isocurvature mode, but the
            # covariance normalization can optionally be forced to use a common adiabatic
            # fiducial (iso_mode=None).
            if iso_mode is None:
                if use_standard_lcdm_amplitude:
                    param_list = ["omega_b", "omega_cdm", "h", "tau_reio", "ln10A_s"]
                    if use_ns:
                        param_list.append("n_s")
                    steps = _default_steps_for_param_list(param_list)
                    scaled10_params = set()
                    print(f"[compute_so_fisher_all] param_list (standard LCDM) = {param_list}")
                    cmb_theo_deriv_raw = compute_cls(
                        lmax=lmax,
                        iso_mode=None,
                        power_mode="standard",
                        omega_b=theta0_run["omega_b"],
                        omega_cdm=theta0_run["omega_cdm"],
                        h=theta0_run["h"],
                        tau_reio=theta0_run["tau_reio"],
                        A_s=float(np.exp(theta0_run["ln10A_s"]) * 1e-10),
                        n_s=float(theta0_run.get("n_s", standard_n_s)),
                    )
                else:
                    param_list = param_list_ad
                    steps = steps_ad
                    scaled10_params = scaled10_params_ad
                    print(f"[compute_so_fisher_all] param_list (legacy binned) = {param_list}")
                    cmb_theo_deriv_raw = compute_cls(lmax=lmax, iso_mode=iso_mode, **theta0_run, **corr_dict_run)
            else:
                param_list = param_list_iso
                steps = steps_iso
                scaled10_params = scaled10_params_iso
                cmb_theo_deriv_raw = compute_cls(lmax=lmax, iso_mode=iso_mode, **theta0_run, **corr_dict_run)
            if use_iso_signal_in_cov or iso_mode is None:
                cmb_theo_cov_raw = cmb_theo_deriv_raw
                if iso_mode is None:
                    print("[compute_so_fisher_all] SO covariance built from adiabatic fiducial spectra (iso_mode=None).")
                else:
                    print(f"[compute_so_fisher_all] SO covariance built from iso fiducial spectra (iso_mode={iso_mode}).")
            else:
                cmb_theo_cov_raw = compute_cls(lmax=lmax, iso_mode=None, **theta0_run, **corr_dict_run)
                print(
                    f"[compute_so_fisher_all] SO covariance built from common adiabatic fiducial spectra "
                    f"(iso_mode=None) while derivatives use iso_mode={iso_mode}."
                )
            fisher_pk = fisher_multi.fisher_forecast(
                theta0={**theta0_run, **corr_dict_run},
                param_list=param_list,
                bands=planck_bands,
                compute_cls=_cls_provider,
                steps=steps,
                scaled_params=scaled10_params,
                ell_max=lmax,
                use_pinv=True,
            )
            fisher_pk = apply_planck_lowE_tau_prior(
                fisher_pk,
                planck_likelihood_mode=planck_likelihood_mode,
                tau_prior_sigma=planck_tau_prior_sigma,
            )
            if planck_likelihood_mode == "TTTEEE_lowE":
                print(
                    f"[compute_so_fisher_all] WARNING: PLANCK_LIKELIHOOD_MODE='TTTEEE_lowE' "
                    f"implemented via tau prior sigma={planck_tau_prior_sigma} (approximation; no true lowE bandpowers)."
                )

            for yr in years:
                yr_tag = f"{yr}yr"
                so_fisher_all[corr]["PK_Lite"][iso_mode][yr_tag] = fisher_pk

                # ── build SO noise dict for this year ───────────────────
                full_noise_dict: dict = {}

                # SAT
                sat_key = _case_key("SAT", yr, 1, 0)
                ell_sa, n_ell_p_sa, _white = noise_calc.Simons_Observatory_V3_SA_noise(
                    1, 0, float(yr), fsky_sat, sat_lmax, dell, beam_stuff=True
                )
                full_noise_dict[sat_key] = {
                    "yrs": float(yr),
                    "sens_mode": 1,
                    "f_mode": 0,
                    "fsky": fsky_sat,
                    "lmax": sat_lmax,
                    "dell": dell,
                    "cuts": [sat_lmin, sat_lmax],
                    "nz_dict": likelihood_multi.make_nz_dict_from_array(n_ell_p_sa, likelihood_multi.frequencies, "SAT"),
                    "ell_nz": ell_sa,
                }

                # LAT (TT) + LAT_pol (EE): use same noise model; pick temp vs pol later
                lat_key = _case_key("LAT", yr, 1, 0)
                ell_la, n_ell_t_la, n_ell_p_la, _white = noise_calc.Simons_Observatory_V3_LA_noise(
                    1, float(yr), fsky_lat, lat_lmax, dell
                )
                # store both under same key; likelihood_multi expects nz_dict to be for the chosen spectrum
                full_noise_dict[lat_key] = {
                    "yrs": float(yr),
                    "sens_mode": 1,
                    "f_mode": 0,
                    "fsky": fsky_lat,
                    "lmax": lat_lmax,
                    "dell": dell,
                    "cuts": [lat_lmin, lat_lmax],
                    "nz_dict": likelihood_multi.make_nz_dict_from_array(n_ell_t_la, likelihood_multi.frequencies, "LAT"),
                    "ell_nz": ell_la,
                }
                latpol_key = _case_key("LAT_pol", yr, 1, 0)
                full_noise_dict[latpol_key] = {
                    **{k: v for k, v in full_noise_dict[lat_key].items() if k not in {"nz_dict"}},
                    "nz_dict": likelihood_multi.make_nz_dict_from_array(n_ell_p_la, likelihood_multi.frequencies, "LAT"),
                }
                latcross_key = _case_key("LAT_cross", yr, 1, 0)
                full_noise_dict[latcross_key] = {
                    **{k: v for k, v in full_noise_dict[lat_key].items() if k not in {"nz_dict"}},
                    "nz_dict": None,
                }

                # ── theory for SO blocks ────────────────────────────────
                cmb_theo = cmb_theo_cov_raw

                setup = {
                    "SAT": {"yrs": float(yr), "sens_mode": 1, "f_mode": 0},
                    "LAT": {"yrs": float(yr), "sens_mode": 1, "f_mode": 0},
                    "LAT_pol": {"yrs": float(yr), "sens_mode": 1, "f_mode": 0},
                    "LAT_cross": {"yrs": float(yr), "sens_mode": 1, "f_mode": 0},
                }

                full_lik_data = likelihood_multi.build_full_lik_data(req_spec, setup, full_noise_dict, cmb_theo)
                full_lik_cov = likelihood_multi.build_full_lik_cov(full_lik_data, setup, full_noise_dict)

                so_bands = fisher_multi.parse_spectrum_bands(full_lik_cov, ["SAT", "LAT", "LAT_pol"])
                fisher_so = fisher_multi.fisher_forecast(
                    theta0={**theta0_run, **corr_dict_run},
                    param_list=param_list,
                    bands=so_bands,
                    compute_cls=_cls_provider,
                    steps=steps,
                    scaled_params=scaled10_params,
                    ell_max=lmax,
                    use_pinv=True,
                )
                so_fisher_all[corr]["SO"][iso_mode][yr_tag] = fisher_so

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(so_fisher_all, f)
    return output_path


def compute_litebird_fisher_grids(
    *,
    output_path: str | Path,
    year_tag: str = "10yr",
    iso_types: list[str | None] = [None, "cdi", "nid", "niv"],
    corr_types: list[str] = ["pcor", "ucor", "acor"],
    use_iso_signal_in_cov: bool = True,
    use_standard_lcdm_amplitude: bool = False,
    use_ns: bool = False,
    standard_A_s: float = 2.1e-9,
    standard_n_s: float = 0.965,
    lmin: int = 2,
    lmax: int = 200,
    dell: int = 10,
    fsky: float = 0.7,
    ell_max_theory: int = 3000,
) -> Path:
    """
    Recompute and save a `lb_fisher_10.pkl`-compatible dictionary:

        out[corr]['LB'][iso][year_tag] = FisherResult

    This uses a simple LiteBIRD-only Fisher built from TT/TE/EE Knox-diagonal
    covariances with an inverse-variance combined noise curve from the
    LiteBIRD table in the original notebook.

    Important: we intentionally do NOT scale LiteBIRD noise with time.
    """
    output_path = Path(output_path)

    repo = _repo_root()
    sys.path.append(str(repo / "cosmocast_makelik"))
    sys.path.append(str(repo / "cosmocast_makelik" / "multi_freq_liq"))

    try:
        from iso_theory import compute_cls
    except Exception as e:
        raise RuntimeError("LiteBIRD recomputation requires `classy` (CLASS) to be installed.") from e

    from cosmocast_makelik.iso_theory import knox_auto_cov, knox_cross_cov
    from cosmocast_makelik.multi_freq_liq.fisher_multi import SpectrumBand
    from cosmocast_makelik.multi_freq_liq import fisher_multi
    import numpy as np

    # shared params/steps (match compute_so_fisher_all)
    steps_abs = {"omega_b": 2.5e-5, "omega_cdm": 1.5e-4, "h": 6.0e-4, "tau_reio": 6.0e-5}
    steps_rr = {"P_RR_1": 0.01, "P_RR_2": 0.01}
    steps_iso = {**steps_rr, "P_II_1": 0.05, "P_II_2": 0.05}
    scaled10_params_iso = {"P_RR_1", "P_RR_2", "P_II_1", "P_II_2"}
    scaled10_params_ad = {"P_RR_1", "P_RR_2"}
    param_list_ad = ["omega_b", "omega_cdm", "h", "tau_reio", "P_RR_1", "P_RR_2"]
    param_list_iso = [*param_list_ad, "P_II_1", "P_II_2"]
    steps_ad = {**steps_abs, **steps_rr}
    steps_iso = {**steps_abs, **steps_iso}

    noise = litebird_noise_curves(lmin=lmin, lmax=lmax, dell=dell)
    ell_bins = np.asarray(noise["ell"], dtype=int)
    n_tt = np.asarray(noise["n_ell_tt"], dtype=float)
    n_ee = np.asarray(noise["n_ell_ee"], dtype=float)

    out: dict[str, Any] = {}

    for corr in corr_types:
        out[corr] = {"LB": {}}
        theta0 = THETA_FULL[corr]["theta0"]
        corr_dict = THETA_FULL[corr]["corr"]

        for iso_mode in iso_types:
            out[corr]["LB"][iso_mode] = {}

            def _cls_provider(*, lmax: int = ell_max_theory, **th):
                if use_standard_lcdm_amplitude and iso_mode is None:
                    return _cls_provider_with_standard_lcdm_amplitude(compute_cls, iso_mode=None)(lmax=int(lmax), **th)
                return _compute_cls_indexed(compute_cls, lmax=int(lmax), iso_mode=iso_mode, **th)

            if iso_mode is None:
                if use_standard_lcdm_amplitude:
                    param_list = ["omega_b", "omega_cdm", "h", "tau_reio", "ln10A_s"]
                    if use_ns:
                        param_list.append("n_s")
                    steps = _default_steps_for_param_list(param_list)
                    scaled10_params = set()
                    theta0_run = _theta0_standard_lcdm(theta0_base=theta0, n_s=standard_n_s, A_s=standard_A_s)
                    corr_dict_run: dict[str, float] = {}
                else:
                    param_list = param_list_ad
                    steps = steps_ad
                    scaled10_params = scaled10_params_ad
                    theta0_run = dict(theta0)
                    corr_dict_run = dict(corr_dict)
            else:
                param_list = param_list_iso
                steps = steps_iso
                scaled10_params = scaled10_params_iso
                theta0_run = dict(theta0)
                corr_dict_run = dict(corr_dict)

            # Fiducial theory for covariance (just needs to cover ell_bins)
            cov_iso_mode = iso_mode if (use_iso_signal_in_cov or iso_mode is None) else None
            if cov_iso_mode is None:
                if iso_mode is None:
                    print("[compute_litebird_fisher_grids] LB covariance built from adiabatic fiducial spectra (iso_mode=None).")
                else:
                    print(
                        f"[compute_litebird_fisher_grids] LB covariance built from common adiabatic fiducial spectra "
                        f"(iso_mode=None) while derivatives use iso_mode={iso_mode}."
                    )
            else:
                print(f"[compute_litebird_fisher_grids] LB covariance built from iso fiducial spectra (iso_mode={iso_mode}).")

            if use_standard_lcdm_amplitude and iso_mode is None:
                # For standard mode, use adiabatic spectra with A_s/n_s.
                import numpy as np

                cls0 = _compute_cls_indexed(
                    compute_cls,
                    lmax=ell_max_theory,
                    iso_mode=None,
                    power_mode="standard",
                    omega_b=theta0_run["omega_b"],
                    omega_cdm=theta0_run["omega_cdm"],
                    h=theta0_run["h"],
                    tau_reio=theta0_run["tau_reio"],
                    A_s=float(np.exp(theta0_run["ln10A_s"]) * 1e-10),
                    n_s=float(theta0_run.get("n_s", standard_n_s)),
                )
            else:
                cls0 = _compute_cls_indexed(compute_cls, lmax=ell_max_theory, iso_mode=cov_iso_mode, **theta0_run, **corr_dict_run)
            c_tt = np.asarray(cls0["TT"])[ell_bins]
            c_ee = np.asarray(cls0["EE"])[ell_bins]
            c_te = np.asarray(cls0["TE"])[ell_bins]

            cov_tt = knox_auto_cov(c_tt + n_tt, ell_bins, dell, fsky)
            cov_ee = knox_auto_cov(c_ee + n_ee, ell_bins, dell, fsky)
            cov_te = knox_cross_cov(c_te, c_tt + n_tt, c_ee + n_ee, ell_bins, dell, fsky)

            bands = [
                SpectrumBand(
                    exp_key="LB",
                    channel="LB_TT",
                    cell_type="TT",
                    ell=ell_bins,
                    dell=int(dell),
                    fsky=float(fsky),
                    cov=cov_tt,
                ),
                SpectrumBand(
                    exp_key="LB",
                    channel="LB_TE",
                    cell_type="TE",
                    ell=ell_bins,
                    dell=int(dell),
                    fsky=float(fsky),
                    cov=cov_te,
                ),
                SpectrumBand(
                    exp_key="LB",
                    channel="LB_EE",
                    cell_type="EE",
                    ell=ell_bins,
                    dell=int(dell),
                    fsky=float(fsky),
                    cov=cov_ee,
                ),
            ]

            fisher_lb = fisher_multi.fisher_forecast(
                theta0={**theta0_run, **corr_dict_run},
                param_list=param_list,
                bands=bands,
                compute_cls=_cls_provider,
                steps=steps,
                scaled_params=scaled10_params,
                ell_max=ell_max_theory,
                use_pinv=True,
            )
            fisher_lb.metadata = {"fsky": float(fsky), "ell_bins": ell_bins.tolist(), "dell": int(dell)}
            out[corr]["LB"][iso_mode][year_tag] = fisher_lb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(out, f)
    return output_path


def smoke_test_standard_lcdm_spectra(*, compute_cls: Callable) -> None:
    """
    Minimal sanity checks for the standard ΛCDM interface:
    - A_s affects TT amplitude in power_mode='standard'
    - n_s affects TT tilt in power_mode='standard'

    Skips (with a message) if CLASS/classy is unavailable.
    """
    import numpy as np

    try:
        cls0 = compute_cls(
            lmax=80,
            iso_mode=None,
            power_mode="standard",
            omega_b=0.02237,
            omega_cdm=0.11933,
            h=0.6766,
            tau_reio=0.0561,
            A_s=2.1e-9,
            n_s=0.965,
        )
    except ModuleNotFoundError as e:
        print(f"[smoke_test_standard_lcdm_spectra] SKIP: {e}")
        return

    cls1 = compute_cls(
        lmax=80,
        iso_mode=None,
        power_mode="standard",
        omega_b=0.02237,
        omega_cdm=0.11933,
        h=0.6766,
        tau_reio=0.0561,
        A_s=2.1e-9 * 1.01,
        n_s=0.965,
    )
    cls2 = compute_cls(
        lmax=80,
        iso_mode=None,
        power_mode="standard",
        omega_b=0.02237,
        omega_cdm=0.11933,
        h=0.6766,
        tau_reio=0.0561,
        A_s=2.1e-9,
        n_s=0.975,
    )

    ell = np.asarray(cls0["ell"], dtype=int)
    tt0 = np.asarray(cls0["TT"], dtype=float)
    tt1 = np.asarray(cls1["TT"], dtype=float)
    tt2 = np.asarray(cls2["TT"], dtype=float)

    # Use ell~50 as a representative point.
    i50 = int(np.where(ell == 50)[0][0]) if np.any(ell == 50) else -1
    if i50 >= 0 and tt0[i50] > 0:
        print("[smoke_test_standard_lcdm_spectra] TT ratio at ell=50 for +1% A_s:", float(tt1[i50] / tt0[i50]))
        print("[smoke_test_standard_lcdm_spectra] TT ratio at ell=50 for n_s=0.975:", float(tt2[i50] / tt0[i50]))
    else:
        print("[smoke_test_standard_lcdm_spectra] WARNING: could not locate ell=50 sample.")
