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
    ell_by_ell_common_dell: int | None = None,
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
        if str(ell_by_ell_policy) == "min_diag_cov_common_bins":
            dd = int(ell_by_ell_common_dell) if ell_by_ell_common_dell is not None else 20
            suffix.append(f"cb{dd}")
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
    ell_max: int | None = None,
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
            # Expand the binned point onto an integer-ell grid so different
            # experiments with different bin centers still overlap when we
            # select an "effective experiment" ell-by-ell.
            half = max(0, int(dell) // 2)
            for l2 in range(int(ell) - half, int(ell) + half + 1):
                if l2 < 2:
                    continue
                if ell_max is not None and int(l2) > int(ell_max):
                    continue
                if (l2 not in n_by_ell) or (n < n_by_ell[l2]):
                    n_by_ell[l2] = float(n)
                    dell_by_ell[l2] = int(dell)
                    fsky_by_ell[l2] = float(fsky)
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
    ell_by_ell_common_bins: Sequence[tuple[int, int]] | None = None,
    ell_by_ell_common_dell: int = 20,
    ell_by_ell_score: str = "sigma_over_sqrt_dell",
    ell_by_ell_hysteresis: float = 1.05,
    ell_by_ell_min_segment_len: int = 2,
    ell_by_ell_score_smooth_window: int = 3,
    verbose: bool = False,
):
    """
    Teo-style effective-experiment combination:
      - infer per-experiment auto-spectrum noise curves from band variances
      - choose how to combine experiments at each integer ell (see ell_by_ell_policy)
      - build a single effective covariance from (C_ell + N_eff) and per-ell f_sky / Δℓ
      - compute one Fisher matrix from the effective bands.

    ell_by_ell_policy:
      - "min_noise": select a single experiment at each ell (can look piecewise at transitions)
      - "inv_var": inverse-variance combine experiments at each ell (smoother across overlaps)
      - "min_diag_cov_common_bins": rebin all experiments onto a common binning, then choose one owner per common bin

    Experiment/f_sky conventions (per requirement):
      - Planck:   0.7
      - LiteBIRD: 0.7
      - SO SAT:   0.1
      - SO LAT:   0.4   (TT from LAT, EE from LAT_pol)
    """
    import numpy as np

    if str(ell_by_ell_policy) not in {"min_noise", "inv_var", "min_diag_cov_common_bins"}:
        raise ValueError(
            f"Unsupported ELL_BY_ELL_POLICY={ell_by_ell_policy!r} "
            "(expected 'min_noise', 'inv_var', or 'min_diag_cov_common_bins')."
        )

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

    def _rolling_median(x: np.ndarray, *, window: int) -> np.ndarray:
        w = int(window)
        if w <= 1 or len(x) == 0:
            return np.asarray(x, dtype=float)
        w = min(w, len(x))
        half = w // 2
        out = np.full_like(np.asarray(x, dtype=float), np.nan, dtype=float)
        for i in range(len(out)):
            lo = max(0, i - half)
            hi = min(len(out), i + half + 1)
            seg = np.asarray(x[lo:hi], dtype=float)
            seg = seg[np.isfinite(seg)]
            if len(seg) == 0:
                continue
            out[i] = float(np.median(seg))
        return out

    def _fix_short_segments(owners: list[str], *, min_len: int) -> list[str]:
        n = len(owners)
        if n == 0 or int(min_len) <= 1:
            return owners
        out = list(owners)
        i = 0
        while i < n:
            j = i + 1
            while j < n and out[j] == out[i]:
                j += 1
            seg_len = j - i
            if seg_len < int(min_len):
                left = out[i - 1] if i - 1 >= 0 else None
                right = out[j] if j < n else None
                repl = left if left is not None else right
                if repl is None:
                    i = j
                    continue
                for k in range(i, j):
                    out[k] = repl
            i = j
        return out
    
    def _select_owners_with_hysteresis(
        *,
        scores_by_exp: dict[str, np.ndarray],
        exp_names: list[str],
        hysteresis: float,
        min_segment_len: int,
    ) -> list[str]:
        """
        Select owner per common bin with hysteresis.

        Important fix:
        If the current owner has no finite score in a bin, immediately switch to the
        best finite owner. Otherwise an experiment such as LiteBIRD can own low ell
        and then incorrectly remain the owner at high ell where it has no coverage.
        """
        import numpy as np

        nbin = len(next(iter(scores_by_exp.values()))) if scores_by_exp else 0
        if nbin == 0:
            return []

        # Best finite experiment per bin.
        best_by_bin: list[str] = []
        for i in range(nbin):
            best = ""
            best_score = np.inf

            for name in exp_names:
                s = float(scores_by_exp[name][i])
                if not np.isfinite(s):
                    continue

                if s < best_score:
                    best_score = s
                    best = name

            best_by_bin.append(best)

        owners: list[str] = []
        current = ""

        for i in range(nbin):
            best = best_by_bin[i]

            # No experiment has coverage in this bin.
            if best == "":
                current = ""
                owners.append("")
                continue

            # No current owner, so take the best available.
            if current == "":
                current = best
                owners.append(current)
                continue

            # If current owner is not available in this bin, force switch.
            if current not in scores_by_exp:
                current = best
                owners.append(current)
                continue

            s_cur = float(scores_by_exp[current][i])
            s_best = float(scores_by_exp[best][i])

            if not np.isfinite(s_cur):
                current = best
                owners.append(current)
                continue

            if not np.isfinite(s_best):
                owners.append(current)
                continue

            # If the current owner is still the best, stay.
            if best == current:
                owners.append(current)
                continue

            # Hysteresis: switch only if the new best is sufficiently better.
            if s_best <= s_cur / float(hysteresis):
                current = best

            owners.append(current)

        owners = _fix_short_segments(owners, min_len=int(min_segment_len))

        # Second safety pass: after short-segment fixing, do not allow an owner to
        # remain assigned to a bin where that owner has no finite score.
        for i, owner in enumerate(owners):
            if owner == "":
                continue

            s_owner = float(scores_by_exp.get(owner, np.full(nbin, np.nan))[i])

            if np.isfinite(s_owner):
                continue

            # Replace invalid owner with the best finite owner for this bin.
            best = best_by_bin[i]
            owners[i] = best

        return owners

    def _band_entries_for_spec(
        *,
        bands: Sequence[Any],
        cell_type: str,
        fsky_override: float | None,
    ) -> list[dict[str, float]]:
        """
        Extract per-band diagonal variances with inferred [ell_min, ell_max] support.
        Returned entries are dicts with: ell_eff, ell_min, ell_max, var, dell, fsky.
        """
        out: list[dict[str, float]] = []
        for b in bands:
            if str(getattr(b, "cell_type", "")) != str(cell_type):
                continue
            ell_arr = np.asarray(getattr(b, "ell"), dtype=float)
            var_arr = np.asarray(getattr(b, "cov"), dtype=float)
            dell = int(getattr(b, "dell", 1))
            fsky = float(getattr(b, "fsky", 1.0)) if fsky_override is None else float(fsky_override)
            if len(ell_arr) != len(var_arr):
                continue
            half = int(dell) // 2
            # Map to inclusive integer bounds with approximate width == dell.
            for e, v in zip(ell_arr, var_arr):
                if not np.isfinite(v) or float(v) <= 0:
                    continue
                cen = int(np.round(float(e)))
                ell_min = int(cen - half)
                ell_max = int(ell_min + int(dell) - 1)
                if ell_max < 2:
                    continue
                ell_min = max(2, ell_min)
                out.append(
                    {
                        "ell_eff": float(cen),
                        "ell_min": float(ell_min),
                        "ell_max": float(ell_max),
                        "var": float(v),
                        "dell": float(dell),
                        "fsky": float(fsky),
                    }
                )
        # Consolidate identical ranges by taking the minimum variance (avoid double counting channels).
        merged: dict[tuple[int, int], dict[str, float]] = {}
        for d in out:
            k = (int(d["ell_min"]), int(d["ell_max"]))
            if k not in merged or float(d["var"]) < float(merged[k]["var"]):
                merged[k] = d
        return list(merged.values())

    def _build_common_bins_for_spec(
        *, entries_by_exp: dict[str, list[dict[str, float]]], common_bins: Sequence[tuple[int, int]] | None, dell: int
    ) -> list[tuple[int, int]]:
        if common_bins is not None:
            out = [(int(a), int(b)) for (a, b) in common_bins if int(b) >= int(a)]
            return out
        ell_min = None
        ell_max = None
        for ent in entries_by_exp.values():
            for d in ent:
                ell_min = int(d["ell_min"]) if ell_min is None else min(int(ell_min), int(d["ell_min"]))
                ell_max = int(d["ell_max"]) if ell_max is None else max(int(ell_max), int(d["ell_max"]))
        if ell_min is None or ell_max is None:
            return []
        ell_min = max(2, int(ell_min))
        ell_max = int(ell_max)
        dd = max(1, int(dell))
        bins: list[tuple[int, int]] = []
        l = ell_min
        while l <= ell_max:
            r = min(ell_max, l + dd - 1)
            bins.append((int(l), int(r)))
            l = r + 1
        return bins

    def _rebin_to_common_bins(
        *,
        entries: list[dict[str, float]],
        common_bins: list[tuple[int, int]],
        coverage_fraction_min: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Map original diagonal variances onto common bins using overlap weighting.
        Returns (ell_eff, var_common, coverage_frac) arrays aligned with common_bins.
        """
        ell_eff = np.array([(a + b) / 2.0 for (a, b) in common_bins], dtype=float)
        var = np.full(len(common_bins), np.nan, dtype=float)
        covfrac = np.zeros(len(common_bins), dtype=float)

        if not entries or not common_bins:
            return ell_eff, var, covfrac

        # Precompute entry intervals.
        e_int = []
        for d in entries:
            a = float(d["ell_min"])
            b = float(d["ell_max"])
            if b < a:
                continue
            e_int.append((a, b, float(d["var"]), float(d["fsky"]), float(d["dell"])))

        for i, (cmin, cmax) in enumerate(common_bins):
            cmin = float(cmin)
            cmax = float(cmax)
            clen = float(cmax - cmin + 1.0)
            if clen <= 0:
                continue
            total_overlap = 0.0
            vv = 0.0
            for (a, b, v, _fsky, _dell) in e_int:
                ov = max(0.0, min(b, cmax) - max(a, cmin) + 1.0)
                if ov <= 0:
                    continue
                w = float(ov / clen)
                total_overlap += ov
                vv += (w * w) * float(v)
            covfrac[i] = float(total_overlap / clen) if clen > 0 else 0.0
            if covfrac[i] >= float(coverage_fraction_min) and vv > 0 and np.isfinite(vv):
                var[i] = float(vv)
        return ell_eff, var, covfrac

    def _cov_matrix_from_bands(
        *,
        bands: Sequence[Any],
        cell_type: str,
        fsky_override: float | None,
    ) -> tuple[np.ndarray, np.ndarray, int, float, bool]:
        """
        Extract an experiment covariance for (cell_type) from SpectrumBand-like objects.

        Returns (ell_centers, cov, dell, fsky, is_diagonal). If no full covariance is
        present, returns a diagonal covariance matrix built from the best available
        diagonal variances across channels.
        """
        import numpy as np

        # Prefer a single band with a full covariance matrix, if present.
        full_candidates = []
        diag_candidates = []
        for b in bands:
            if str(getattr(b, "cell_type", "")) != str(cell_type):
                continue
            ell = np.asarray(getattr(b, "ell"), dtype=float)
            cov = np.asarray(getattr(b, "cov"), dtype=float)
            dell = int(getattr(b, "dell", 1))
            fsky = float(getattr(b, "fsky", 1.0)) if fsky_override is None else float(fsky_override)
            if cov.ndim == 2 and cov.shape[0] == cov.shape[1] and cov.shape[0] == len(ell):
                full_candidates.append((ell, cov, dell, fsky))
            elif cov.ndim == 1 and len(cov) == len(ell):
                diag_candidates.append((ell, cov, dell, fsky))

        if full_candidates:
            # Choose the largest matrix (most bins).
            ell, cov, dell, fsky = sorted(full_candidates, key=lambda t: int(len(t[0])), reverse=True)[0]
            return np.asarray(ell, dtype=float), np.asarray(cov, dtype=float), int(dell), float(fsky), False

        if not diag_candidates:
            return np.array([], dtype=float), np.zeros((0, 0), dtype=float), 1, float(fsky_override or 1.0), True

        # Conservative across channels: keep minimum variance per ell center.
        # (Assumes channels are measuring the same estimator with shared signal; we avoid inv-var combining here.)
        base_ell = None
        best_var: dict[int, float] = {}
        best_dell: dict[int, int] = {}
        best_fsky: dict[int, float] = {}
        for ell, var, dell, fsky in diag_candidates:
            if base_ell is None:
                base_ell = np.asarray(ell, dtype=float)
            for e, v in zip(np.asarray(ell, dtype=float), np.asarray(var, dtype=float)):
                ee = int(np.round(float(e)))
                vv = float(v)
                if not np.isfinite(vv) or vv <= 0:
                    continue
                if (ee not in best_var) or (vv < best_var[ee]):
                    best_var[ee] = vv
                    best_dell[ee] = int(dell)
                    best_fsky[ee] = float(fsky)

        ells = np.array(sorted(best_var.keys()), dtype=float)
        if len(ells) == 0:
            return np.array([], dtype=float), np.zeros((0, 0), dtype=float), 1, float(fsky_override or 1.0), True
        var_vec = np.array([best_var[int(e)] for e in ells], dtype=float)
        cov = np.diag(var_vec)
        # Metadata: keep conservative (min) across selected points.
        dell_eff = int(min(best_dell[int(e)] for e in ells))
        fsky_eff = float(min(best_fsky[int(e)] for e in ells))
        return ells, cov, dell_eff, fsky_eff, True

    def _rebin_cov_matrix_to_common_bins(
        *,
        ell_centers: np.ndarray,
        cov: np.ndarray,
        dell: int,
        common_bins: list[tuple[int, int]],
        coverage_fraction_min: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Rebin a covariance matrix defined on original bins (ell_centers with scalar dell)
        onto common bins using overlap weights.

        Returns (ell_eff_common, cov_common, coverage_frac_common, W) where W is the
        overlap weight matrix (n_common, n_orig).
        """
        import numpy as np

        ell_centers = np.asarray(ell_centers, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if len(ell_centers) == 0:
            n = len(common_bins)
            return (
                np.array([(a + b) / 2.0 for (a, b) in common_bins], dtype=float),
                np.zeros((n, n), dtype=float),
                np.zeros(n, dtype=float),
                np.zeros((n, 0), dtype=float),
            )
        if cov.ndim == 1:
            cov = np.diag(np.asarray(cov, dtype=float))
        if cov.shape[0] != cov.shape[1] or cov.shape[0] != len(ell_centers):
            raise ValueError("cov must be square with size matching ell_centers")

        # Original bin intervals inferred from scalar dell around center (same convention as elsewhere).
        half = int(dell) // 2
        orig_bins = []
        for e in ell_centers:
            cen = int(np.round(float(e)))
            omin = max(2, int(cen - half))
            omax = int(omin + int(dell) - 1)
            orig_bins.append((float(omin), float(omax)))

        ell_eff_common = np.array([(a + b) / 2.0 for (a, b) in common_bins], dtype=float)
        n_common = len(common_bins)
        n_orig = len(orig_bins)
        W = np.zeros((n_common, n_orig), dtype=float)
        covfrac = np.zeros(n_common, dtype=float)
        for i, (cmin_i, cmax_i) in enumerate(common_bins):
            cmin = float(cmin_i)
            cmax = float(cmax_i)
            clen = float(cmax - cmin + 1.0)
            total_overlap = 0.0
            if clen <= 0:
                continue
            for j, (omin, omax) in enumerate(orig_bins):
                ov = max(0.0, min(omax, cmax) - max(omin, cmin) + 1.0)
                if ov <= 0:
                    continue
                w = float(ov / clen)
                W[i, j] = w
                total_overlap += ov
            covfrac[i] = float(total_overlap / clen) if clen > 0 else 0.0

        cov_common = W @ cov @ W.T
        # Mask bins that don't meet coverage.
        ok = covfrac >= float(coverage_fraction_min)
        # We keep the full matrix but will treat uncovered bins as NaN on the diagonal for scoring;
        # and they will be excluded from the effective band list.
        diag = np.diag(cov_common).astype(float)
        diag[~ok] = np.nan
        # Write back NaNs only on diagonal; keep off-diags as computed (they won't be used if bins excluded).
        np.fill_diagonal(cov_common, diag)
        return ell_eff_common, cov_common, covfrac, W

    # Build noise dictionaries for TT/EE per "effective experiment"
    n_tt_planck, dell_tt_planck, fsky_tt_planck = _noise_dict_from_bands_auto(
        bands=planck_bands, spectrum="TT", fid_c_ell=cls_cov, fsky_override=0.7, ell_max=int(ell_max_theory)
    )
    n_ee_planck, dell_ee_planck, fsky_ee_planck = _noise_dict_from_bands_auto(
        bands=planck_bands, spectrum="EE", fid_c_ell=cls_cov, fsky_override=0.7, ell_max=int(ell_max_theory)
    )
    n_tt_lb, dell_tt_lb, fsky_tt_lb = _noise_dict_from_bands_auto(
        bands=litebird_bands, spectrum="TT", fid_c_ell=cls_cov, fsky_override=0.7, ell_max=int(ell_max_theory)
    )
    n_ee_lb, dell_ee_lb, fsky_ee_lb = _noise_dict_from_bands_auto(
        bands=litebird_bands, spectrum="EE", fid_c_ell=cls_cov, fsky_override=0.7, ell_max=int(ell_max_theory)
    )
    n_tt_sat, dell_tt_sat, fsky_tt_sat = _noise_dict_from_bands_auto(
        bands=so_sat, spectrum="TT", fid_c_ell=cls_cov, fsky_override=0.1, ell_max=int(ell_max_theory)
    )
    n_ee_sat, dell_ee_sat, fsky_ee_sat = _noise_dict_from_bands_auto(
        bands=so_sat, spectrum="EE", fid_c_ell=cls_cov, fsky_override=0.1, ell_max=int(ell_max_theory)
    )
    n_tt_lat, dell_tt_lat, fsky_tt_lat = _noise_dict_from_bands_auto(
        bands=so_lat_tt, spectrum="TT", fid_c_ell=cls_cov, fsky_override=0.4, ell_max=int(ell_max_theory)
    )
    n_ee_lat, dell_ee_lat, fsky_ee_lat = _noise_dict_from_bands_auto(
        bands=so_lat_ee, spectrum="EE", fid_c_ell=cls_cov, fsky_override=0.4, ell_max=int(ell_max_theory)
    )

    sources = {
        "Planck": {"TT": (n_tt_planck, dell_tt_planck, fsky_tt_planck), "EE": (n_ee_planck, dell_ee_planck, fsky_ee_planck)},
        "LiteBIRD": {"TT": (n_tt_lb, dell_tt_lb, fsky_tt_lb), "EE": (n_ee_lb, dell_ee_lb, fsky_ee_lb)},
        "SO SAT": {"TT": (n_tt_sat, dell_tt_sat, fsky_tt_sat), "EE": (n_ee_sat, dell_ee_sat, fsky_ee_sat)},
        "SO LAT": {"TT": (n_tt_lat, dell_tt_lat, fsky_tt_lat), "EE": (n_ee_lat, dell_ee_lat, fsky_ee_lat)},
    }

    # TE availability:
    # Even if the input bands do not explicitly include TE, we can still build an
    # approximate TE covariance from the inferred TT and EE noises, as long as
    # both T and E are available (this is what `_select_te` uses).
    has_te = {
        "Planck": bool(n_tt_planck) and bool(n_ee_planck),
        "LiteBIRD": bool(n_tt_lb) and bool(n_ee_lb),
        "SO SAT": bool(n_tt_sat) and bool(n_ee_sat),
        "SO LAT": bool(n_tt_lat) and bool(n_ee_lat),
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
            c = float(np.asarray(cls_cov[spec])[int(l)])

            if str(ell_by_ell_policy) == "min_noise":
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
                var[i] = (
                    2.0
                    / ((2.0 * float(l) + 1.0) * float(best_fsky) * float(best_dell))
                    * (c + float(best_n)) ** 2
                )
            else:
                # "inv_var": treat experiments as independent measurements of the
                # same C_ell and inverse-variance combine their bandpower errors.
                inv_var_sum = 0.0
                n_contrib = 0
                for name, src in sources.items():
                    n_map, dell_map, fsky_map = src.get(spec, ({}, {}, {}))
                    if l not in n_map:
                        continue
                    n = float(n_map[l])
                    dell = float(dell_map.get(l, 1))
                    fsky = float(fsky_map.get(l, 1.0))
                    v = 2.0 / ((2.0 * float(l) + 1.0) * fsky * dell) * (c + n) ** 2
                    if np.isfinite(v) and v > 0:
                        inv_var_sum += 1.0 / float(v)
                        n_contrib += 1
                if n_contrib == 0 or inv_var_sum <= 0 or not np.isfinite(inv_var_sum):
                    var[i] = np.nan
                    continue
                counts["__ncontrib__"] = counts.get("__ncontrib__", 0) + n_contrib
                var[i] = 1.0 / inv_var_sum
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
            c_te = float(np.asarray(cls_cov["TE"])[int(l)])
            c_tt = float(np.asarray(cls_cov["TT"])[int(l)])
            c_ee = float(np.asarray(cls_cov["EE"])[int(l)])

            if str(ell_by_ell_policy) == "min_noise":
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
                var[i] = (
                    1.0
                    / ((2.0 * float(l) + 1.0) * float(best_fsky) * float(best_dell))
                    * (c_te**2 + (c_tt + float(best_ntt)) * (c_ee + float(best_nee)))
                )
            else:
                inv_var_sum = 0.0
                n_contrib = 0
                for name, src in sources.items():
                    if not has_te.get(name, False):
                        continue
                    ntt_map, dell_tt, fsky_tt = src["TT"]
                    nee_map, dell_ee, fsky_ee = src["EE"]
                    if (l not in ntt_map) or (l not in nee_map):
                        continue
                    ntt = float(ntt_map[l])
                    nee = float(nee_map[l])
                    dell = float(min(float(dell_tt.get(l, 1)), float(dell_ee.get(l, 1))))
                    fsky = float(min(float(fsky_tt.get(l, 1.0)), float(fsky_ee.get(l, 1.0))))
                    v = (
                        1.0
                        / ((2.0 * float(l) + 1.0) * fsky * dell)
                        * (c_te**2 + (c_tt + ntt) * (c_ee + nee))
                    )
                    if np.isfinite(v) and v > 0:
                        inv_var_sum += 1.0 / float(v)
                        n_contrib += 1
                if n_contrib == 0 or inv_var_sum <= 0 or not np.isfinite(inv_var_sum):
                    var[i] = np.nan
                    continue
                counts["__ncontrib__"] = counts.get("__ncontrib__", 0) + n_contrib
                var[i] = 1.0 / inv_var_sum
        mask = np.isfinite(var)
        return ell_sorted[mask], var[mask], counts

    effective_covariance_is_diagonal_flag = False
    if str(ell_by_ell_policy) != "min_diag_cov_common_bins":
        ell_tt, cov_tt, counts_tt = _select_auto("TT")
        ell_ee, cov_ee, counts_ee = _select_auto("EE")
        ell_te, cov_te, counts_te = _select_te()

        owner_bins = None
        owner_map = None
        common_bins_by_spec = None
    else:
        # Conservative, common-bin ownership selection based on diagonal covariances.
        if str(ell_by_ell_score) != "sigma_over_sqrt_dell":
            raise ValueError(
                f"Unsupported ell_by_ell_score={ell_by_ell_score!r} (expected 'sigma_over_sqrt_dell')."
            )

        exp_names = ["Planck", "SO SAT", "SO LAT", "LiteBIRD"]
        bands_by_exp_spec: dict[str, dict[str, tuple[Sequence[Any], float | None]]] = {
            "Planck": {"TT": (planck_bands, 0.7), "TE": (planck_bands, 0.7), "EE": (planck_bands, 0.7)},
            "LiteBIRD": {"TT": (litebird_bands, 0.7), "TE": (litebird_bands, 0.7), "EE": (litebird_bands, 0.7)},
            "SO SAT": {"TT": (so_sat, 0.1), "TE": (so_sat, 0.1), "EE": (so_sat, 0.1)},
            # For SO LAT, use LAT bands for TT/TE and LAT_pol for EE.
            "SO LAT": {"TT": (so_lat_tt, 0.4), "TE": (so_lat_tt, 0.4), "EE": (so_lat_ee, 0.4)},
        }

        common_bins_by_spec = {}
        owner_bins = {}
        owner_map = {}
        counts_tt = {k: 0 for k in exp_names}
        counts_te = {k: 0 for k in exp_names}
        counts_ee = {k: 0 for k in exp_names}

        effective_covariance_is_diagonal = False

        def _do_spec(spec: str) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, Any]]:
            """
            Returns (ell_eff_int, cov_diag_for_bands, counts, diag_meta).

            Also stores full covariance selection details in diag_meta for later Fisher build.
            """
            entries_by_exp = {}
            for name in exp_names:
                bands_here, fsky_ovr = bands_by_exp_spec[name][spec]
                entries_by_exp[name] = _band_entries_for_spec(
                    bands=bands_here, cell_type=spec, fsky_override=float(fsky_ovr) if fsky_ovr is not None else None
                )

            common_bins = _build_common_bins_for_spec(
                entries_by_exp=entries_by_exp, common_bins=ell_by_ell_common_bins, dell=int(ell_by_ell_common_dell)
            )
            common_bins_by_spec[spec] = common_bins
            if not common_bins:
                return np.array([], dtype=int), np.array([], dtype=float), {k: 0 for k in exp_names}, {}

            dell_common = np.array([float(b - a + 1) for (a, b) in common_bins], dtype=float)

            # Build rebinned covariance matrices per experiment (full if available).
            cov_common_by_exp: dict[str, np.ndarray] = {}
            score_by_exp: dict[str, np.ndarray] = {}
            score_smooth_by_exp: dict[str, np.ndarray] = {}
            ell_eff_common = np.array([(a + b) / 2.0 for (a, b) in common_bins], dtype=float)
            diag_cov_available_by_exp: dict[str, bool] = {}
            used_diagonal_by_exp: dict[str, bool] = {}

            for name in exp_names:
                bands_here, fsky_ovr = bands_by_exp_spec[name][spec]
                ell0, cov0, dell0, fsky0, is_diag0 = _cov_matrix_from_bands(
                    bands=bands_here, cell_type=spec, fsky_override=float(fsky_ovr) if fsky_ovr is not None else None
                )
                used_diagonal_by_exp[name] = bool(is_diag0)
                diag_cov_available_by_exp[name] = (len(ell0) > 0)
                if len(ell0) == 0:
                    cov_common_by_exp[name] = np.full((len(common_bins), len(common_bins)), np.nan, dtype=float)
                    score_by_exp[name] = np.full(len(common_bins), np.nan, dtype=float)
                    continue
                ell_eff_i, cov_common, covfrac_i, _W = _rebin_cov_matrix_to_common_bins(
                    ell_centers=ell0,
                    cov=cov0,
                    dell=int(dell0),
                    common_bins=common_bins,
                    coverage_fraction_min=0.8,
                )
                ell_eff_common = ell_eff_i
                cov_common_by_exp[name] = cov_common
                var_diag = np.diag(cov_common).astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    score = np.sqrt(var_diag) / np.sqrt(dell_common)
                score_by_exp[name] = score

            # Smooth log(score) for ownership decisions only.
            for name in exp_names:
                s = np.asarray(score_by_exp[name], dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ls = np.log(s)
                ls_s = _rolling_median(ls, window=int(ell_by_ell_score_smooth_window))
                score_smooth_by_exp[name] = np.exp(ls_s)

            owners = _select_owners_with_hysteresis(
                scores_by_exp=score_smooth_by_exp,
                exp_names=exp_names,
                hysteresis=float(ell_by_ell_hysteresis),
                min_segment_len=int(ell_by_ell_min_segment_len),
            )
            owner_map[spec] = owners

            # Build effective covariance: preserve within-owner sub-blocks if available;
            # cross-owner covariance set to 0 (conservative wrt unknown cross-owner correlations).
            nbin = len(common_bins)
            cov_eff = np.zeros((nbin, nbin), dtype=float)
            counts = {k: 0 for k in exp_names}
            bins_meta = []
            for i, (bmin, bmax) in enumerate(common_bins):
                owner = owners[i] if i < len(owners) else ""
                # If owner missing for this bin, fall back to best available.
                if owner == "" or not np.isfinite(float(score_smooth_by_exp.get(owner, np.array([np.nan]))[i])):
                    best = None
                    best_s = None
                    for nm in exp_names:
                        ss = float(score_smooth_by_exp[nm][i])
                        if not np.isfinite(ss):
                            continue
                        if best_s is None or ss < best_s:
                            best_s = ss
                            best = nm
                    owner = best or owner
                if owner in counts:
                    counts[owner] += 1
                bins_meta.append(
                    {
                        "ell_min": int(bmin),
                        "ell_max": int(bmax),
                        "ell_eff": float(ell_eff_common[i]),
                        "owner": str(owner),
                        "score_by_exp": {k: float(score_by_exp[k][i]) for k in exp_names},
                        "score_smooth_by_exp": {k: float(score_smooth_by_exp[k][i]) for k in exp_names},
                        "owner_cov_is_diagonal": bool(used_diagonal_by_exp.get(owner, True)),
                    }
                )

            # Fill cov_eff by owner blocks.
            for i in range(nbin):
                oi = owners[i] if i < len(owners) else ""
                for j in range(nbin):
                    oj = owners[j] if j < len(owners) else ""
                    if oi == "" or oj == "" or oi != oj:
                        continue
                    cmat = cov_common_by_exp.get(oi)
                    if cmat is None:
                        continue
                    val = float(cmat[i, j])
                    if np.isfinite(val):
                        cov_eff[i, j] = val

            owner_bins[spec] = bins_meta

            # Diagnostics band (diagonal only) for plotting/compat: use diag(cov_eff).
            var_eff = np.diag(cov_eff).astype(float)
            mask = np.isfinite(var_eff) & (var_eff > 0)
            ell_out = np.asarray(ell_eff_common, dtype=float)[mask].astype(int)
            cov_diag_out = np.asarray(var_eff[mask], dtype=float)

            diag_meta = {
                "common_bins": common_bins,
                "ell_eff_common": np.asarray(ell_eff_common, dtype=float),
                "owners": list(owners),
                "cov_eff": cov_eff,
                "mask": np.asarray(mask, dtype=bool),
                "cov_common_by_exp": cov_common_by_exp,
                "used_diagonal_by_exp": used_diagonal_by_exp,
            }
            return ell_out, cov_diag_out, counts, diag_meta

        ell_tt, cov_tt, counts_tt, _meta_tt = _do_spec("TT")
        ell_te, cov_te, counts_te, _meta_te = _do_spec("TE")
        ell_ee, cov_ee, counts_ee, _meta_ee = _do_spec("EE")

        # Determine whether any spec fell back to diagonal-only covariance for all owners.
        # (We treat "diagonal-only" as used whenever at least one owner lacks full cov.)
        effective_covariance_is_diagonal = False
        for _m in [_meta_tt, _meta_te, _meta_ee]:
            used_diag = (_m.get("used_diagonal_by_exp") or {}) if isinstance(_m, dict) else {}
            if any(bool(v) for v in used_diag.values()):
                effective_covariance_is_diagonal = True
        effective_covariance_is_diagonal_flag = bool(effective_covariance_is_diagonal)
        if verbose and effective_covariance_is_diagonal_flag:
            print(
                "[ell_by_ell:min_diag_cov_common_bins] WARNING: at least one selected owner covariance "
                "was diagonal-only; Fisher uses diagonal fallback for those bins."
            )

        # Store for metadata / potential downstream diagnostics.
        owner_bins["_cov_matrices"] = {"TT": _meta_tt, "TE": _meta_te, "EE": _meta_ee}

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

    if str(ell_by_ell_policy) != "min_diag_cov_common_bins":
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
    else:
        # Full-covariance Fisher in common-bin space (per spectrum), using the
        # selected-owner covariance sub-blocks whenever available.
        import numpy as np

        def _perturb(th0: dict[str, float], param: str, delta: float) -> dict[str, float]:
            th = dict(th0)
            if param in scaled10_params:
                th[param] = (1e10 * th0[param] + delta) * 1e-10
            else:
                th[param] = th0[param] + delta
            return th

        # Prepare per-spectrum bin ell lists and covariance matrices (masked).
        cov_meta = (owner_bins or {}).get("_cov_matrices", {}) if isinstance(owner_bins, dict) else {}
        spec_info: dict[str, dict[str, Any]] = {}
        for spec in ["TT", "TE", "EE"]:
            m = cov_meta.get(spec, {}) or {}
            ell_eff = np.asarray(m.get("ell_eff_common", []), dtype=float)
            mask = np.asarray(m.get("mask", []), dtype=bool)
            cov_eff = np.asarray(m.get("cov_eff", np.zeros((0, 0))), dtype=float)
            if len(ell_eff) == 0 or cov_eff.size == 0 or len(mask) != len(ell_eff):
                continue
            ell_sel = np.asarray(ell_eff[mask], dtype=float).astype(int)
            cov_sel = np.asarray(cov_eff[np.ix_(mask, mask)], dtype=float)
            spec_info[spec] = {"ell": ell_sel, "cov": cov_sel}

        npar = len(param_list)
        F = np.zeros((npar, npar), dtype=float)

        # Precompute all derivatives per spectrum.
        D_by_spec: dict[str, np.ndarray] = {spec: np.zeros((npar, len(info["ell"])), dtype=float) for spec, info in spec_info.items()}
        for ip, p in enumerate(param_list):
            if p not in steps:
                raise KeyError(f"No step size provided for parameter '{p}'")
            step = float(steps[p])
            th_hi = _perturb(theta_fid, p, +step)
            th_lo = _perturb(theta_fid, p, -step)
            cls_hi = _cls_provider(lmax=int(ell_max_theory), **th_hi)
            cls_lo = _cls_provider(lmax=int(ell_max_theory), **th_lo)
            for spec, info in spec_info.items():
                ell_sel = info["ell"]
                arr_hi = np.asarray(cls_hi[spec], dtype=float)
                arr_lo = np.asarray(cls_lo[spec], dtype=float)
                for k, e in enumerate(ell_sel):
                    D_by_spec[spec][ip, k] = (float(arr_hi[int(e)]) - float(arr_lo[int(e)])) / (2.0 * step)

        # Accumulate Fisher blocks per spectrum.
        # Keep a local note; exported in metadata below.
        # (True if we had to fall back to diagonal-only covariance anywhere.)
        # effective_covariance_is_diagonal_flag is set earlier in this policy branch.
        n_pinv = 0
        conds: list[float] = []
        for spec, info in spec_info.items():
            cov = np.asarray(info["cov"], dtype=float)
            if cov.shape[0] == 0:
                continue
            try:
                cond = float(np.linalg.cond(cov))
            except Exception:
                cond = float("inf")
            conds.append(cond)
            try:
                cov_inv = np.linalg.inv(cov)
            except Exception:
                cov_inv = np.linalg.pinv(cov)
                n_pinv += 1
            d = D_by_spec[spec]  # (npar, nbins)
            F += d @ cov_inv @ d.T

        F = 0.5 * (F + F.T)
        Cov_params = np.linalg.pinv(F)
        sigma = np.sqrt(np.diag(Cov_params))
        fisher = fisher_multi.FisherResult(
            F=F,
            Cov_params=Cov_params,
            sigma=sigma,
            dC=[np.zeros(0, dtype=float) for _ in range(npar)],
            bands=list(bands_eff),
            param_list=list(param_list),
        )
        # Attach some solver diagnostics into metadata below.
        if isinstance(getattr(fisher, "metadata", None), dict):
            pass
    fisher.metadata = {
        "combination_mode": "ell_by_ell",
        "ell_by_ell_policy": ell_by_ell_policy,
        "ell_by_ell_score": ell_by_ell_score,
        "ell_by_ell_common_dell": int(ell_by_ell_common_dell),
        "ell_by_ell_hysteresis": float(ell_by_ell_hysteresis),
        "ell_by_ell_min_segment_len": int(ell_by_ell_min_segment_len),
        "ell_by_ell_score_smooth_window": int(ell_by_ell_score_smooth_window),
        "use_iso_signal_in_cov": bool(use_iso_signal_in_cov),
        "iso_mode": iso_mode,
        "counts": {"TT": counts_tt, "TE": counts_te, "EE": counts_ee},
        "n_ell": {"TT": int(len(ell_tt)), "TE": int(len(ell_te)), "EE": int(len(ell_ee))},
    }
    if str(ell_by_ell_policy) == "min_diag_cov_common_bins":
        fisher.metadata["owner_map"] = owner_map or {}
        fisher.metadata["owner_bins"] = owner_bins or {}
        fisher.metadata["ell_by_ell_common_bins"] = common_bins_by_spec or {}
        fisher.metadata["effective_covariance_is_diagonal"] = bool(effective_covariance_is_diagonal_flag)

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
    ell_by_ell_common_bins: Sequence[tuple[int, int]] | None = None,
    ell_by_ell_common_dell: int = 20,
    ell_by_ell_score: str = "sigma_over_sqrt_dell",
    ell_by_ell_hysteresis: float = 1.05,
    ell_by_ell_min_segment_len: int = 2,
    ell_by_ell_score_smooth_window: int = 3,
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
            ell_by_ell_common_dell=int(ell_by_ell_common_dell),
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
                ell_by_ell_common_bins=ell_by_ell_common_bins,
                ell_by_ell_common_dell=int(ell_by_ell_common_dell),
                ell_by_ell_score=str(ell_by_ell_score),
                ell_by_ell_hysteresis=float(ell_by_ell_hysteresis),
                ell_by_ell_min_segment_len=int(ell_by_ell_min_segment_len),
                ell_by_ell_score_smooth_window=int(ell_by_ell_score_smooth_window),
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

        # Ensure Planck lowE/tau-prior handling is applied to the *final* Fisher,
        # including joint_cov and ell_by_ell outputs built from bands.
        unified = apply_planck_lowE_tau_prior(
            unified,
            planck_likelihood_mode=planck_likelihood_mode,
            tau_prior_sigma=planck_tau_prior_sigma,
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
                    ell_by_ell_common_bins=ell_by_ell_common_bins,
                    ell_by_ell_common_dell=int(ell_by_ell_common_dell),
                    ell_by_ell_score=str(ell_by_ell_score),
                    ell_by_ell_hysteresis=float(ell_by_ell_hysteresis),
                    ell_by_ell_min_segment_len=int(ell_by_ell_min_segment_len),
                    ell_by_ell_score_smooth_window=int(ell_by_ell_score_smooth_window),
                    verbose=False,
                )
                pk_so_lb = unified
            else:
                raise ValueError(f"Unknown combination_mode={combination_mode!r}")

            pk_so = apply_planck_lowE_tau_prior(
                pk_so,
                planck_likelihood_mode=planck_likelihood_mode,
                tau_prior_sigma=planck_tau_prior_sigma,
            )

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


def add_gaussian_prior_to_fisher(
    fisher_obj: Any,
    param_name: str = "tau_reio",
    sigma: float = 0.007,
) -> Any:
    """
    Add a Gaussian prior to a Fisher object.

    This adds 1/sigma^2 to the diagonal Fisher element for `param_name` and
    refreshes covariance / marginalized sigmas when possible.

    Returns a (typically new) Fisher object. If `param_name` is not present,
    returns the input object unchanged.
    """
    import numpy as np

    if fisher_obj is None:
        return fisher_obj

    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Invalid prior sigma={sigma!r}; expected a finite positive float.")

    param_list = getattr(fisher_obj, "param_list", None)
    if not param_list:
        return fisher_obj
    if param_name not in param_list:
        return fisher_obj

    # Prefer the object's native API when available.
    with_prior = getattr(fisher_obj, "with_prior", None)
    if callable(with_prior):
        return with_prior({str(param_name): float(sigma)})

    # Fallback for Fisher-like objects.
    F = getattr(fisher_obj, "F", None)
    if F is None:
        return fisher_obj
    F2 = np.array(F, dtype=float, copy=True)
    i = list(param_list).index(param_name)
    F2[i, i] += 1.0 / (sigma**2)

    Cov = np.linalg.pinv(F2)
    sig = np.sqrt(np.diag(Cov))

    # Try to preserve type if it looks like a `FisherResult`.
    try:
        return type(fisher_obj)(
            F=F2,
            Cov_params=Cov,
            sigma=sig,
            dC=getattr(fisher_obj, "dC", []),
            bands=getattr(fisher_obj, "bands", []),
            param_list=list(param_list),
            metadata=getattr(fisher_obj, "metadata", {}) or {},
        )
    except Exception:
        # Last resort: update in-place if the object is mutable.
        try:
            fisher_obj.F = F2
            if hasattr(fisher_obj, "Cov_params"):
                fisher_obj.Cov_params = Cov
            if hasattr(fisher_obj, "sigma"):
                fisher_obj.sigma = sig
        except Exception:
            pass
        return fisher_obj


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

    Notes
    -----
    This helper is intentionally idempotent (it will not apply the same tau
    prior twice) when the Fisher object's metadata indicates the prior has
    already been added.
    """
    if planck_likelihood_mode == "pliklite_lowT":
        return fisher
    if planck_likelihood_mode != "TTTEEE_lowE":
        raise ValueError(
            f"Unknown planck_likelihood_mode={planck_likelihood_mode!r} (expected 'pliklite_lowT' or 'TTTEEE_lowE')."
        )
    if "tau_reio" not in fisher.param_list:
        return fisher

    meta = getattr(fisher, "metadata", None)
    if isinstance(meta, dict):
        gp = meta.get("gaussian_priors", {}) or {}
        existing = gp.get("tau_reio", None)
        if existing is not None and abs(float(existing) - float(tau_prior_sigma)) < 1e-15:
            return fisher
        if meta.get("planck_lowE") == "approximated_by_tau_prior" and abs(
            float(meta.get("planck_tau_prior_sigma", tau_prior_sigma)) - float(tau_prior_sigma)
        ) < 1e-15:
            return fisher

    out = add_gaussian_prior_to_fisher(fisher, param_name="tau_reio", sigma=float(tau_prior_sigma))
    out_meta = dict(getattr(out, "metadata", {}) or {}) if isinstance(getattr(out, "metadata", None), dict) else {}
    gp = dict(out_meta.get("gaussian_priors", {}) or {})
    gp["tau_reio"] = float(tau_prior_sigma)
    out_meta["gaussian_priors"] = gp
    out_meta |= {
        "planck_likelihood_mode": planck_likelihood_mode,
        "planck_lowE": "approximated_by_tau_prior",
        "planck_tau_prior_sigma": float(tau_prior_sigma),
    }
    out.metadata = out_meta
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

                # Include LAT_cross so TE bands are generated for SO as well.
                so_bands = fisher_multi.parse_spectrum_bands(full_lik_cov, ["SAT", "LAT", "LAT_pol", "LAT_cross"])
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
