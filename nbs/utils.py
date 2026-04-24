from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from typing import Any


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
    return "ad" if iso is None else str(iso)


def _maybe_skip(path: Path, skip_existing: bool) -> bool:
    return skip_existing and path.exists()


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
    out_base: str | Path = "images/unified_constraints",
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
    """
    scaled10_params = scaled10_params or {"P_RR_1", "P_RR_2", "P_II_1", "P_II_2"}
    out_base = Path(out_base)
    os.makedirs(out_base, exist_ok=True)
    label_map = label_map or LATEX_LABELS

    for corr in corr_plot:
        theta0 = theta_full[corr]["theta0"]
        for iso in iso_plot:
            iso_key = _iso_tag(iso)
            pk = so_fisher_all[corr]["PK_Lite"][iso][year_tag]
            so = so_fisher_all[corr]["SO"][iso][year_tag]
            lb = muci[corr]["LB"][iso][year_tag]

            unified = pk.combine(so).combine(lb)

            out_dir = out_base / year_tag / corr / iso_key
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
                regular_params = ["omega_b", "omega_cdm", "h", "tau_reio"]
                scaled_params = ["P_RR_1", "P_RR_2", "P_II_1", "P_II_2"]
                triangle_overlay(
                    [(pk, f"Planck {year_tag}"), (so, f"SO {year_tag}"), (lb, f"LiteBIRD {year_tag}")],
                    theta0=theta0,
                    subset=regular_params,
                    scaled10_params=set(),
                    save_path=out_dir / f"overlay_regular_{year_tag}.pdf",
                    label_map=label_map,
                    skip_existing=skip_existing,
                )
                triangle_overlay(
                    [(pk, f"Planck {year_tag}"), (so, f"SO {year_tag}"), (lb, f"LiteBIRD {year_tag}")],
                    theta0=theta0,
                    subset=scaled_params,
                    scaled10_params=scaled10_params,
                    save_path=out_dir / f"overlay_scaled_{year_tag}.pdf",
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


def load_planck_pliklite_data(
    planck_root: str | Path = "/home/sa5705/scratch/scratch/cobaya_packages/data/planck_2018_pliklite_native/",
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

    # Optional: prepend low-ell bins if helper is present.
    try:
        from prepend_low_ell import prepend_planck_low_ell

        lite_data = prepend_planck_low_ell(lite_data)
    except Exception:
        pass

    return lite_data


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
    steps_scaled10 = {
        "P_RR_1": 0.01,
        "P_RR_2": 0.01,
        "P_II_1": 0.05,
        "P_II_2": 0.05,
    }
    scaled10_params = {"P_RR_1", "P_RR_2", "P_II_1", "P_II_2"}
    param_list = [
        "omega_b",
        "omega_cdm",
        "h",
        "tau_reio",
        "P_RR_1",
        "P_RR_2",
        "P_II_1",
        "P_II_2",
    ]
    steps = {**steps_abs, **steps_scaled10}

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

            def _cls_provider(*, lmax: int = lmax, **th):
                # fisher_multi will pass `lmax=...`; accept it and forward.
                return _compute_cls_indexed(compute_cls, lmax=int(lmax), iso_mode=iso_mode, **th)

            # Planck-only Fisher (year-independent)
            # Raw theory for building SO mock "data" (expects arrays aligned with `ell`)
            cmb_theo_raw = compute_cls(lmax=lmax, iso_mode=iso_mode, **theta0, **corr_dict)
            fisher_pk = fisher_multi.fisher_forecast(
                theta0={**theta0, **corr_dict},
                param_list=param_list,
                bands=planck_bands,
                compute_cls=_cls_provider,
                steps=steps,
                scaled_params=scaled10_params,
                ell_max=lmax,
                use_pinv=True,
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
                cmb_theo = cmb_theo_raw

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
                    theta0={**theta0, **corr_dict},
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
    steps_scaled10 = {"P_RR_1": 0.01, "P_RR_2": 0.01, "P_II_1": 0.05, "P_II_2": 0.05}
    scaled10_params = {"P_RR_1", "P_RR_2", "P_II_1", "P_II_2"}
    param_list = [
        "omega_b",
        "omega_cdm",
        "h",
        "tau_reio",
        "P_RR_1",
        "P_RR_2",
        "P_II_1",
        "P_II_2",
    ]
    steps = {**steps_abs, **steps_scaled10}

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
                return _compute_cls_indexed(compute_cls, lmax=int(lmax), iso_mode=iso_mode, **th)

            # Fiducial theory for covariance (just needs to cover ell_bins)
            cls0 = _compute_cls_indexed(compute_cls, lmax=ell_max_theory, iso_mode=iso_mode, **theta0, **corr_dict)
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
                theta0={**theta0, **corr_dict},
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
