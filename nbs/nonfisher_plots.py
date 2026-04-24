from __future__ import annotations

from pathlib import Path
from typing import Any


def has_classy() -> bool:
    try:
        import classy  # noqa: F401

        return True
    except Exception:
        return False


def _maybe_skip(path: Path, skip_existing: bool) -> bool:
    return skip_existing and path.exists()


def _infer_plancklite_noise_from_cov(lite_data: dict, *, fsky_lite: float = 0.7) -> dict[str, dict[str, Any]]:
    """
    Infer an effective N_ell from the diagonal covariance + C_ell using
    a Knox-like prefactor, matching the approach in the original notebook.
    """
    import numpy as np

    out: dict[str, dict[str, Any]] = {}

    # TT / EE (auto)
    for pol, key in [("TT", "PK_lite"), ("EE", "PK_lite_pol")]:
        c_ell = lite_data[pol]["c_ell"]
        cov = lite_data[pol]["cov"]
        ell = lite_data[pol]["ell"]
        dell = lite_data[pol]["dell"]
        covpref = 2 / (2 * ell + 1) / fsky_lite / dell
        n_ell = np.sqrt(cov / covpref) - c_ell
        out[key] = {"c_ell": c_ell, "n_ell": n_ell, "cov": cov, "ell": ell, "dell": dell}

    # TE (cross) – we don't try to infer it robustly; keep as zero noise for plotting
    pol = "TE"
    out["PK_lite_cross"] = {
        "c_ell": lite_data[pol]["c_ell"],
        "n_ell": lite_data[pol]["c_ell"] * 0,
        "cov": lite_data[pol]["cov"],
        "ell": lite_data[pol]["ell"],
        "dell": lite_data[pol]["dell"],
    }
    return out


def build_full_lik_data_with_noise(
    req_spec: dict,
    setup: dict,
    full_noise_dict: dict,
    cmb_theo_dict: dict,
) -> dict:
    """
    Minimal version of the older notebook `build_full_lik_data`, but returns
    both `c_ell` and `n_ell` separately.
    """
    from copy import deepcopy
    import numpy as np

    def _interp_to(x_new, x_old, y_old):
        return np.interp(x_new, x_old, y_old)

    def _apply_cuts(powspec, ells, cuts):
        lmin, lmax = cuts
        mask = (ells >= lmin) & (ells <= lmax)
        return ells[mask], powspec[mask]

    def _make_cross_noise(cell_nz_1, cell_nz_2):
        return np.sqrt(cell_nz_1 * cell_nz_2)

    out = {"metadata": {}, "data": {}}

    for exp_key in req_spec:
        check_cross = exp_key.endswith("_cross") or exp_key == "PK_cross"
        setup_exp = setup.get(exp_key, {})

        if exp_key == "SAT":
            case_key = f"{exp_key}_y{setup_exp['yrs']:.0f}_sm{setup_exp['sens_mode']:.0f}fm{setup_exp['f_mode']:.0f}"
            c_ell_theo = cmb_theo_dict["EE"]
            cell_type = "EE"
        elif exp_key == "LAT":
            case_key = f"{exp_key}_y{setup_exp['yrs']:.0f}_sm{setup_exp['sens_mode']:.0f}fm{setup_exp['f_mode']:.0f}"
            c_ell_theo = cmb_theo_dict["TT"]
            cell_type = "TT"
        elif exp_key == "LAT_pol":
            case_key = f"{exp_key}_y{setup_exp['yrs']:.0f}_sm{setup_exp['sens_mode']:.0f}fm{setup_exp['f_mode']:.0f}"
            c_ell_theo = cmb_theo_dict["EE"]
            cell_type = "EE"
        elif exp_key == "LAT_cross":
            case_key = f"{exp_key}_y{setup_exp['yrs']:.0f}_sm{setup_exp['sens_mode']:.0f}fm{setup_exp['f_mode']:.0f}"
            c_ell_theo = cmb_theo_dict["TE"]
            cell_type = "TE"
            check_cross = True
        else:
            raise KeyError(f"Unsupported exp_key: {exp_key}")

        meta = deepcopy(full_noise_dict[case_key])
        nz_dict = {} if check_cross else deepcopy(meta["nz_dict"])

        out["metadata"][exp_key] = deepcopy(meta)
        out["metadata"][exp_key].pop("nz_dict", None)
        out["metadata"][exp_key].pop("ell_nz", None)
        out["metadata"][exp_key]["cell_type"] = cell_type
        out["data"][exp_key] = {}

        for spec_key in req_spec[exp_key]:
            left, right = spec_key.split("x")
            f1 = left.split("_")[-1]
            f2 = right.split("_")[-1]

            ell_new = meta["ell_nz"]
            cuts = meta["cuts"]
            ell_old_cut, cell_cmb_cut = _apply_cuts(c_ell_theo, cmb_theo_dict["ell"], cuts)
            cell_cmb_cut = _interp_to(ell_new, ell_old_cut, cell_cmb_cut)
            ell_new_cut, cell_cmb_cut = _apply_cuts(cell_cmb_cut, ell_new, cuts)

            if check_cross:
                nz_arr_cut = cell_cmb_cut * 0
            else:
                if f1 == f2:
                    nz_arr = nz_dict[f"f{f1}"]
                else:
                    nz_arr = _make_cross_noise(nz_dict[f"f{f1}"], nz_dict[f"f{f2}"])
                _, nz_arr_cut = _apply_cuts(nz_arr, ell_new, cuts)

            out["data"][exp_key][spec_key] = {"c_ell": cell_cmb_cut, "n_ell": nz_arr_cut, "ell": ell_new_cut}

    return out


def make_unified_noise_simple(full_lik_data: dict) -> dict:
    """
    Collapse each experiment's multiple spectra into a single (c_ell, n_ell, cov).

    This assumes all spectra for a given experiment share the same ell grid (true for
    our SO noise dict construction).
    """
    import numpy as np
    from cosmocast_makelik.iso_theory import knox_auto_cov, knox_cross_cov

    out = {"metadata": full_lik_data["metadata"], "data": {}}

    for exp_key, spectra in full_lik_data["data"].items():
        if not spectra:
            continue
        first = next(iter(spectra.values()))
        ell = first["ell"]
        c_ell = first["c_ell"]

        inv_sum = np.zeros_like(c_ell, dtype=float)
        for s in spectra.values():
            n = s["n_ell"]
            if np.all(n == 0):
                continue
            inv_sum += 1.0 / n
        n_ell = (1.0 / inv_sum) if np.all(inv_sum != 0) else c_ell * 0

        meta = full_lik_data["metadata"][exp_key]
        fsky = meta["fsky"]
        dell = meta["dell"]

        cell_type = meta["cell_type"]
        if cell_type in ("TT", "EE"):
            cov = knox_auto_cov(c_ell + n_ell, ell, dell, fsky)
        else:
            cov = (c_ell * 0) + 1.0

        out["data"][exp_key] = {"c_ell": c_ell, "n_ell": n_ell, "cov": cov, "ell": ell, "dell": dell}

    # TE covariance needs TT+EE totals; handle LAT_cross specifically
    if "LAT_cross" in out["data"] and "LAT" in out["data"] and "LAT_pol" in out["data"]:
        te = out["data"]["LAT_cross"]
        tt = out["data"]["LAT"]
        ee = out["data"]["LAT_pol"]
        te["cov"] = knox_cross_cov(
            te["c_ell"],
            tt["c_ell"] + tt["n_ell"],
            ee["c_ell"] + ee["n_ell"],
            te["ell"],
            te["dell"],
            out["metadata"]["LAT_cross"]["fsky"],
        )

    return out


def compute_and_save_noise_and_cov_plots(
    *,
    out_dir: str | Path = "images/non_fisher",
    year: int = 5,
    fsky_sat: float = 0.1,
    fsky_lat: float = 0.4,
    fsky_plancklite: float = 0.7,
    sat_lmin: int = 30,
    sat_lmax: int = 400,
    lat_lmin: int = 400,
    lat_lmax: int = 3000,
    dell: int = 10,
    skip_existing: bool = True,
) -> None:
    """
    Save additional non-Fisher plots:
    - Unified noise curves (TT and EE)
    - Diagonal covariance diagnostic plots (TT, EE, TE)

    Requires `classy` (CLASS) + `cobaya` (for PlikLite loading).
    """
    if not has_classy():
        print("Skipping noise/cov plots: `classy` is not installed.")
        return

    import numpy as np
    import matplotlib.pyplot as plt
    from copy import deepcopy

    import utils
    import noise_calc_modified as noise_calc
    from cosmocast_makelik.iso_theory import compute_cls
    from cosmocast_makelik.multi_freq_liq.likelihood_multi import (
        frequencies,
        SAT_pairs_all,
        LAT_pairs_all,
        make_nz_dict_from_array,
    )
    from colors import assign_colors

    out_dir = Path(out_dir) / f"y{year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lite_data = utils.load_planck_pliklite_data()
    planck_unif = _infer_plancklite_noise_from_cov(lite_data, fsky_lite=fsky_plancklite)

    theta0 = utils.THETA_FULL["ad"]["theta0"]
    cmb_theo = compute_cls(lmax=4000, iso_mode=None, **theta0)

    # --- LiteBIRD noise (fixed table; intentionally NOT time-scaled) ---
    lb_noise = utils.litebird_noise_curves(lmin=2, lmax=200, dell=10)
    lb_ell = np.asarray(lb_noise["ell"], dtype=int)
    lb_n_tt = np.asarray(lb_noise["n_ell_tt"], dtype=float)
    lb_n_ee = np.asarray(lb_noise["n_ell_ee"], dtype=float)
    # Interp theory onto LB ell bins
    cmb_ell = np.asarray(cmb_theo["ell"], dtype=float)
    lb_c_tt = np.interp(lb_ell, cmb_ell, np.asarray(cmb_theo["TT"], dtype=float))
    lb_c_ee = np.interp(lb_ell, cmb_ell, np.asarray(cmb_theo["EE"], dtype=float))

    # --- build SO noise dict ---
    full_noise_dict: dict[str, Any] = {}

    sat_key = f"SAT_y{year}_sm1fm0"
    full_noise_dict[sat_key] = {
        "yrs": float(year),
        "sens_mode": 1,
        "f_mode": 0,
        "fsky": fsky_sat,
        "lmax": sat_lmax,
        "dell": dell,
        "cuts": [sat_lmin, sat_lmax],
    }
    ell_sa, n_ell_p_sa, _white = noise_calc.Simons_Observatory_V3_SA_noise(
        1, 0, float(year), fsky_sat, sat_lmax, dell, beam_stuff=True
    )
    full_noise_dict[sat_key]["nz_dict"] = make_nz_dict_from_array(n_ell_p_sa, frequencies, "SAT")
    full_noise_dict[sat_key]["ell_nz"] = ell_sa

    lat_key = f"LAT_y{year}_sm1fm0"
    latpol_key = f"LAT_pol_y{year}_sm1fm0"
    latcross_key = f"LAT_cross_y{year}_sm1fm0"
    for k in (lat_key, latpol_key, latcross_key):
        full_noise_dict[k] = {
            "yrs": float(year),
            "sens_mode": 1,
            "f_mode": 0,
            "fsky": fsky_lat,
            "lmax": lat_lmax,
            "dell": dell,
            "cuts": [lat_lmin, lat_lmax],
        }
    ell_la, n_ell_t_la, n_ell_p_la, _white = noise_calc.Simons_Observatory_V3_LA_noise(
        1, float(year), fsky_lat, lat_lmax, dell
    )
    full_noise_dict[lat_key]["nz_dict"] = make_nz_dict_from_array(n_ell_t_la, frequencies, "LAT")
    full_noise_dict[lat_key]["ell_nz"] = ell_la
    full_noise_dict[latpol_key]["nz_dict"] = make_nz_dict_from_array(n_ell_p_la, frequencies, "LAT")
    full_noise_dict[latpol_key]["ell_nz"] = ell_la
    full_noise_dict[latcross_key]["nz_dict"] = None
    full_noise_dict[latcross_key]["ell_nz"] = ell_la

    # --- build full_lik_data with separated c_ell and n_ell ---
    req_spec = {
        "SAT": SAT_pairs_all,
        "LAT": LAT_pairs_all,
        "LAT_pol": LAT_pairs_all,
        "LAT_cross": LAT_pairs_all,
    }
    setup = {
        "SAT": {"yrs": float(year), "sens_mode": 1, "f_mode": 0},
        "LAT": {"yrs": float(year), "sens_mode": 1, "f_mode": 0},
        "LAT_pol": {"yrs": float(year), "sens_mode": 1, "f_mode": 0},
        "LAT_cross": {"yrs": float(year), "sens_mode": 1, "f_mode": 0},
    }
    # Note: build_full_lik_data_with_noise expects the case_key to be present in full_noise_dict.
    full_lik_data = build_full_lik_data_with_noise(req_spec, setup, full_noise_dict, cmb_theo)
    so_unif = make_unified_noise_simple(full_lik_data)

    # Merge PlanckLite blocks in (for plotting)
    so_unif["metadata"].update({k: {"fsky": fsky_plancklite} for k in planck_unif.keys()})
    so_unif["data"].update(planck_unif)

    # Add LiteBIRD blocks for plotting (fixed-table noise; no time scaling)
    from cosmocast_makelik.iso_theory import knox_auto_cov

    so_unif["metadata"]["LB"] = {"fsky": 0.7, "dell": 10, "cuts": [2, 200]}
    so_unif["data"]["LB"] = {
        "c_ell": lb_c_ee,
        "n_ell": lb_n_ee,
        "cov": knox_auto_cov(lb_c_ee + lb_n_ee, lb_ell, 10, 0.7),
        "ell": lb_ell,
        "dell": 10,
    }
    so_unif["metadata"]["LB_TT"] = {"fsky": 0.7, "dell": 10, "cuts": [2, 200]}
    so_unif["data"]["LB_TT"] = {
        "c_ell": lb_c_tt,
        "n_ell": lb_n_tt,
        "cov": knox_auto_cov(lb_c_tt + lb_n_tt, lb_ell, 10, 0.7),
        "ell": lb_ell,
        "dell": 10,
    }

    # --- noise curves ---
    ee_path = out_dir / "ee_noise_unified.pdf"
    if not _maybe_skip(ee_path, skip_existing):
        plt.figure()
        plt.plot(so_unif["data"]["SAT"]["ell"], so_unif["data"]["SAT"]["c_ell"], color="black")
        plt.plot(so_unif["data"]["LAT_pol"]["ell"], so_unif["data"]["LAT_pol"]["c_ell"], color="black")
        exps = ["LB", "PK_lite_pol", "LAT_pol", "SAT"]
        colors = assign_colors(exps)
        labels = ["LiteBIRD", "PlanckLite", "LAT", "SAT"]
        for exp, lab in zip(exps, labels):
            plt.plot(
                so_unif["data"][exp]["ell"],
                np.abs(so_unif["data"][exp]["n_ell"]),
                label=lab,
                color=colors[exp],
            )
        plt.ylabel(r"$N_\ell$")
        plt.xlabel(r"$\ell$")
        plt.loglog()
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.savefig(ee_path, bbox_inches="tight")
        plt.close()

    tt_path = out_dir / "tt_noise_unified.pdf"
    if not _maybe_skip(tt_path, skip_existing):
        plt.figure()
        plt.plot(so_unif["data"]["PK_lite"]["ell"], so_unif["data"]["PK_lite"]["c_ell"], color="black")
        plt.plot(so_unif["data"]["LAT"]["ell"], so_unif["data"]["LAT"]["c_ell"], color="black")
        exps = ["PK_lite", "LAT"]
        colors = assign_colors(exps)
        labels = ["PlanckLite", "LAT"]
        for exp, lab in zip(exps, labels):
            plt.plot(
                so_unif["data"][exp]["ell"],
                np.abs(so_unif["data"][exp]["n_ell"]),
                label=lab,
                color=colors[exp],
            )
        plt.ylabel(r"$N_\ell$")
        plt.xlabel(r"$\ell$")
        plt.loglog()
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.savefig(tt_path, bbox_inches="tight")
        plt.close()

    # --- cov-diagonal diagnostics ---
    def _plot_cov_diag(exps: list[str], cell_type: str, out_path: Path):
        plt.figure()
        colors = assign_colors(exps)
        for exp in exps:
            d = so_unif["data"][exp]
            ell = d["ell"]
            cov = d["cov"]
            dell_here = d["dell"]
            # follow the original script: sigma*sqrt(dell) = sqrt(Var * dell)
            err = np.sqrt(cov * dell_here)
            plt.plot(ell, err, label=exp, color=colors.get(exp, None))
        plt.ylabel(r"$\sigma \sqrt{\Delta \ell}$")
        plt.xlabel(r"$\ell$")
        plt.loglog()
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    cov_tt = out_dir / "cov_diag_TT.pdf"
    if not _maybe_skip(cov_tt, skip_existing):
        _plot_cov_diag(["LAT", "PK_lite"], "TT", cov_tt)

    cov_ee = out_dir / "cov_diag_EE.pdf"
    if not _maybe_skip(cov_ee, skip_existing):
        _plot_cov_diag(["LB", "LAT_pol", "SAT", "PK_lite_pol"], "EE", cov_ee)

    cov_te = out_dir / "cov_diag_TE.pdf"
    if not _maybe_skip(cov_te, skip_existing):
        _plot_cov_diag(["LAT_cross", "PK_lite_cross"], "TE", cov_te)
